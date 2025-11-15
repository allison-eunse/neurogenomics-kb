#!/usr/bin/env python3
"""
Automated pass/fail gate for Codex two-cycle workflows.

Runs repository validations, linters, and type checks, then records the results
so you can compare cycle N vs cycle N+1 before handing work back to Codex.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_ENV = "CODEX_GATE_STATE"
PYTHON = os.environ.get("PYTHON", sys.executable)


Status = Literal["pass", "fail", "skip"]


class ResultRecord(TypedDict, total=False):
    name: str
    category: str
    status: Status
    duration_s: float
    command: list[str]
    description: str
    skipped_reason: str


@dataclass(frozen=True)
class Check:
    name: str
    category: str  # tests | lint | typecheck
    command: Sequence[str]
    description: str
    paths: tuple[str, ...] = ()

    def is_impacted(self, changed: set[str] | None) -> bool:
        if changed is None or not self.paths:
            return True
        normalized = {_normalize_path(item) for item in changed}
        for prefix in self.paths:
            target = _normalize_path(prefix)
            if _matches(target, normalized, prefix.endswith("/")):
                return True
        return False


def _normalize_path(path: str) -> str:
    cleaned = path.strip().lstrip("./")
    return cleaned[:-1] if cleaned.endswith("/") else cleaned


def _matches(prefix: str, changed: set[str], is_dir_hint: bool) -> bool:
    for file_path in changed:
        if is_dir_hint:
            if file_path == prefix or file_path.startswith(f"{prefix}/"):
                return True
        else:
            if file_path == prefix:
                return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tests/lints/typechecks as a Codex pass/fail gate."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "fast"),
        default="full",
        help="Full mode includes MkDocs builds; fast mode skips them for a quick signal.",
    )
    parser.add_argument(
        "--since",
        metavar="GIT_REF",
        help="Skip checks if their watched paths did not change since the given git ref.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure instead of running every check.",
    )
    parser.add_argument(
        "--label",
        help="Optional label for this cycle (e.g., cycle1, cycle2). Stored under the state dir.",
    )
    parser.add_argument(
        "--state-dir",
        help="Override location for storing snapshots (default: ~/.cache/codex_gate/<repo>).",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="List configured checks for the selected mode and exit.",
    )
    return parser.parse_args()


def resolve_state_dir(override: str | None) -> Path:
    if override:
        path = Path(override).expanduser()
    else:
        env_path = os.environ.get(DEFAULT_STATE_ENV)
        if env_path:
            path = Path(env_path).expanduser()
        else:
            path = Path.home() / ".cache" / "codex_gate" / ROOT.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def gather_changed_files(ref: str | None) -> set[str] | None:
    if not ref:
        return None
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{ref}..HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("git is not available; skipping change detection.")
        return None
    if result.returncode != 0:
        print(f"Unable to diff against '{ref}'. Running every check.")
        return None
    files = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return files


def build_checks(mode: str) -> list[Check]:
    checks: list[Check] = [
        Check(
            name="validate-model-cards",
            category="tests",
            command=[PYTHON, "scripts/manage_kb.py", "validate", "models"],
            description="Ensure required fields exist on every model card.",
            paths=("kb/model_cards/",),
        ),
        Check(
            name="validate-datasets",
            category="tests",
            command=[PYTHON, "scripts/manage_kb.py", "validate", "datasets"],
            description="Confirm dataset cards reference valid schemas.",
            paths=("kb/datasets/", "docs/data/"),
        ),
        Check(
            name="validate-links",
            category="tests",
            command=[PYTHON, "scripts/manage_kb.py", "validate", "links"],
            description="Check integration cards reference existing datasets/models.",
            paths=("kb/integration_cards/", "kb/model_cards/", "kb/datasets/"),
        ),
        Check(
            name="ci-docs",
            category="tests",
            command=[PYTHON, "scripts/manage_kb.py", "ci", "docs"],
            description="Verify every doc/walkthrough declares a valid model-id marker.",
            paths=("docs/models/", "docs/code_walkthroughs/", "kb/model_cards/"),
        ),
        Check(
            name="ci-weights",
            category="tests",
            command=[PYTHON, "scripts/manage_kb.py", "ci", "weights"],
            description="Ensure weights/license metadata blocks are populated.",
            paths=("kb/model_cards/",),
        ),
    ]
    if mode == "full":
        checks.append(
            Check(
                name="mkdocs-build",
                category="tests",
                command=["mkdocs", "build", "--strict"],
                description="Build the docs site to catch Markdown/frontmatter issues.",
                paths=("docs/", "kb/", "mkdocs.yml"),
            )
        )
    checks.extend(
        [
            Check(
                name="ruff",
                category="lint",
                command=[PYTHON, "-m", "ruff", "check", "scripts"],
                description="Python lint (scripts/).",
                paths=("scripts/",),
            ),
            Check(
                name="mypy",
                category="typecheck",
                command=[PYTHON, "-m", "mypy", "scripts/manage_kb.py", "scripts/codex_gate.py"],
                description="Static type analysis for repo tooling.",
                paths=("scripts/",),
            ),
        ]
    )
    return checks


def list_checks(checks: Iterable[Check]) -> None:
    print("Configured checks:")
    for check in checks:
        path_hint = ", ".join(check.paths) if check.paths else "(always)"
        print(f"- {check.category:9} {check.name:22} :: {check.description} | watched: {path_hint}")


def run_checks(
    checks: list[Check],
    changed: set[str] | None,
    since: str | None,
    fail_fast: bool,
    mode: str,
) -> tuple[list[ResultRecord], bool]:
    results: list[ResultRecord] = []
    failed = False
    print(f"Running {len(checks)} checks (mode={mode}).")
    if since:
        if changed is None:
            print(
                "- Skip detection requested with "
                f"'{since}', but diff failed. Running all checks."
            )
        else:
            print(f"- Skip detection: {len(changed)} files changed since '{since}'.")
    for check in checks:
        if changed is not None and not check.is_impacted(changed):
            record = _record_skip(check, since)
            results.append(record)
            _print_result_line(record)
            continue
        record = _execute_check(check)
        results.append(record)
        _print_result_line(record)
        if record["status"] == "fail":
            failed = True
            if fail_fast:
                break
    return results, failed


def _execute_check(check: Check) -> ResultRecord:
    start = time.perf_counter()
    process = subprocess.run(check.command, cwd=ROOT, check=False)
    duration = time.perf_counter() - start
    status: Status = "pass" if process.returncode == 0 else "fail"
    return {
        "name": check.name,
        "category": check.category,
        "status": status,
        "duration_s": round(duration, 2),
        "command": list(check.command),
        "description": check.description,
    }


def _record_skip(check: Check, since: str | None) -> ResultRecord:
    reason = "untracked files unaffected"
    if since:
        reason = f"no changes vs {since}"
    return {
        "name": check.name,
        "category": check.category,
        "status": "skip",
        "duration_s": 0.0,
        "command": list(check.command),
        "description": check.description,
        "skipped_reason": reason,
    }


def _print_result_line(record: ResultRecord) -> None:
    status = record["status"]
    name = record["name"]
    category = record["category"]
    duration = f"{record['duration_s']:.2f}s"
    extra = ""
    if status == "skip":
        extra = f" ({record.get('skipped_reason', 'skipped')})"
    print(f"[{status.upper():4}] {category:10} {name:24} {duration}{extra}")


def summarize(results: list[ResultRecord]) -> dict[str, int]:
    summary = {"pass": 0, "fail": 0, "skip": 0}
    for record in results:
        status = record["status"]
        summary[status] = summary.get(status, 0) + 1
    print(
        f"Summary: {summary.get('pass', 0)} pass | "
        f"{summary.get('fail', 0)} fail | {summary.get('skip', 0)} skipped"
    )
    return summary


def load_snapshot(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_snapshot(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def diff(previous: dict[str, object] | None, current: dict[str, object]) -> None:
    if not previous:
        return

    def _build_lookup(payload: dict[str, object], key: str) -> dict[str, str]:
        results = payload.get(key)
        lookup: dict[str, str] = {}
        if not isinstance(results, list):
            return lookup
        for record in results:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            status = record.get("status")
            if isinstance(name, str) and isinstance(status, str):
                lookup[name] = status
        return lookup

    previous_lookup = _build_lookup(previous, "results")
    current_lookup = _build_lookup(current, "results")
    regressions = sorted(
        name
        for name, status in current_lookup.items()
        if status == "fail" and previous_lookup.get(name) != "fail"
    )
    fixes = sorted(
        name
        for name, status in current_lookup.items()
        if status == "pass" and previous_lookup.get(name) == "fail"
    )
    if fixes:
        print("Improved since last run: " + ", ".join(fixes))
    if regressions:
        print("Regressions since last run: " + ", ".join(regressions))


def main() -> None:
    args = parse_args()
    checks = build_checks(args.mode)
    if args.list_checks:
        list_checks(checks)
        return
    state_dir = resolve_state_dir(args.state_dir)
    changed = gather_changed_files(args.since)
    results, failed = run_checks(checks, changed, args.since, args.fail_fast, args.mode)
    summary = summarize(results)
    snapshot = {
        "timestamp": datetime.now(UTC).isoformat(),
        "label": args.label,
        "mode": args.mode,
        "since": args.since,
        "summary": summary,
        "results": results,
    }
    latest_path = state_dir / "latest.json"
    previous = load_snapshot(latest_path)
    diff(previous, snapshot)
    write_snapshot(latest_path, snapshot)
    if args.label:
        write_snapshot(state_dir / f"{args.label}.json", snapshot)
    print(f"Stored snapshot in {state_dir}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
