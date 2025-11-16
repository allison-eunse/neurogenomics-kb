#!/usr/bin/env python3
"""Utility CLI for Neurogenomics KB metadata, docs, and CI checks."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from rapidfuzz import fuzz
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
KB_ROOT = ROOT / "kb"
DOCS_ROOT = ROOT / "docs"
MODEL_DIR = KB_ROOT / "model_cards"
DATASET_DIR = KB_ROOT / "datasets"
INTEGRATION_DIR = KB_ROOT / "integration_cards"
RAG_DIR = KB_ROOT / "rag"

console = Console()
app = typer.Typer(
    help="Manage Neurogenomics KB metadata",
    rich_markup_mode="markdown",
)
catalog_app = typer.Typer(
    help="Generate catalogs from YAML metadata",
    rich_markup_mode="markdown",
)
validate_app = typer.Typer(
    help="Validate cards and references",
    rich_markup_mode="markdown",
)
ci_app = typer.Typer(
    help="CI-style checks for docs ↔ metadata consistency",
    rich_markup_mode="markdown",
)
ops_app = typer.Typer(
    help="Helper snippets for embeddings, inference, and walkthrough drafts",
    rich_markup_mode="markdown",
)

app.add_typer(catalog_app, name="catalog")
app.add_typer(validate_app, name="validate")
app.add_typer(ci_app, name="ci")
app.add_typer(ops_app, name="ops")

MODEL_REQUIRED_FIELDS = [
    "id",
    "name",
    "repo",
    "weights",
    "license",
    "context_length",
    "tasks",
]
DATASET_REQUIRED_FIELDS = [
    "id",
    "name",
    "storage_location",
    "required_columns",
    "schema_ref",
    "modalities",
]
MODEL_ID_PATTERN = re.compile(
    r"<!--\s*model-id:\s*([a-z0-9_\-]+)\s*-->",
    re.IGNORECASE,
)

CATALOG_OUT_OPTION = typer.Option(
    None,
    help="Write markdown table to this path",
)
CATALOG_INCLUDE_UNVERIFIED_OPTION = typer.Option(
    True,
    help="Include cards still pending review",
)
DESTINATION_MARKDOWN_OPTION = typer.Option(
    None,
    help="Destination markdown",
)
WALKTHROUGH_OUT_OPTION = typer.Option(
    None,
    help="Destination markdown",
)
WALKTHROUGH_OVERWRITE_OPTION = typer.Option(
    False,
    help="Overwrite existing file",
)
RAG_INDEX_OUT_OPTION = typer.Option(
    RAG_DIR / "index.json",
    help="Output JSON",
)


def _load_cards(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.yaml")):
        if path.stem.lower().startswith("template"):
            continue
        data = yaml.safe_load(path.read_text()) or {}
        items.append((path, data))
    return items


def _ensure_dir(entity: str, path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    console.log(f"[{entity}] writing to {path.relative_to(ROOT)}")
    return path


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def _summarize_tasks(card: dict[str, Any], max_items: int = 3) -> str:
    tasks = card.get("tasks") or []
    if not isinstance(tasks, list):
        return str(tasks)
    summary = ", ".join(tasks[:max_items])
    if len(tasks) > max_items:
        summary += " …"
    return summary or "—"


def _bool_icon(value: bool) -> str:
    return "✅" if value else "⚠️"


def _load_model_index(include_unverified: bool = True) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for path, data in _load_cards(MODEL_DIR):
        data.setdefault("id", path.stem)
        data.setdefault("verified", False)
        if not include_unverified and not data["verified"]:
            continue
        data["_path"] = path
        cards.append(data)
    return cards


def _load_dataset_index() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for path, data in _load_cards(DATASET_DIR):
        data.setdefault("id", path.stem)
        data["_path"] = path
        cards.append(data)
    return cards


def _load_integration_index() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for path, data in _load_cards(INTEGRATION_DIR):
        data.setdefault("id", path.stem)
        data["_path"] = path
        cards.append(data)
    return cards


@catalog_app.command("models")
def catalog_models(
    out: Path = CATALOG_OUT_OPTION,
    include_unverified: bool = CATALOG_INCLUDE_UNVERIFIED_OPTION,
):
    """Render a markdown summary of all model cards."""
    cards = _load_model_index(include_unverified=include_unverified)
    if not cards:
        raise typer.Exit(code=1)
    headers = ["Model", "Domain", "Context", "Verified", "Key tasks", "Tags"]
    rows: list[list[str]] = []
    for card in cards:
        name = f"[{card['name']}]({card['repo']})"
        context = str(card.get("context_length", "?"))
        tags = card.get("tags") or []
        rows.append(
            [
                name,
                card.get("modality", card.get("domain", "?")),
                context,
                _bool_icon(card.get("verified", False)),
                _summarize_tasks(card),
                ", ".join(tags[:4]) or "—",
            ]
        )
    markdown = (
        "# Model catalog\n\n"
        + _markdown_table(headers, rows)
        + "\n\n_Generated via `python scripts/manage_kb.py catalog models`_.\n"
    )
    if out:
        path = _ensure_dir("catalog", out)
        path.write_text(markdown)
    else:
        console.print(markdown)


@catalog_app.command("datasets")
def catalog_datasets(out: Path = DESTINATION_MARKDOWN_OPTION):
    cards = _load_dataset_index()
    if not cards:
        raise typer.Exit(code=1)
    headers = ["Dataset", "Modalities", "Records", "Access", "Verified"]
    rows: list[list[str]] = []
    for card in cards:
        counts = card.get("counts") or {}
        record_hint = counts.get("records") or counts.get("subjects") or counts.get("base_pairs")
        rows.append(
            [
                f"`{card['id']}`",
                ", ".join(card.get("modalities", [])) or "—",
                str(record_hint or "—"),
                card.get("access", "—"),
                _bool_icon(card.get("verified", False)),
            ]
        )
    markdown = (
        "# Dataset catalog\n\n"
        + _markdown_table(headers, rows)
        + "\n\n_Generated via `python scripts/manage_kb.py catalog datasets`_.\n"
    )
    if out:
        path = _ensure_dir("catalog", out)
        path.write_text(markdown)
    else:
        console.print(markdown)


@catalog_app.command("integrations")
def catalog_integrations(out: Path = DESTINATION_MARKDOWN_OPTION):
    cards = _load_integration_index()
    if not cards:
        raise typer.Exit(code=1)
    headers = ["Integration", "Models", "Datasets", "Status", "Verified"]
    rows: list[list[str]] = []
    for card in cards:
        rows.append(
            [
                f"`{card['id']}`",
                ", ".join(card.get("models", [])) or "—",
                ", ".join(card.get("datasets", [])) or "—",
                card.get("status", "—"),
                _bool_icon(card.get("verified", False)),
            ]
        )
    markdown = (
        "# Integration catalog\n\n"
        + _markdown_table(headers, rows)
        + "\n\n_Generated via `python scripts/manage_kb.py catalog integrations`_.\n"
    )
    if out:
        path = _ensure_dir("catalog", out)
        path.write_text(markdown)
    else:
        console.print(markdown)


@validate_app.command("models")
def validate_models() -> None:
    """Ensure every model card has the required fields populated."""
    cards = _load_model_index(include_unverified=True)
    missing: list[str] = []
    for card in cards:
        for field in MODEL_REQUIRED_FIELDS:
            if not card.get(field):
                missing.append(f"{card['id']}: missing `{field}`")
    if missing:
        for line in missing:
            console.print(f"[red]model-card:[/] {line}")
        raise typer.Exit(code=1)
    console.print("[green]All model cards pass required-field checks.")


@validate_app.command("datasets")
def validate_datasets() -> None:
    cards = _load_dataset_index()
    missing: list[str] = []
    for card in cards:
        for field in DATASET_REQUIRED_FIELDS:
            if not card.get(field):
                missing.append(f"{card['id']}: missing `{field}`")
        schema_ref = card.get("schema_ref")
        if schema_ref:
            doc_path = schema_ref.split("#")[0]
            if not (ROOT / doc_path).exists():
                missing.append(f"{card['id']}: schema_ref target `{doc_path}` not found")
    if missing:
        for line in missing:
            console.print(f"[red]dataset:[/] {line}")
        raise typer.Exit(code=1)
    console.print("[green]All dataset cards pass required-field checks.")


@validate_app.command("links")
def validate_links() -> None:
    """Check that every integration references existing models/datasets."""
    model_ids = {card["id"] for card in _load_model_index(include_unverified=True)}
    dataset_ids = {card["id"] for card in _load_dataset_index()}
    issues: list[str] = []
    for card in _load_integration_index():
        for model_id in card.get("models", []):
            if model_id not in model_ids:
                issues.append(f"{card['id']}: unknown model `{model_id}`")
        for dataset_id in card.get("datasets", []):
            if dataset_id not in dataset_ids:
                issues.append(f"{card['id']}: unknown dataset `{dataset_id}`")
    if issues:
        for issue in issues:
            console.print(f"[red]integration:[/] {issue}")
        raise typer.Exit(code=1)
    console.print("[green]All integration references resolve.")


def _extract_model_id(markdown: str) -> str | None:
    match = MODEL_ID_PATTERN.search(markdown)
    return match.group(1).strip().lower() if match else None


def _scan_model_docs(paths: Iterable[Path]) -> list[tuple[Path, str | None]]:
    results: list[tuple[Path, str | None]] = []
    for md_path in paths:
        if md_path.name.startswith("index"):
            continue
        model_id = _extract_model_id(md_path.read_text())
        results.append((md_path, model_id))
    return results


@ci_app.command("docs")
def ci_docs() -> None:
    """Ensure docs/models/* and docs/code_walkthroughs/* declare `model-id` markers."""
    model_ids = {card["id"] for card in _load_model_index(include_unverified=True)}
    doc_paths = list((DOCS_ROOT / "models").rglob("*.md"))
    walk_paths = list((DOCS_ROOT / "code_walkthroughs").glob("*.md"))
    issues: list[str] = []
    for path, doc_model_id in _scan_model_docs(doc_paths + walk_paths):
        if not doc_model_id:
            issues.append(f"{path.relative_to(ROOT)} missing <!-- model-id: ... --> marker")
            continue
        if doc_model_id not in model_ids:
            missing_id = doc_model_id
            assert missing_id is not None
            closest = max(model_ids, key=lambda mid: fuzz.ratio(mid, missing_id))
            note = (
                f"{path.relative_to(ROOT)} references unknown model `{missing_id}` "
                f"(did you mean `{closest}`? If you forgot to add a model card, "
                "run `python scripts/manage_kb.py add-model <model-id>`)."
            )
            issues.append(note)
    if issues:
        for issue in issues:
            console.print(f"[red]doc:[/] {issue}")
        raise typer.Exit(code=1)
    console.print("[green]All model docs declare valid model-id markers.")


@ci_app.command("weights")
def ci_weights() -> None:
    """Fail if any model doc lacks links to weights/licenses."""
    cards = _load_model_index(include_unverified=True)
    issues: list[str] = []
    for card in cards:
        weights = card.get("weights") or {}
        license_block = card.get("license") or {}
        if not any(weights.values()):
            issues.append(f"{card['id']}: weights field empty")
        if not any(license_block.values()):
            issues.append(f"{card['id']}: license field empty")
    if issues:
        for issue in issues:
            console.print(f"[red]weights:[/] {issue}")
        raise typer.Exit(code=1)
    console.print("[green]All cards include weights + license metadata.")


@ops_app.command("embeddings")
def ops_embeddings(
    model_id: str = typer.Argument(..., help="Model slug (matches model card id)"),
) -> None:
    snippets = OPS_CONFIG.get(model_id)
    if not snippets or "embeddings" not in snippets:
        raise typer.Exit(code=2)
    console.rule(f"Embedding command for {model_id}")
    console.print(snippets["embeddings"])


@ops_app.command("inference")
def ops_inference(model_id: str = typer.Argument(...)) -> None:
    snippets = OPS_CONFIG.get(model_id)
    if not snippets or "inference" not in snippets:
        raise typer.Exit(code=2)
    console.rule(f"Inference snippet for {model_id}")
    console.print(snippets["inference"])


@ops_app.command("walkthrough-draft")
def ops_walkthrough_draft(
    model_id: str = typer.Argument(...),
    out: Path = WALKTHROUGH_OUT_OPTION,
    overwrite: bool = WALKTHROUGH_OVERWRITE_OPTION,
):
    cards = {card["id"]: card for card in _load_model_index(include_unverified=True)}
    if model_id not in cards:
        raise typer.Exit(code=2)
    card = cards[model_id]
    content = WALKTHROUGH_TEMPLATE.format(
        name=card["name"],
        repo=card["repo"],
        summary=card.get("summary", ""),
        context=card.get("context_length", "?"),
        checkpoints=(
            "\n".join(f"- {item}" for item in card.get("checkpoints", [])) or "- (add checkpoints)"
        ),
    )
    if out:
        path = _ensure_dir("walkthrough", out)
        if path.exists() and not overwrite:
            raise typer.BadParameter(f"{path} already exists; pass --overwrite to replace")
        path.write_text(content)
    else:
        console.print(content)


@app.command("rag-index")
def build_rag_index(out: Path = RAG_INDEX_OUT_OPTION) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "models": _load_model_index(include_unverified=True),
        "datasets": _load_dataset_index(),
        "integrations": _load_integration_index(),
    }
    path = _ensure_dir("rag", out)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    console.print(f"[green]Wrote RAG index to {path.relative_to(ROOT)}")


OPS_CONFIG: dict[str, dict[str, str]] = {
    "caduceus": {
        "embeddings": """```
torchrun --standalone --nproc-per-node=8 external_repos/caduceus/vep_embeddings.py \
  --model_name_or_path kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
  --seq_len 131072 --bp_per_token 1 --embed_dump_batch_size 1 --rcps
```""",
        "inference": """```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
)
tokenizer = AutoTokenizer.from_pretrained(
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
)
inputs = tokenizer("ACGT" * 1000, return_tensors="pt")
logits = model(**inputs).logits
```""",
    },
    "dnabert2": {
        "embeddings": """```
python external_repos/dnabert2/finetune/train.py \
  --model_name_or_path zhihan1996/DNABERT-2-117M \
  --data_path /path/to/GUE \
  --run_name dnabert2_gue --model_max_length 512 --per_device_train_batch_size 8
```""",
        "inference": """```python
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
emb = model(tokenizer("ACGT", return_tensors="pt")["input_ids"])[0]
```""",
    },
    "generator": {
        "embeddings": """```
python external_repos/generator/src/tasks/downstream/sequence_understanding.py \
  --model_name GenerTeam/GENERator-v2-eukaryote-1.2b-base \
  --dataset_name GenerTeam/gener-tasks --subset_name gene_classification --batch_size 8 --bf16
```""",
        "inference": """```
python external_repos/generator/src/tasks/downstream/fine_tuning.py \
  --model_name GenerTeam/GENERator-v2-prokaryote-1.2b-base \
  --dataset_name GenerTeam/sequence-recovery --num_train_epochs 3 --bf16
```""",
    },
    "evo2": {
        "embeddings": """```python
from evo2 import Evo2
model = Evo2('evo2_7b')
seqs = ['ACGTACGT']
scores = model.score_sequences(seqs, batch_size=1)
```""",
        "inference": """```python
from evo2 import Evo2
model = Evo2('evo2_7b')
out = model.generate(prompt_seqs=["ACGT"], n_tokens=256, temperature=0.8)
print(out.sequences[0])
```""",
    },
    "brainlm": {
        "embeddings": """```
python external_repos/brainlm/train.py \
  --config configs/pretrain.yaml \
  --output_dir outputs/brainlm_pretrain
```""",
        "inference": """```
python external_repos/brainlm/brainlm_tutorial.ipynb \
  # run via Jupyter to create zero-shot predictions
```""",
    },
    "brainmt": {
        "embeddings": """```
torchrun --nproc-per-node=4 external_repos/brainmt/src/brainmt/train.py \
  task=regression dataset.fmri.img_path=/path/to/tensors
```""",
        "inference": """```
python external_repos/brainmt/src/brainmt/inference.py \
  task=classification inference.checkpoint_path=/path/to.ckpt
```""",
    },
    "swift": {
        "embeddings": """```
python external_repos/swift/project/main.py \
  --dataset_name UKB --downstream_task sex --pretraining --use_contrastive
```""",
        "inference": """```
python external_repos/swift/project/main.py \
  --dataset_name UKB --test_only \
  --test_ckpt_path pretrained_models/contrastive_pretrained.ckpt
```""",
    },
}


WALKTHROUGH_TEMPLATE = """# {name} code walkthrough

Repo: {repo}

## Snapshot
- Context length: {context}
- Summary: {summary}
- Checkpoints:\n{checkpoints}

## Sections to flesh out
1. Architecture backbone
2. Data and loaders
3. Training entrypoints
4. Inference and embeddings
5. Pitfalls & integration hooks

"""


if __name__ == "__main__":
    app()
