# Me-LLaMA Code Walkthrough

> **KB references:** Model card (pending) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
Me-LLaMA extends LLaMA-2/3 checkpoints through 129B-token continual pre-training (biomedical + clinical + general corpora) followed by 214K-instruction LoRA-based tuning, then evaluates models with a custom `lm_eval` harness covering 12 medical QA/NLP benchmarks. The README documents data composition, optimizer settings, and HPC footprint, while the `src/` tree contains a prompt-aware evaluation CLI, task registry, and optional OpenAI-backed `ChatLM` for API comparisons.^[```1:188:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/README.md```][```1:97:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/eval.py```][```1:210:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/evaluator.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| Continual-pretrained LLaMA2/3 (13B/70B/8B) + instruction tuning + prompt wrapper evaluation harness.^[```33:134:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/README.md```][```1:97:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/eval.py```] | Uses AdamW (lr 8e-6), cosine schedule (5% warmup), bf16, DeepSpeed model parallelism, 8×H100 for LoRA instruction tuning.^[```80:90:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/README.md```] | 129B tokens with 15 : 1 : 4 biomedical : clinical : general ratio plus 214K instruction samples; guidelines + PubMed + general corpora.^[```68:77:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/README.md```] | `poetry run python src/eval.py --model hf-causal-vllm --tasks PUBMEDQA,...` with optional OpenAI API keys; default PYTHONPATH extends `src/` and `medical-evaluation` for metrics.^[```143:188:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/README.md```][```1:14:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/scripts/run_evaluation.sh```] | Custom `lm_eval` fork with medical tasks (`tasks/vital_measure.py`), prompt templating, JSON output, caching, and OpenAI-compatible `ChatLM` for API baselines.^[```1:210:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/evaluator.py```][```1:120:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/tasks/vital_measure.py```][```12:165:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/chatlm.py```] | [github.com/BIDS-Xu-Lab/Me-LLaMA](https://github.com/BIDS-Xu-Lab/Me-LLaMA) |

### Environment & Hardware Notes
- **Poetry-first install.** Clone the repo, run `poetry install`, export `PYTHONPATH="$repo/src:$repo/src/medical-evaluation:$repo/src/metrics/BARTScore"`, and optionally set `CUDA_VISIBLE_DEVICES` before running evaluation scripts.^[```1:14:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/scripts/run_evaluation.sh```]
- **Metric assets.** Download `bart_score.pth` (BARTScore), install `en_core_web_lg`, and keep Stanford CoreNLP + multilingual extras for evaluation tasks that rely on syntax or multilingual scoring.^[```143:188:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/README.md```]
- **API comparisons.** Set `OPENAI_API_SECRET_KEY` when using `--model gpt-4` (see README instructions) so `ChatLM` can enqueue HTTPX requests with exponential backoff.^[```178:188:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/README.md```][```12:165:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/chatlm.py```]

## Key Components

### Evaluation Entrypoint (`src/eval.py`)
`parse_args` exposes the same knobs as upstream `lm_eval` (model/model_args/tasks/few-shot/batching/output). `main()` resolves task patterns, optional description dicts, and forwards everything to `evaluator.simple_evaluate`, writing JSON to `--output_path` when provided.

```13:94:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/eval.py
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--model_prompt", default="no_prompt", choices=list(MODEL_PROMPT_MAP.keys()))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()
...
    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        model_prompt=args.model_prompt
    )
```

### Evaluator & Task Dispatch (`src/evaluator.py`)
`simple_evaluate` instantiates a Hugging Face (or OpenAI) LM, wraps it with a caching layer, builds the medical task dictionary, and runs `evaluate` to orchestrate prompts, few-shot contexts, turn-based conversations, and metric aggregation. Bootstrap statistics, JSON logging, and caching directories mirror upstream `lm_eval` APIs for drop-in adoption.

```19:133:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/evaluator.py
@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    max_batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    model_prompt=None
):
    random.seed(1234)
    np.random.seed(1234)

    assert len(tasks) != 0, "No tasks specified"

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        if model[:3] != "gpt":
            lm = lm_eval.models.get_model(model).create_from_arg_string(
                model_args, {"batch_size": batch_size, "max_batch_size": max_batch_size, "device": device}
            )
        else:
            lm = ChatLM(model)
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    if not no_cache:
        lm = lm_eval.base.CachingLM(
            lm,
            "lm_cache/"
            + (model if isinstance(model, str) else model.model.config._name_or_path)
            + "_"
            + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
            + ".db",
        )

    task_dict = ta.get_task_dict(tasks)
...
    results["config"] = {
        "model": (model if isinstance(model, str) else model.model.config._name_or_path),
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "batch_sizes": list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else [],
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "description_dict": description_dict,
    }
```

### Medical Task Registry (`src/tasks/__init__.py`)
`TASK_REGISTRY` maps human-readable task names to custom task classes (PubMedQA, MedQA, MedMCQA, etc.). Pattern-matched CLI arguments expand into this registry, so adding a new dataset is a matter of appending to `TASK_REGISTRY` or creating a JSON-backed task via `add_json_task`.

```9:34:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/tasks/__init__.py
TASK_REGISTRY = {
    "PUBMEDQA": vital_measure.PUBMEDQA,
    "MedQA": vital_measure.MedQA,
    "MedMCQA": vital_measure.MedMCQA,
    "EmrQA": vital_measure.EmrQA,
    "i2b2": vital_measure.I2B2,
    "DDI2013": vital_measure.DDI2013,
    "hoc": vital_measure.HoC,
    "MTSample": vital_measure.MTSample,
    "PUBMEDSUM": vital_measure.PubmedSum,
    "MimicSum": vital_measure.MimicSum,
    "BioNLI": vital_measure.BioNLI,
    "MedNLI": vital_measure.MedNLI,
}
```

### Task Definitions & Metrics (`src/tasks/vital_measure.py`)
`Classification` (and its subclasses) provides language-cleaning, response parsing, accuracy/F1/MCC aggregation, and sequence labeling utilities. `SequentialLabeling`/`NER` extend this to HTML BIO alignment, while summarization tasks use Rouge/BARTScore (loaded via Poetry extras).

```50:181:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/tasks/vital_measure.py
class Classification(Task):
    CALCULATE_MCC = False
    LOWER_CASE = True
    FIRST_LETTER = False
    VERSION = 1
    EVAL_LAST_TURN = True

    def reformulate_turn_req(self, req, turn_request, turn):
        return req

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def construct_requests(self, doc, ctx):
        cont_request = rf.greedy_until(ctx, {"until": None})
        return cont_request
...
    def aggregation(self):
        metrics = {
            "acc": mean,
            "missing": mean,
            "f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
        }
        if self.CALCULATE_MCC:
            metrics["mcc"] = self.matthews_corrcoef
        return metrics
```

### ChatLM + Prompt Templates (`src/chatlm.py`, `src/model_prompt.py`)
For API-based baselines, `ChatLM` batches requests with asyncio + HTTPX, uses exponential backoff, and overrides `greedy_until`. Prompt wrappers in `model_prompt.py` add `Human/Assistant` prefixes when `--model_prompt mellama_prompt` is provided.

```12:154:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/chatlm.py
async def single_chat(client, **kwargs):
    ...

class ChatLM(BaseLM):
    REQ_CHUNK_SIZE = 10

    def __init__(self, model, truncate=False):
        super().__init__()

        import openai

        self.model = model
        self.truncate = truncate
        api_key = os.environ["OPENAI_API_SECRET_KEY"]
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
...
    def greedy_until(self, requests):
        if not requests:
            return []
        res = []
        ...
            responses = asyncio.run(oa_completion(
                url="https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                model=self.model,
                messages=[{"role": "user", "content": inp} for inp in inps],
                max_tokens=self.max_gen_toks,
                temperature=0.0,
            ))
```

```1:12:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/model_prompt.py
def no_prompt(ctx):
    return ctx


def mellama_prompt(ctx):
    return f'Human: \n{ctx}\n\nAssistant: \n'


MODEL_PROMPT_MAP = {
    "no_prompt": no_prompt,
    "mellama_prompt": mellama_prompt,
}
```

## Integration Hooks (Language ↔ KB)
- **Extend the task roster.** Add a new `Task` subclass in `tasks/vital_measure.py`, register it via `TASK_REGISTRY`, and it becomes instantly available to KB experiment configs via `--tasks` filters.^[```9:34:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/tasks/__init__.py```]
- **Standardize prompts.** Use `MODEL_PROMPT_MAP` to ensure generated transcripts align with KB annotation guidelines (e.g., always wrap with `Human/Assistant` before ingesting outputs into evaluation cards).^[```1:12:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/model_prompt.py```]
- **Capture evaluation metadata.** `simple_evaluate` returns JSON with config, bootstrap stats, and per-task metrics; store these blobs under `kb/results/` to track longitudinal performance as you fine-tune domain-specific adapters.^[```19:133:/Users/allison/Projects/neuro-omics-kb/external_repos/me-lamma/src/evaluator.py```]

