Integration cards capture multimodal pipelines:

```
id: unique_slug
name: Title
summary: 2–3 sentences
models:
  - model_id
  - ...
datasets:
  - dataset_id
pipelines:
  - bulleted free text describing scripts/configs
status: idea|designing|running
outputs:
  - path: relative/path
    description: artifact description
ci_checks:
  - optional list of preflight commands
owner: Person or team
verified: false
last_updated: YYYY-MM-DD
```
