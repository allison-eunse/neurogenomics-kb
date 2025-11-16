

md. I’ll also keep carrying the specific repository + license notice on every walkthrough page so each doc makes its licensing context explicit.

# Dataset Card Template

## Metadata
- **Dataset name:** `<UKB/ABCD/etc>`
- **Source / accession:** `<URL or data agreement>`
- **License / DUA:** `<e.g., UKB Research Analysis Tool terms>`
- **Primary contact:** `<PI / analyst>`
- **Related YAML:** ``kb/datasets/<id>.yaml``

## Access & Storage
- Paths to raw data (e.g., `/secure/ukb/raw`), processed derivatives, and sync locations.
- Encryption / PHI considerations; who has access.

## Cohort Summary
| Split | N subjects | Notes |
| --- | --- | --- |
| Train | `<N>` | `<filters>` |
| Val | `<N>` |  |
| Test | `<N>` |  |

Add stratification notes (sex balance, age range, sites).

## Overlap & Pairing Logic
- How subjects overlap with other modalities (e.g., gene × sMRI intersection).
- Inclusion/exclusion rules, QC flags (motion, missing genes, etc.).

## Preprocessing Pipelines
- Imaging: software versions, atlases, smoothing, censoring thresholds.
- Genetics: variant calling, phasing, annotation, gene list definition.
- Clinical: diagnosis coding, questionnaire scoring.

## Available Features & Covariates
- Feature tables (e.g., FreeSurfer ROIs, FC z-maps, gene embeddings).
- Covariates recommended for residualization (age, sex, site, motion, PCs, SES).
- Missingness patterns and imputation strategy if required.

## QC & Harmonization
- Thresholds (FD < 0.3 mm, min TR length).
- Site/scanner harmonization (e.g., ComBat) or reasons for not applying.
- Logs / reports location.

## Notes & Risks
- Any data use restrictions (no redistribution, publication review).
- Confounds or biases to monitor (ancestry imbalance, site skew).
- Planned updates or reprocessing tasks.

## References
- Link to cohort publications / documentation.
- Internal tickets or notebooks for preprocessing runs.

