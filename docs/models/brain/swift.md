# SwiFT

## Architecture overview
Swin-style 3D/4D backbone adapted for fMRI volumes, with temporal attention for BOLD sequences.

## Expected inputs
- Preprocessed BOLD volumes (subject-session)
- Optional mask/confound channels as additional tokens

## Inference plan
- Produce subject-level embeddings via windowed pooling
- Aggregate across sessions before downstream prediction

## Links
- Repo: TBD
- Weights: TBD



