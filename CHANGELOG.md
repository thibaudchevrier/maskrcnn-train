## v0.4.0 (2026-09-28)

### BREAKING CHANGE

- commands are `python -m fashion_seg prepare` and
`uv run --project families/<...> python -m fashion_seg_<family> train|package|evaluate`;
DVC stages are package_<model> and evaluate_<model>.
- commands are now `python -m fashion_seg prepare|train|package|evaluate`;
DVC stages package_legacy/package_torchvision become package@legacy/package@torchvision;
params.yaml sections train_<family>, legacy_model and torchvision_model become
train.<family>, models.<name> and tracking.

### Fix

- **torchvision**: keep the validation losses when a finished run resumes

### Refactor

- one uv project and entrypoint per model family
- ports-and-adapters architecture for the training workflow

## v0.3.0 (2026-09-28)

### Feat

- **torchvision**: torchvision Mask R-CNN v2 trainer and generic packaging

## v0.2.1 (2026-09-28)

### Fix

- **legacy**: serve the 2021 model with the anchors it was trained with

## v0.2.0 (2026-09-28)

### Feat

- **evaluate**: COCO mAP of the packaged model on the validation split

## v0.1.1 (2026-09-27)

### Fix

- **serving**: use the released contract and Matterport inference packages

## v0.1.0 (2026-09-28)

### Feat

- Matterport trainer, shared core package, prepare stage
- uv project and MLflow packaging of the 2021 Mask R-CNN model
- adding data explore python module
- adding remote artifact for ml experiment
- adding imaterialist data tracked by dvc

### Fix

- drop the legacy MLflow UI target

### Refactor

- **data**: replace pandas with polars
