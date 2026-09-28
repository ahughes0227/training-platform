# {{ display_name }}

This folder describes one inspected object and its defect classes. The editable files are `object.yaml`, `dataset.yaml`, `experiment.yaml`, and `vertex.yaml`. Runs and artifacts are found with `defect run list --object {{ slug }}` and `defect run status RUN_ID`.

Edit `class_catalog` in `object.yaml`: each class needs a stable ID, its meaning, aliases, and optional positive/negative reference examples. `class-catalog.yaml` is a readable snapshot generated during setup/review. Approve the object with `defect object review-catalog object.yaml --reviewer YOUR_ID --approve` using the operator's `DEFECT_CLASS_CATALOG_ROOT`. A draft or legacy catalog supports local exploration; cloud admission requires the exact approved catalog.

Increment the catalog version when meanings, order, IDs, or aliases change, then rebuild the dataset. A version cannot be reused with different meaning. Model exports preserve this catalog and preprocessing/training rules in `model/semantics.json`; releases pin its fingerprint and all bundle files.

Do not put images, model weights, secrets, or generated shards in this Git folder. Dataset versions and run outputs are stored in GCS; model versions are tracked in MLflow.
