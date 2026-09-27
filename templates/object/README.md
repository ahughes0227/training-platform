# {{ display_name }}

This folder describes one inspected object and its defect classes. The editable files are `object.yaml`, `dataset.yaml`, `experiment.yaml`, and `vertex.yaml`. Runs and artifacts are found with `defect run list --object {{ slug }}` and `defect run status RUN_ID`.

Do not put images, model weights, secrets, or generated shards in this Git folder. Dataset versions and run outputs are stored in GCS; model versions are tracked in MLflow.
