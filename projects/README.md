# Object projects

Each object type gets its own folder. Run `defect object init`, or use `defect train guided "..."` after cloud settings are configured. Inside an object folder:

| File or folder | What it contains |
| --- | --- |
| `object.yaml` | Object name and ordered defect classes |
| `dataset.yaml` | Label sources, accepted aliases, splits, and GCS destination |
| `dataset-version.yaml` | Immutable published dataset reference |
| `experiment.yaml` | Model weights and training choices |
| `runtime.yaml` | Exact certified trainer image digest |
| `vertex.yaml` | GPU machine, project, service account, and spending limit |
| `run-request.yaml` | Reviewed input used for a guided submission |
| `runs/<run-id>.yaml` | Small locator with current state and remote artifact addresses |

Large images, dataset shards, checkpoints, and logs belong in GCS, MLflow, and Cloud Logging. Use `defect run status RUN_ID` for the latest status; a run locator is a snapshot from its last CLI update.
