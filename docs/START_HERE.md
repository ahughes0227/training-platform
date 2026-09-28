# Start here

This platform is for one task: name the class of a defect that has already been found and cropped. Give each inspected object type its own project. The platform keeps a record of what images, labels, settings, software image, and model produced each result.

For the complete system contract and acceptance criteria, read [the full requirements document](REQUIREMENTS.md). It distinguishes required behavior, prevention checks, current evidence, and unfinished work.

## What you provide

- An object name and at least two defect classes, with explicit meanings. Optional example/exclusion references help a human resolve ambiguous labels.
- A CSV manifest or BigQuery table with one row per crop and columns identifying the image and label. Images may be local files or `gs://` objects. Optional product/lot IDs help keep related crops out of different train/test sets.
- The DINOv3 weight location and checksum, a GCP project/region, and a spending limit before cloud training. Ask your administrator for the GCS, MLflow, and Vertex settings; leave them unset for local exploration.

## Typical flow

1. When cloud settings are configured, run `defect train guided "OBJECT DETAILS, NOTES, LABEL CSV OR BIGQUERY TABLE"`. The guided path uses LiteLLM for up to three rounds of questions, writes a project folder, reviews label exceptions, builds a versioned dataset, and submits a paid run only after its cost and dataset checks pass. It returns a run ID.
2. For manual setup, run `defect object init` to create a plain-language object folder from `templates/object/`.
3. Edit `object.yaml` and `dataset.yaml`. Define stable IDs, meanings and aliases in `class_catalog`; approve with `defect object review-catalog OBJECT.yaml --reviewer YOUR_ID --approve` using your operator catalog store. Increment the catalog version when it changes. `class-catalog.yaml` is a generated readable snapshot. Run `defect dataset preview DATASET.yaml --object OBJECT.yaml` to see label mappings, missing images, and conflicts. Resolve the exceptions it lists.
4. Run `defect dataset build DATASET.yaml --object OBJECT.yaml`. It creates an immutable dataset version in GCS (or locally for testing) and prints its version ID and manifest location.
5. Edit the generated experiment and Vertex settings, or use `defect setup ask` with notes. The agent suggests values; the CLI validates them.
6. Set `DEFECT_CONTROL_SERVICE_URL` to the deployed control URL, then run `defect train start projects/OBJECT/`. The platform checks the dataset, certified image digest, credentials, and estimated cost against your configured per-run limit. It returns a run ID without waiting for training.
7. Use `defect run status RUN_ID` or `defect run list` to find the Vertex job, logs, MLflow record, evaluation, checkpoint, and any failure explanation.
8. Review MCC, macro F1, per-class errors, uncertain-case review rate, and diagnostic heatmaps. A candidate does not become live until you explicitly promote it.

Cloud submission also requires an operator-published infrastructure capability record with a pinned checksum and current observations. See [module contracts and migration](MODULE_CONTRACTS.md). A legacy or draft vocabulary remains available for local exploration; it cannot self-authorize a paid run.

The CLI output should always show the next useful action on a failure. Changing labels, learning rate, epochs, augmentations, or GPU count does not inherently rebuild the trainer image.

## Where things live

| Item | Location |
| --- | --- |
| Reviewed vocabulary snapshot | `projects/<object>/class-catalog.yaml`; approved immutable copy in the configured catalog store |
| Dataset meaning and evidence | Dataset `semantics.json` and source/split manifests |
| Model meaning and bundle checksums | Run `model/semantics.json` and `model/integrity.json` |
| Editable object definition | `projects/<object>/object.yaml` |
| Editable data and experiment settings | `projects/<object>/*.yaml` |
| Accepted dataset snapshot and WebDataset TAR shards | `gs://.../datasets/<object>/<version>/` |
| Job outputs and evaluation report | The run's `output_uri`, shown by `defect run status RUN_ID` |
| Experiment metrics and model versions | MLflow tracking and registry |
| Live prediction service | RayService `defect-<object>` in GKE |

The repository is an index and source of templates, not a place to store images or model weights.
