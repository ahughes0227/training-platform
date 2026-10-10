# Goals: from a target to a model or a report

You state what a model must achieve and when to stop trying. The platform trains a baseline and checks it against the goal. It keeps launching the next experiment until the goal is met or a stop rule fires. It ends in one of two outcomes:

- **Model delivered:** the model package, plus a scorecard comparing each target with what was achieved.
- **Goal not met:** a report covering why it stopped, the closest result, which classes fell short and what they were mistaken for, what was tried, and what would most likely reach the goal.

Each experiment trains on this machine, or on Vertex AI with `--vertex`. The loop, the goal and the reports are the same either way.

## Try it

The demo builds a tiny synthetic problem and runs it end to end on a CPU in a few seconds. It uses three classes of 32×32 crops and a miniature DINOv3 backbone, with no GPU and no downloads. It needs the `train` and `data` extras.

```bash
uv sync --extra train --extra data
uv run defect goal demo /tmp/goal-demo                       # met after a couple of runs
uv run defect goal demo /tmp/goal-demo-hard --unreachable    # two classes look identical
```

Each prints the runs and the path of `REPORT.md`.

## Run it on Vertex AI

`scripts/gcp_goal_setup.sh` does the one-time setup in a GCP project, then runs the demo with each experiment as a Vertex AI job. Run it from the repository root in Cloud Shell, or anywhere `gcloud` is logged in as a project owner. Billing must be on. It is safe to re-run.

```bash
gcloud config set project YOUR_PROJECT
bash scripts/gcp_goal_setup.sh
```

It enables the APIs, then creates a bucket, an Artifact Registry repository, and a `defect-trainer` service account for the jobs to run as. It builds the trainer image with Cloud Build and writes `vertex.yaml`. The demo jobs run on a CPU machine (`n1-standard-4`), so no GPU quota is needed. Each job takes a few minutes to start.

`vertex.yaml` is what `--vertex` reads:

```yaml
project: my-project
region: us-central1
service_account: defect-trainer@my-project.iam.gserviceaccount.com
staging_uri: gs://my-project-defect-goals/goals   # requests and run outputs
image_digest: us-central1-docker.pkg.dev/my-project/defect-platform/defect-trainer@sha256:...
machine_type: n1-standard-4
accelerator_type: NVIDIA_L4    # optional, with accelerator_count
accelerator_count: 1
runtime: { ... }               # a certified runtime record, for releasable models
allow_uncertified_image: true  # or a development image, for evaluation only
```

With `--vertex`, the dataset and the backbone weights must be in GCS. Build the dataset with a `gs://` output and upload the weights. The image must be pinned by digest. Models count as releasable only when they are trained on a certified runtime. The report says which runtime trained the model. A goal that is interrupted re-attaches to a running job instead of starting a second one. Each attempt at a goal (a goal and dataset pair) stages its requests and outputs in its own folder under `staging_uri`, so a rebuilt dataset or a changed goal never collides with an earlier attempt's runs.

## Your own goal

`defect goal start goal.yaml` takes one file:

```yaml
goal:
  goal_id: panel-v1             # lowercase; names the runs and the goal directory
  object_slug: panel
  primary_metric: mcc           # or macro_f1, measured on validation data
  min_primary: 0.8
  min_class_recall: 0.75        # optional floor for every class
  max_review_rate: 0.2          # optional; needs max_review_error_rate in the experiment
  max_runs: 8                   # stop after this many runs
  deadline: 2026-10-15T17:00:00Z  # optional
  plateau_runs: 3               # stop when this many runs in a row don't improve by
  min_improvement: 0.01         # ...at least this much
  min_validation_support: 30    # fewer validation samples than this for a weak class is a data request
dataset: dataset-version.yaml   # written by `defect dataset build`, or inline
experiment: { ... }             # the baseline ExperimentConfig
```

Runs, the state, `scorecard.json` and `REPORT.md` go to `goals/<goal_id>/` next to the file, or to `--dir`. `defect goal status <dir>` shows progress. Running `start` again with the same file resumes an interrupted goal. A changed goal needs a new directory.

## How the next experiment is chosen

1. **Analysis.** The best run so far goes to the Analysis plane. A class short on validation samples, or with suspected label errors, becomes a data request, and the goal stops with it, because tuning can't fix missing evidence. Otherwise Analysis proposes class weights, focal loss or backbone fine-tuning. The proposal passes the plane handoff gate (recorded in `handoffs.sqlite`) before it runs.
2. **Fallbacks.** If Analysis has nothing new, the runner tries one change at a time: more epochs, a lower learning rate, fine-tuning one more backbone layer, then focal loss.
3. **Limits.** Only the hyperparameters Analysis may tune change. The dataset, class catalog, backbone weights and runtime never do. Every decision uses validation data. The test split is read once, for the delivered model.

The goal stops on a met goal, the run limit, the deadline, a plateau, a data request, no admissible change left, a failed run, or a dataset with an empty split.
