from __future__ import annotations

import pytest
from pydantic import ValidationError
from test_control import FakeVertex, fixtures

from defect_platform.contracts import ExperimentConfig, ModelSpec, RunState
from defect_platform.control.controller import RunController
from defect_platform.control.search import (
    EarlyStoppingPolicy,
    ExperimentSearch,
    RunControllerSubmitter,
    SearchBudget,
    SearchSpace,
    SearchState,
    TrialScore,
    TrialStatus,
)
from defect_platform.control.store import SQLiteRunStore
from defect_platform.trainer.metrics import evaluate_predictions


def base_experiment(**updates) -> ExperimentConfig:
    values = {"experiment_id": "exp-1", "object_slug": "panel", "dataset_version_id": "ds-1",
              "runtime_id": "runtime-1", "catalog_sha256": "c" * 64,
              "model": ModelSpec(weights_uri="gs://weights/dino.safetensors",
                                 weights_sha256="b" * 64)}
    return ExperimentConfig(**{**values, **updates})


class FakeSubmitter:
    def __init__(self, cost: float = 10.0, fail_with: Exception | None = None):
        self.cost, self.fail_with = cost, fail_with
        self.submitted: list[tuple[ExperimentConfig, str]] = []
        self.stopped: list[tuple[str, str]] = []

    def estimate_cost(self, experiment):
        return self.cost

    def submit(self, experiment, idempotency_key):
        if self.fail_with:
            raise self.fail_with
        self.submitted.append((experiment, idempotency_key))
        return f"run-{len(self.submitted)}"

    def stop(self, run_id, reason):
        self.stopped.append((run_id, reason))


SPACE = SearchSpace(choices={"learning_rate": [1e-4, 3e-4, 1e-3], "model.dropout": [0.1, 0.3]},
                    seed=7)


def make_search(*, budget=None, space=SPACE, submitter=None, early_stopping=None, state=None):
    return ExperimentSearch(
        base=base_experiment(), space=space,
        budget=budget or SearchBudget(max_trials=20, max_total_cost_usd=1000, max_concurrent=10),
        submitter=submitter or FakeSubmitter(), early_stopping=early_stopping, state=state,
        id_factory=lambda: "search-1")


def score(mcc, f1=0.5):
    return TrialScore(mcc=mcc, macro_f1=f1)


def test_space_rejects_pinned_and_unknown_fields():
    for path in ("dataset_version_id", "runtime_id", "catalog_sha256", "model.weights_uri",
                 "model.image_size", "model", "not_a_field"):
        with pytest.raises(ValidationError):
            SearchSpace(choices={path: [1]})
    with pytest.raises(ValidationError, match="unique"):
        SearchSpace(choices={"seed": [1, 1]})


def test_invalid_choice_is_rejected_before_any_submission():
    submitter = FakeSubmitter()
    with pytest.raises(ValidationError):
        make_search(space=SearchSpace(choices={"learning_rate": [1e-3, -1.0]}), submitter=submitter)
    assert submitter.submitted == []


def test_proposals_cover_space_once_and_keep_immutable_inputs_pinned():
    submitter = FakeSubmitter()
    search = make_search(submitter=submitter)

    launched = search.step()

    assert len(launched) == 1 + SPACE.size  # base config, then every grid point
    assert search.state.halted_reason == "search space exhausted"
    assert launched[0].overrides == {}
    assert len({trial.fingerprint for trial in launched}) == len(launched)
    assert len({trial.experiment.experiment_id for trial in launched}) == len(launched)
    base = base_experiment()
    for trial in launched:
        experiment = trial.experiment
        assert experiment.experiment_id.startswith("exp-1-t")
        assert (experiment.dataset_version_id, experiment.runtime_id, experiment.catalog_sha256,
                experiment.model.weights_uri, experiment.model.weights_sha256) == (
                base.dataset_version_id, base.runtime_id, base.catalog_sha256,
                base.model.weights_uri, base.model.weights_sha256)
    assert {(t.experiment.learning_rate, t.experiment.model.dropout) for t in launched[1:]} == {
        (lr, d) for lr in (1e-4, 3e-4, 1e-3) for d in (0.1, 0.3)}
    keys = [key for _, key in submitter.submitted]
    assert len(set(keys)) == len(keys) and all(key.startswith("search:search-1:") for key in keys)


def test_grid_point_equal_to_base_is_not_run_twice():
    space = SearchSpace(choices={"learning_rate": [1e-3, 3e-4]})  # 1e-3 is the base value
    launched = make_search(space=space).step()
    assert [trial.experiment.learning_rate for trial in launched] == [1e-3, 3e-4]


def test_proposal_order_is_deterministic_for_a_seed():
    first = [t.overrides for t in make_search().step()]
    second = [t.overrides for t in make_search().step()]
    assert first == second


def test_concurrency_trial_and_cost_budgets_are_enforced():
    search = make_search(budget=SearchBudget(max_trials=4, max_total_cost_usd=25,
                                             max_concurrent=2))
    first = search.step()
    assert len(first) == 2 and search.state.reserved_cost_usd == 20
    assert search.step() == []  # no free slot

    search.complete(first[0].trial_id, score(0.5))
    assert search.step() == []
    assert search.state.halted_reason.startswith("cost budget exhausted")
    assert search.state.reserved_cost_usd == 20 and search.remaining_usd == 5

    capped = make_search(budget=SearchBudget(max_trials=3, max_total_cost_usd=1000,
                                             max_concurrent=10))
    assert len(capped.step()) == 3
    assert capped.state.halted_reason == "trial budget exhausted"


def test_early_stop_does_not_refund_reserved_cost():
    search = make_search(budget=SearchBudget(max_trials=10, max_total_cost_usd=30,
                                             max_concurrent=3),
                         early_stopping=EarlyStoppingPolicy(grace_epochs=1, min_peers=2))
    a, b, c = search.step()
    search.report(a.trial_id, 1, score(0.6))
    search.report(b.trial_id, 1, score(0.5))
    assert search.report(c.trial_id, 1, score(0.1)) is not None
    assert search.state.reserved_cost_usd == 30
    assert search.step() == []
    assert search.state.halted_reason.startswith("cost budget exhausted")


def test_median_rule_stops_weak_trial_across_runs_and_calls_submitter():
    submitter = FakeSubmitter()
    search = make_search(budget=SearchBudget(max_trials=4, max_total_cost_usd=1000,
                                             max_concurrent=4),
                         submitter=submitter,
                         early_stopping=EarlyStoppingPolicy(grace_epochs=2, min_peers=2))
    a, b, c, d = search.step()

    # Two earlier runs finish with full curves.
    for trial, values in ((a, (0.3, 0.5, 0.6)), (b, (0.2, 0.4, 0.55))):
        for epoch, value in enumerate(values, start=1):
            assert search.report(trial.trial_id, epoch, score(value)) is None
        search.complete(trial.trial_id, score(values[-1]))

    # Inside the grace period nothing stops, even when far behind.
    assert search.report(c.trial_id, 1, score(-0.2)) is None
    reason = search.report(c.trial_id, 2, score(0.1))
    assert reason is not None and "median" in reason
    assert c.status == TrialStatus.STOPPED and c.stop_reason == reason
    assert submitter.stopped == [(c.run_id, reason)]

    # A trial at or above the peer median keeps running; ties in MCC fall to macro F1.
    assert search.report(d.trial_id, 1, score(0.3)) is None
    assert search.report(d.trial_id, 2, score(0.4, 0.9)) is None
    assert d.status == TrialStatus.RUNNING
    with pytest.raises(ValueError, match="not running"):
        search.report(c.trial_id, 3, score(0.9))


def test_early_stopping_needs_enough_peers():
    policy = EarlyStoppingPolicy(grace_epochs=1, min_peers=3)
    search = make_search(early_stopping=policy)
    a, b, c, *_ = search.step()
    search.report(a.trial_id, 1, score(0.9))
    search.report(b.trial_id, 1, score(0.8))
    assert search.report(c.trial_id, 1, score(-0.5)) is None


def test_observations_must_advance_within_configured_epochs():
    search = make_search()
    trial = search.step()[0]
    search.report(trial.trial_id, 2, score(0.1))
    with pytest.raises(ValueError, match="strictly increasing"):
        search.report(trial.trial_id, 2, score(0.2))
    with pytest.raises(ValueError, match="configured epochs"):
        search.report(trial.trial_id, trial.experiment.epochs + 1, score(0.2))
    with pytest.raises(ValidationError):
        TrialScore(mcc=1.5, macro_f1=0.5)


def test_leaderboard_ranks_by_mcc_then_macro_f1():
    search = make_search()
    a, b, c, d, *_ = search.step()
    search.complete(a.trial_id, score(0.70, 0.60))
    search.complete(b.trial_id, score(0.80, 0.50))
    search.complete(c.trial_id, score(0.80, 0.65))
    search.fail(d.trial_id, "VERTEX_CAPACITY: quota")

    assert [t.trial_id for t in search.leaderboard()] == [c.trial_id, b.trial_id, a.trial_id]
    assert search.best().trial_id == c.trial_id


def test_leaderboard_can_include_stopped_trials_by_best_observation():
    search = make_search(early_stopping=EarlyStoppingPolicy(grace_epochs=1, min_peers=1))
    a, b, *_ = search.step()
    search.report(a.trial_id, 1, score(0.5))
    search.complete(a.trial_id, score(0.5))
    search.report(b.trial_id, 1, score(0.2))
    assert b.status == TrialStatus.STOPPED
    assert [t.trial_id for t in search.leaderboard()] == [a.trial_id]
    assert [t.trial_id for t in search.leaderboard(include_stopped=True)] == [a.trial_id, b.trial_id]


def test_score_from_evaluation_matches_metrics_selection_key():
    evaluation = evaluate_predictions([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0],
                                      ["crack", "dent", "scratch"])
    assert TrialScore.from_evaluation(evaluation).selection_key == evaluation.selection_key


def test_consecutive_submission_failures_halt_search():
    submitter = FakeSubmitter(fail_with=RuntimeError("workflow unavailable"))
    search = make_search(budget=SearchBudget(max_trials=10, max_total_cost_usd=1000,
                                             max_concurrent=1, max_consecutive_failures=2),
                         submitter=submitter)
    assert search.step() == []
    assert [t.status for t in search.state.trials] == [TrialStatus.FAILED] * 2
    assert "2 consecutive trial failures" in search.state.halted_reason
    assert search.state.reserved_cost_usd == 20  # failed submissions keep their reservation
    assert search.finished


def test_state_round_trips_and_resume_continues_without_repeats():
    budget = SearchBudget(max_trials=3, max_total_cost_usd=1000, max_concurrent=3)
    search = make_search(budget=budget)
    first = search.step()
    saved = SearchState.model_validate_json(search.state.model_dump_json())

    saved.halted_reason = None
    resumed = make_search(budget=SearchBudget(max_trials=7, max_total_cost_usd=1000,
                                              max_concurrent=10), state=saved)
    more = resumed.step()

    assert len(first) == 3 and len(more) == 4
    fingerprints = [t.fingerprint for t in resumed.state.trials]
    assert len(set(fingerprints)) == 7
    with pytest.raises(ValueError, match="different base experiment"):
        ExperimentSearch(base=base_experiment(seed=1), space=SPACE, budget=budget,
                         submitter=FakeSubmitter(), state=saved)


def test_run_controller_submitter_applies_run_guardrails_and_records_early_stop(tmp_path):
    runtime, experiment, _dataset, job, catalog, catalogs = fixtures(tmp_path)
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    vertex = FakeVertex()
    controller = RunController(store=store, runtimes=catalog, datasets=catalog, vertex=vertex,
                               catalogs=catalogs)
    search = ExperimentSearch(
        base=experiment, space=SearchSpace(choices={"learning_rate": [3e-4]}),
        budget=SearchBudget(max_trials=5, max_total_cost_usd=45, max_concurrent=5),
        submitter=RunControllerSubmitter(controller, job),
        early_stopping=EarlyStoppingPolicy(grace_epochs=1, min_peers=1))

    a, b = search.step()

    assert search.state.reserved_cost_usd == 2 * controller.estimate_cost(job)
    assert [call["runtime"].image_digest for call in vertex.calls] == [runtime.image_digest] * 2
    assert {controller.get(t.run_id).experiment_id for t in (a, b)} == {a.trial_id, b.trial_id}
    search.report(a.trial_id, 1, score(0.6))
    search.report(b.trial_id, 1, score(0.1))
    stopped = controller.get(b.run_id)
    assert stopped.state == RunState.CANCELED and stopped.failure_code == "RUN_CANCELED"
    assert "EARLY_STOPPED" in stopped.failure_message
    # The early stop must halt the paid job, not only relabel the run.
    assert vertex.canceled == [(stopped.vertex_job_name, job.region)]

    over_cap = RunControllerSubmitter(controller, job.model_copy(update={"max_run_cost_usd": 19}))
    rejected = ExperimentSearch(
        base=experiment, space=SearchSpace(choices={"seed": [1]}),
        budget=SearchBudget(max_trials=5, max_total_cost_usd=1000, max_consecutive_failures=1),
        submitter=over_cap)
    assert rejected.step() == []
    assert "exceeds configured run cap" in rejected.state.trials[0].failure
