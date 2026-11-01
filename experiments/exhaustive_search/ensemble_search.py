"""Deterministic finite ensemble selection from already cross-fitted predictions.

``run`` tunes recipes on the supplied labels. Its selected score is therefore a
development score, not validation of the selection procedure. For nested use,
call it on each outer training partition's inner out-of-fold predictions, then
fit its chosen members on that outer training partition and score the untouched
outer partition. This module never fits a base model or reads result files.

The selection heuristic is fixed: require mean fold AUC at least the current
submission's mean fold AUC, then maximize mean AUC + 0.05 * mean TPR. This is a
research heuristic, not the instructor's grading formula. Exact baseline ties
retain the baseline. Probability blends use only probability-producing models;
decision-only singleton models retain their raw margins to avoid saturation.
"""

from itertools import combinations

import numpy as np
from scipy.special import logit


TARGET_FPR = 0.01
TPR_WEIGHT = 0.05


def batch_metrics(labels, scores, desired_fpr=TARGET_FPR):
    """Return exact AUC, interpolated TPR, and standardized partial AUC.

    Rows of ``scores`` are candidate recipes. Tied scores form one ROC endpoint,
    matching sklearn's ROC convention and the instructor's linear interpolation.
    Partial AUC uses sklearn's McClish standardization at ``desired_fpr``.
    """
    y = np.asarray(labels)
    values = np.asarray(scores, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if y.ndim != 1 or values.ndim != 2 or values.shape[1] != len(y):
        raise ValueError("Scores must have shape (recipes, number of labels).")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(values).all():
        raise ValueError("Labels must be binary and scores finite.")
    positive = float(y.sum())
    negative = float(len(y) - positive)
    if positive == 0 or negative == 0 or not 0 < desired_fpr < 1:
        raise ValueError("Both classes and an FPR strictly between zero and one are required.")
    if len(values) == 0:
        return {key: np.empty(0) for key in ("auc", "tpr", "pauc")}

    order = np.argsort(-values, axis=1, kind="stable")
    descending = np.take_along_axis(values, order, axis=1)
    true_positive = np.cumsum(y[order], axis=1, dtype=float)
    false_positive = np.arange(1, len(y) + 1)[None, :] - true_positive
    endpoints = np.column_stack((descending[:, :-1] != descending[:, 1:],
                                 np.ones(len(values), dtype=bool)))
    positions = np.arange(len(y))[None, :]
    latest_endpoint = np.maximum.accumulate(np.where(endpoints, positions, -1), axis=1)
    previous = np.column_stack((np.full(len(values), -1), latest_endpoint[:, :-1]))
    tp_with_origin = np.column_stack((np.zeros(len(values)), true_positive))
    fp_with_origin = np.column_stack((np.zeros(len(values)), false_positive))
    prior_tp = np.take_along_axis(tp_with_origin, previous + 1, axis=1)
    prior_fp = np.take_along_axis(fp_with_origin, previous + 1, axis=1)
    trapezoids = (false_positive - prior_fp) * (true_positive + prior_tp) * 0.5
    auc = np.sum(np.where(endpoints, trapezoids, 0.0), axis=1) / (positive * negative)

    limit = desired_fpr * negative
    before = np.max(np.where(endpoints & (false_positive <= limit), positions, -1), axis=1)
    after = np.min(np.where(endpoints & (positions > before[:, None]), positions, len(y)), axis=1)
    rows = np.arange(len(values))
    fp0, tp0 = fp_with_origin[rows, before + 1], tp_with_origin[rows, before + 1]
    fp1, tp1 = false_positive[rows, after], true_positive[rows, after]
    tpr = (tp0 + (tp1 - tp0) * (limit - fp0) / (fp1 - fp0)) / positive

    partial = np.sum(np.where(endpoints & (false_positive <= limit), trapezoids, 0.0),
                     axis=1) / (positive * negative)
    partial += (desired_fpr - fp0 / negative) * (tp0 / positive + tpr) * 0.5
    chance_area = 0.5 * desired_fpr ** 2
    pauc = 0.5 * (1.0 + (partial - chance_area) / (desired_fpr - chance_area))
    return {"auc": auc, "tpr": tpr, "pauc": pauc}


def weight_grid(model_count, random_count=5000, seed=340):
    """Enumerate the entire declared pair/triple grid plus seeded sparse draws.

    Singletons are evaluated separately across every available model, including
    models outside the ensemble pool. Sparse Dirichlet draws are a finite random
    sample of continuous weights, not exhaustion of every possible weighting.
    """
    if model_count < 1 or random_count < 0:
        raise ValueError("At least one model and a nonnegative random count are required.")
    weights = []
    for first, second in combinations(range(model_count), 2):
        for first_weight in (0.1, 0.25, 0.5, 0.75, 0.9):
            row = np.zeros(model_count)
            row[first], row[second] = first_weight, 1.0 - first_weight
            weights.append(row)
    for positions in combinations(range(model_count), 3):
        for values in ((1 / 3, 1 / 3, 1 / 3), (0.5, 0.25, 0.25),
                       (0.25, 0.5, 0.25), (0.25, 0.25, 0.5)):
            row = np.zeros(model_count)
            row[list(positions)] = values
            weights.append(row)
    if model_count >= 2:
        rng = np.random.default_rng(seed)
        for _ in range(random_count):
            size = int(rng.integers(2, min(5, model_count) + 1))
            positions = rng.choice(model_count, size, replace=False)
            row = np.zeros(model_count)
            row[positions] = rng.dirichlet(np.full(size, 2.0))
            weights.append(row)
    return np.array(weights, dtype=float).reshape(-1, model_count)


def family_key(record):
    spec = record["spec"]
    return f"{spec['source']}:{spec.get('params', {}).get('family', spec['source'])}"


def _order_key(summary, index):
    return (-float(summary["auc"][index] + TPR_WEIGHT * summary["tpr"][index]),
            -float(summary["auc"][index]), -float(summary["tpr"][index]), index)


def choose_pool(records, summary, baseline_index, limit=24):
    """Include current submission, then diverse high-objective probabilities.

    One representative per family is considered first, in objective order. Any
    remaining slots receive second representatives in that same order. No family
    supplies more than two models. Decision-only models remain singleton options.
    """
    if limit < 1:
        raise ValueError("The ensemble pool must have at least one slot.")
    probability = [i for i, record in enumerate(records)
                   if record["score_kind"] == "probability"]
    ordered = sorted(probability, key=lambda i: (*_order_key(summary, i)[:-1], records[i]["id"]))
    selected = [baseline_index]
    counts = {family_key(records[baseline_index]): 1}
    for maximum in (1, 2):
        for index in ordered:
            family = family_key(records[index])
            if index in selected or counts.get(family, 0) >= maximum:
                continue
            if len(selected) == limit:
                return selected
            selected.append(index)
            counts[family] = counts.get(family, 0) + 1
    return selected


def _validate_inputs(records, predictions, labels, splits):
    values = np.asarray(predictions, dtype=float)
    y = np.asarray(labels)
    if y.ndim != 1 or values.ndim != 2 or values.shape != (len(records), len(y)):
        raise ValueError("Predictions must align with records and global labels.")
    validation_sets = []
    for split in splits:
        ids = np.asarray(split["validation"] if isinstance(split, dict) else split, dtype=int)
        if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
            raise ValueError("Each validation fold must contain distinct row IDs.")
        if np.any(ids < 0) or np.any(ids >= len(y)):
            raise ValueError("Validation IDs exceed the global labels array.")
        if isinstance(split, dict) and "train" in split:
            if np.intersect1d(ids, np.asarray(split["train"], dtype=int)).size:
                raise ValueError("Training and validation IDs must not overlap.")
        if len(np.unique(y[ids])) != 2 or not np.isin(y[ids], [0, 1]).all():
            raise ValueError("Every validation fold must contain both binary classes.")
        validation_sets.append(ids)
    if not validation_sets:
        raise ValueError("At least one validation fold is required.")
    used = np.concatenate(validation_sets)
    if len(np.unique(used)) != len(used):
        raise ValueError("A single OOF prediction vector cannot represent overlapping validation folds.")
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Model IDs must be unique.")
    valid = []
    excluded = []
    for index, record in enumerate(records):
        reason = None
        if "error" in record:
            reason = "model evaluation failed"
        elif record.get("score_kind") not in ("probability", "decision"):
            reason = "unsupported score kind"
        elif not np.isfinite(values[index, used]).all():
            reason = "missing or nonfinite OOF predictions"
        elif record["score_kind"] == "probability" and (
                np.any(values[index, used] < 0) or np.any(values[index, used] > 1)):
            reason = "probability outside [0, 1]"
        if reason is None:
            valid.append(index)
        else:
            excluded.append({"id": record["id"], "reason": reason})
    kept = [records[i] for i in valid]
    baseline = [i for i, record in enumerate(kept) if record["spec"]["source"] == "current"]
    if len(baseline) != 1 or kept[baseline[0]]["score_kind"] != "probability":
        raise ValueError("Exactly one valid probability-producing current submission is required.")
    return kept, values[valid], y, validation_sets, baseline[0], excluded


def _metrics_on_folds(predictions, labels, validation_sets, batch_size):
    arrays = {key: np.empty((len(predictions), len(validation_sets)))
              for key in ("auc", "tpr", "pauc")}
    for fold, ids in enumerate(validation_sets):
        for start in range(0, len(predictions), batch_size):
            end = min(start + batch_size, len(predictions))
            values = batch_metrics(labels[ids], predictions[start:end, ids])
            for key in arrays:
                arrays[key][start:end, fold] = values[key]
    return arrays


def _recipe(records, positions, weights, method, fold_values, baseline_auc):
    means = {key: float(np.mean(value)) for key, value in fold_values.items()}
    active = np.flatnonzero(np.asarray(weights) > 0)
    return {
        "method": method,
        "model_ids": [records[positions[i]]["id"] for i in active],
        "specs": [records[positions[i]]["spec"] for i in active],
        "weights": [float(weights[i]) for i in active],
        **means,
        "objective": means["auc"] + TPR_WEIGHT * means["tpr"],
        "auc_guard_pass": means["auc"] >= baseline_auc,
        "folds": [{key: float(values[i]) for key, values in fold_values.items()}
                  for i in range(len(next(iter(fold_values.values()))))],
    }


def run(records, predictions, labels, splits, *, seed=340, pool_limit=24,
        random_count=5000, batch_size=128, top_n=20):
    """Search the fixed finite recipe library and return JSON-serializable results.

    ``predictions`` is models x global rows; NaNs outside the phase's validation
    rows are permitted. ``splits`` contains dicts with global validation IDs (and
    optionally train IDs), or arrays of validation IDs. Failed records can remain
    aligned with all-NaN prediction rows; they are explicitly reported excluded.
    Returned recipes contain model IDs, complete specs, weights, and method.
    """
    if batch_size < 1 or top_n < 1:
        raise ValueError("Batch size and top_n must be positive.")
    records, predictions, y, validation_sets, baseline_index, excluded = _validate_inputs(
        records, predictions, labels, splits)
    single_folds = _metrics_on_folds(predictions, y, validation_sets, batch_size)
    summary = {key: values.mean(axis=1) for key, values in single_folds.items()}
    baseline_auc = float(summary["auc"][baseline_index])
    singleton_recipes = [
        _recipe(records, [index], [1.0], record["score_kind"],
                {key: value[index] for key, value in single_folds.items()}, baseline_auc)
        for index, record in enumerate(records)
    ]
    baseline = singleton_recipes[baseline_index]
    chosen = baseline
    top = []
    evaluated = 0
    eligible = 0

    def consider(recipe):
        nonlocal chosen, top, evaluated, eligible
        evaluated += 1
        if not recipe["auc_guard_pass"]:
            return
        eligible += 1
        # A strict comparison retains baseline on exact objective ties.
        if recipe["objective"] > chosen["objective"]:
            chosen = recipe
        top.append(recipe)
        top.sort(key=lambda item: (-item["objective"], -item["auc"], -item["tpr"],
                                   len(item["model_ids"]), tuple(item["model_ids"]), item["method"]))
        del top[top_n:]

    # Start with baseline to make an exact tie favor the existing submission.
    consider(baseline)
    for index, recipe in enumerate(singleton_recipes):
        if index != baseline_index:
            consider(recipe)

    pool = choose_pool(records, summary, baseline_index, pool_limit)
    weights = weight_grid(len(pool), random_count=random_count, seed=seed)
    pool_predictions = predictions[pool]
    for method in ("probability", "logit"):
        basis = (pool_predictions if method == "probability" else
                 logit(np.clip(pool_predictions, 1e-6, 1 - 1e-6)))
        for start in range(0, len(weights), batch_size):
            batch = weights[start:start + batch_size]
            fold_values = {key: np.empty((len(batch), len(validation_sets)))
                           for key in ("auc", "tpr", "pauc")}
            for fold, ids in enumerate(validation_sets):
                scores = batch @ basis[:, ids]
                batch_values = batch_metrics(y[ids], scores)
                for key in fold_values:
                    fold_values[key][:, fold] = batch_values[key]
            # Only materialize detailed recipes that pass the AUC guard. All
            # evaluated rows still count, including rejected recipes.
            means = {key: value.mean(axis=1) for key, value in fold_values.items()}
            for i, weight in enumerate(batch):
                if means["auc"][i] < baseline_auc:
                    evaluated += 1
                    continue
                recipe = _recipe(records, pool, weight, method,
                                 {key: value[i] for key, value in fold_values.items()}, baseline_auc)
                consider(recipe)

    probability_singles = sum(record["score_kind"] == "probability" for record in records)
    return {
        "chosen": chosen,
        "baseline": baseline,
        "top_recipes": top,
        "best_singleton": max(singleton_recipes, key=lambda item: item["objective"]),
        "best_eligible_singleton": max((item for item in singleton_recipes if item["auc_guard_pass"]),
                                       key=lambda item: item["objective"]),
        "pool_model_ids": [records[index]["id"] for index in pool],
        "pool_families": [family_key(records[index]) for index in pool],
        "excluded_models": excluded,
        "counts": {
            "singleton_models": len(records),
            "probability_singletons": probability_singles,
            "decision_singletons": len(records) - probability_singles,
            "pool_models": len(pool),
            "weight_vectors": len(weights),
            "blend_methods": 2,
            "blend_evaluations": 2 * len(weights),
            "total_recipe_evaluations": evaluated,
            "auc_guard_passes": eligible,
        },
        "protocol": {
            "target_fpr": TARGET_FPR,
            "objective": "mean_fold_auc + 0.05 * mean_fold_tpr",
            "auc_guard": "mean_fold_auc >= current_submission_mean_fold_auc",
            "pool_limit": pool_limit, "max_models_per_family": 2,
            "random_sparse_draws": random_count if len(pool) >= 2 else 0,
            "random_seed": seed,
            "logit_clip": [1e-6, 1 - 1e-6],
            "stacking": "Not evaluated: supplied OOF scores are insufficient for clean additional meta-crossfitting.",
            "interpretation": "Selection on these labels is exploratory; nested outer scoring must be separate.",
        },
    }


def self_test():
    """Check ties, partial AUC, pool constraints, and baseline guard synthetically."""
    from collections import Counter
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.default_rng(340)
    checks = 0
    for size in (11, 47, 300):
        y = np.tile([0, 1], (size + 1) // 2)[:size]
        rng.shuffle(y)
        scores = np.vstack((rng.normal(size=size), rng.integers(0, 4, size=size),
                            np.ones(size), y, -y, np.arange(size)))
        result = batch_metrics(y, scores)
        for index, row in enumerate(scores):
            fpr, tpr, _ = roc_curve(y, row)
            np.testing.assert_allclose(result["auc"][index], roc_auc_score(y, row), atol=1e-12, rtol=0)
            np.testing.assert_allclose(result["tpr"][index], np.interp(TARGET_FPR, fpr, tpr), atol=1e-12, rtol=0)
            np.testing.assert_allclose(result["pauc"][index], roc_auc_score(y, row, max_fpr=TARGET_FPR),
                                       atol=1e-12, rtol=0)
            checks += 3
    weights = weight_grid(24)
    assert len(weights) == 14476
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-14, rtol=0)
    assert np.all(weights >= 0) and np.all((weights > 0).sum(axis=1) <= 5)
    np.testing.assert_array_equal(weight_grid(4, 7), weight_grid(4, 7))

    y = np.tile([0, 1], 60)
    records = [{"id": "current", "spec": {"source": "current", "params": {}},
                "score_kind": "probability"}]
    predictions = [0.2 + 0.6 * y]
    for index in range(9):
        records.append({"id": f"model-{index}", "spec": {
            "source": "archive", "params": {"family": f"group-{index // 3}"}},
            "score_kind": "probability"})
        predictions.append(rng.random(len(y)))
    records.append({"id": "decision", "spec": {"source": "archive", "params": {"family": "linear"}},
                    "score_kind": "decision"})
    predictions.append(1000.0 * y)
    predictions = np.array(predictions)
    before = predictions.copy()
    splits = [{"validation": list(range(start, start + 40))} for start in (0, 40, 80)]
    result = run(records, predictions, y, splits, random_count=10, batch_size=9)
    assert result["chosen"]["model_ids"] == ["current"]
    assert result["chosen"]["auc_guard_pass"]
    assert max(Counter(result["pool_families"]).values()) <= 2
    assert "decision" not in result["pool_model_ids"]
    assert result["counts"]["singleton_models"] == 11
    assert result["counts"]["total_recipe_evaluations"] == 11 + 2 * result["counts"]["weight_vectors"]
    np.testing.assert_array_equal(predictions, before)
    assert result["baseline"]["auc"] == 1.0
    # Nested phase arrays retain global row numbering and deliberately leave
    # non-phase rows NaN. Only the union of validation IDs must be finite.
    nested = run(records, np.pad(predictions, ((0, 0), (0, 20)), constant_values=np.nan),
                 np.r_[y, np.tile([0, 1], 10)], splits, random_count=0)
    assert nested["chosen"]["model_ids"] == ["current"]
    return {"metric_comparisons": checks, "weight_vectors_for_24_models": len(weights),
            "pool_constraints": "passed", "baseline_tie_retention": "passed",
            "inputs_unchanged": "passed", "nan_outside_phase": "passed"}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic metric and selector checks.")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        parser.error("Import run(...) for search, or supply --self-test.")
