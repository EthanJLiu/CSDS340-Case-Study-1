"""Paired stability comparison of recipes frozen by the completed screen.

Run only after ``selection_screen.json`` exists:
    .venv/bin/python repeated_validation.py --jobs 2

All 25 splits and the promotion rule come from stability_protocol.json. This
script never changes a recipe or tunes a parameter using its validation scores.
These are reused development data, not an independent test of hidden ranking.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from scipy.special import logit

import search


HERE = Path(__file__).resolve().parent
DESTINATION = HERE / "results" / "repeated_validation"
MODEL_NAMES = ("baseline", "compatible", "all_libraries")
METRICS = ("auc", "tpr", "pauc")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blend(recipe, member_scores):
    """Match the screen's raw-score arithmetic exactly, including logit space."""
    weights = np.asarray(recipe["weights"], dtype=float)
    if weights.ndim != 1 or len(weights) != len(recipe["model_ids"]):
        raise ValueError("Recipe weights must align with member IDs.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Recipe requires finite nonnegative weights summing to one.")
    values = np.array([member_scores[identifier] for identifier in recipe["model_ids"]])
    method = recipe["method"]
    if method == "decision":
        if len(values) != 1 or weights[0] != 1:
            raise ValueError("Raw decision scores may only be used as singletons.")
        return values[0].copy()
    if method not in ("probability", "logit"):
        raise ValueError(f"Unsupported recipe method: {method}")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Probability blends require finite probabilities in [0, 1].")
    basis = values if method == "probability" else logit(np.clip(values, 1e-6, 1 - 1e-6))
    return weights @ basis


def load_inputs():
    protocol = json.loads((HERE / "protocol.json").read_text())
    search.validate_provenance(protocol)
    stability = json.loads((HERE / "stability_protocol.json").read_text())
    selection_path = HERE / "selection_screen.json"
    if not selection_path.exists():
        raise FileNotFoundError("Complete screen selection before starting repeated validation.")
    selected = json.loads(selection_path.read_text())
    if selected["phase"] != "screen":
        raise ValueError("Only frozen screen-selected recipes belong in this comparison.")
    specs = {spec["id"]: spec for spec in protocol["specs"]}
    current = [spec for spec in protocol["specs"] if spec["source"] == "current"]
    if len(current) != 1:
        raise ValueError("Exactly one original submission is required.")
    baseline = {"model_ids": [current[0]["id"]], "specs": current,
                "weights": [1.0], "method": "probability"}
    recipes = {"baseline": baseline, **{
        name: selected["procedures"][name]["chosen"]
        for name in ("compatible", "all_libraries")}}
    for name, recipe in recipes.items():
        if len(recipe["model_ids"]) != len(recipe["specs"]):
            raise ValueError(f"Incomplete recipe specifications: {name}")
        for identifier, spec in zip(recipe["model_ids"], recipe["specs"]):
            if identifier not in specs or spec != specs[identifier]:
                raise ValueError(f"Recipe differs from frozen model library: {name}/{identifier}")
    raw = search.data()
    universe = set(range(len(raw)))
    for seed in stability["seeds"]:
        folds = [fold for fold in stability["folds"] if fold["seed"] == seed]
        seen = []
        if len(folds) != 5:
            raise ValueError("Every stability repetition must contain five folds.")
        for fold in folds:
            training, validation = set(fold["train"]), set(fold["validation"])
            if training & validation or training | validation != universe:
                raise ValueError("Stability train/validation partitions overlap or omit rows.")
            if len(training) != len(fold["train"]) or len(validation) != len(fold["validation"]):
                raise ValueError("Repeated row ID inside a stability fold.")
            seen.extend(fold["validation"])
        if len(seen) != len(raw) or set(seen) != universe:
            raise ValueError("Each row must be scored exactly once per repetition.")
    if len(stability["folds"]) != 25:
        raise ValueError("This frozen comparison requires exactly 25 folds.")
    provenance = {path.name: sha256(path) for path in (
        HERE / "protocol.json", HERE / "stability_protocol.json", selection_path,
        HERE / "search.py", HERE / "ensemble_search.py", HERE / "assess_search.py",
        Path(__file__).resolve(),
    )}
    provenance["recipes"] = recipes
    provenance["frozen_model_and_data_hashes"] = protocol["sha256"]
    return raw, recipes, specs, stability, provenance


def run_fold(index, fold, raw, recipes, specifications, provenance):
    metadata_path = DESTINATION / f"fold_{index:02d}.json"
    scores_path = DESTINATION / f"fold_{index:02d}.npz"
    if metadata_path.exists():
        record = json.loads(metadata_path.read_text())
        if record["provenance"] != provenance:
            raise ValueError(f"Cached fold {index} belongs to different inputs or code.")
        if record["seed"] != fold["seed"] or record["fold"] != fold["fold"]:
            raise ValueError("Cached fold numbering changed.")
        with np.load(scores_path, allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached["validation_ids"], fold["validation"])
            scores = cached["recipe_scores"].copy()
        return record, scores
    train, validation = raw[fold["train"]], raw[fold["validation"]]
    identifiers = sorted({identifier for recipe in recipes.values() for identifier in recipe["model_ids"]})
    predictions, members = {}, {}
    started = time.perf_counter()
    for identifier in identifiers:
        beginning = time.perf_counter()
        notices = []
        try:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                scores = search.fit_scores(specifications[identifier], train[:, :-1],
                                           train[:, -1], validation[:, :-1])
            scores = np.asarray(scores, dtype=float)
            if scores.shape != (len(validation),) or not np.isfinite(scores).all():
                raise ValueError("Invalid score shape or nonfinite predictions.")
            predictions[identifier] = scores
            members[identifier] = {"seconds": time.perf_counter() - beginning}
        except Exception as exc:
            members[identifier] = {"seconds": time.perf_counter() - beginning,
                                   "error": f"{type(exc).__name__}: {exc}"}
        members[identifier]["warnings"] = sorted({
            f"{notice.category.__name__}: {notice.message}" for notice in notices})
    outputs = np.full((len(MODEL_NAMES), len(validation)), np.nan)
    results = {}
    for model_index, name in enumerate(MODEL_NAMES):
        failed = [identifier for identifier in recipes[name]["model_ids"] if identifier not in predictions]
        if failed:
            results[name] = {"error": "Required member fits failed", "member_ids": failed}
            continue
        outputs[model_index] = blend(recipes[name], predictions)
        results[name] = search.metrics(validation[:, -1], outputs[model_index])
    record = {"index": index, "seed": fold["seed"], "fold": fold["fold"],
              "training_rows": len(train), "validation_rows": len(validation),
              "metrics": results, "members": members,
              "seconds": time.perf_counter() - started, "provenance": provenance}
    member_outputs = np.array([predictions.get(identifier, np.full(len(validation), np.nan))
                               for identifier in identifiers])
    np.savez_compressed(scores_path, validation_ids=np.asarray(fold["validation"]),
                        member_ids=np.array(identifiers), member_scores=member_outputs,
                        model_names=np.array(MODEL_NAMES), recipe_scores=outputs)
    search.write_json(metadata_path, record)
    return record, outputs


def summarize(records, model_names, seeds):
    summary = {}
    for name in model_names:
        selected = [record["metrics"][name] for record in records]
        failed = [record["index"] for record in records if "error" in record["metrics"][name]]
        item = {"complete": not failed, "failed_fold_indices": failed}
        if failed:
            summary[name] = item
            continue
        item["mean_fold_metrics"] = {key: float(np.mean([row[key] for row in selected])) for key in METRICS}
        item["fold_standard_deviations"] = {
            key: float(np.std([row[key] for row in selected], ddof=1)) for key in METRICS}
        repetitions = []
        for seed in seeds:
            subset = [record for record in records if record["seed"] == seed]
            entry = {"seed": seed, **{
                key: float(np.mean([record["metrics"][name][key] for record in subset])) for key in METRICS}}
            if all("error" not in record["metrics"]["baseline"] for record in subset):
                entry["delta_from_baseline"] = {key: float(np.mean([
                    record["metrics"][name][key] - record["metrics"]["baseline"][key]
                    for record in subset])) for key in METRICS}
            repetitions.append(entry)
        item["repetitions"] = repetitions
        if all("error" not in record["metrics"]["baseline"] for record in records):
            differences = {key: np.array([record["metrics"][name][key] -
                                          record["metrics"]["baseline"][key] for record in records])
                           for key in METRICS}
            item["paired_comparison"] = {
                key: {"mean_delta": float(values.mean()),
                      "standard_deviation_of_fold_deltas": float(values.std(ddof=1)),
                      "wins": int(np.sum(values > 1e-12)),
                      "ties": int(np.sum(np.abs(values) <= 1e-12)),
                      "losses": int(np.sum(values < -1e-12)),
                      "repetition_wins": sum(row["delta_from_baseline"][key] > 1e-12
                                             for row in repetitions)}
                for key, values in differences.items()}
            item["repetition_wins_both_metrics"] = sum(
                all(row["delta_from_baseline"][key] > 1e-12 for key in ("auc", "tpr"))
                for row in repetitions)
            item["passes_repeated_portion_of_promotion_rule"] = (
                all(item["paired_comparison"][key]["mean_delta"] > 0 for key in ("auc", "tpr"))
                and item["repetition_wins_both_metrics"] >= 4)
        summary[name] = item
    return summary


def main(jobs=2):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    raw, recipes, specifications, stability, provenance = load_inputs()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = DESTINATION / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("Refusing to reuse stability caches after changing frozen inputs or code.")
    if not manifest.exists():
        search.write_json(manifest, provenance)
    started = time.perf_counter()
    completed = []
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(run_fold)(index, fold, raw, recipes, specifications, provenance)
            for index, fold in enumerate(stability["folds"]))
        for record, scores in results:
            completed.append((record, scores))
            print(json.dumps({"completed_folds": len(completed), "total_folds": 25,
                              "seconds": round(time.perf_counter() - started, 2),
                              "failed_recipes_this_fold": sum("error" in row for row in record["metrics"].values())}),
                  flush=True)
    completed.sort(key=lambda pair: pair[0]["index"])
    records = [record for record, _ in completed]
    predictions = np.full((len(MODEL_NAMES), len(stability["seeds"]), len(raw)), np.nan)
    for record, scores in completed:
        repeat = stability["seeds"].index(record["seed"])
        validation_ids = stability["folds"][record["index"]]["validation"]
        predictions[:, repeat, validation_ids] = scores
    report = {
        "purpose": stability["purpose"], "promotion_rule": stability["promotion_rule"],
        "provenance": provenance, "recipes": recipes,
        "model_order": list(MODEL_NAMES), "repetition_seeds": stability["seeds"],
        "prediction_shape": list(predictions.shape),
        "summary": summarize(records, MODEL_NAMES, stability["seeds"]),
        "folds": [{key: value for key, value in record.items() if key != "provenance"} for record in records],
        "seconds_this_invocation": time.perf_counter() - started,
        "sum_recorded_fold_seconds": sum(record["seconds"] for record in records),
        "promotion_requires_additional_nested_gate": True,
        "interpretation": "Fixed recipes on reused development rows. Fold and repetition results "
                          "are paired stability diagnostics, not independent trials or an unbiased "
                          "estimate of final leaderboard performance. Do not choose new recipes from these results.",
    }
    np.savez_compressed(HERE / "repeated_predictions.npz", scores=predictions,
                        model_names=np.array(MODEL_NAMES), seeds=np.array(stability["seeds"]),
                        labels=raw[:, -1], row_ids=np.arange(len(raw)))
    search.write_json(HERE / "repeated_results.json", report)
    print(json.dumps({"summary": report["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    main(args.jobs)
