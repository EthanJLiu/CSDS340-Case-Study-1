"""Assess completed searches without using outer labels to select recipes.

Examples (use the task venv and its documented OpenMP library environment):
    python assess_search.py --phase screen
    python assess_search.py --phase nested_0
    python assess_search.py --phase nested_summary
    python assess_search.py --phase nested

The screen is exploratory. Nested evaluation independently repeats the selector
on each outer training partition's inner OOF predictions. Historical reuse of
these released emails remains a limitation of both analyses.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.special import logit

import ensemble_search
import search


HERE = Path(__file__).resolve().parent
PROCEDURES = ("compatible", "all_libraries")
METRICS = ("auc", "tpr", "pauc")


def protocol():
    value = json.loads((HERE / "protocol.json").read_text())
    search.validate_provenance(value)
    return value


def _read_phase(phase, plan):
    if phase not in plan["phases"]:
        raise ValueError(f"Unknown phase: {phase}")
    ids = plan["phases"][phase]["ids"]
    directory = HERE / "results" / phase
    missing = [model_id for model_id in ids
               if not (directory / f"{model_id}.json").is_file()]
    if missing:
        raise RuntimeError(f"Phase {phase} is incomplete: {len(missing)} of "
                           f"{len(ids)} model records missing; first IDs: {missing[:5]}")
    specs = {spec["id"]: spec for spec in plan["specs"]}
    rows = []
    for model_id in ids:
        record = json.loads((directory / f"{model_id}.json").read_text())
        if record.get("id") != model_id or record.get("spec") != specs[model_id]:
            raise ValueError(f"Cached record/spec mismatch: {phase}/{model_id}")
        rows.append(record)
    return rows


def load_phase(phase):
    """Return successful records and model x 3000 global-row OOF predictions.

    All expected JSON records must exist, including failure records. Failures
    are excluded here and are reported by ``phase_audit`` in saved artifacts.
    Predictions outside a nested phase's outer training partition must be NaN.
    """
    plan = protocol()
    records = _read_phase(phase, plan)
    splits = plan["phases"][phase]["splits"]
    validation = np.concatenate([np.asarray(s["validation"], dtype=int) for s in splits])
    if len(np.unique(validation)) != len(validation):
        raise ValueError("Validation rows overlap in this OOF phase.")
    row_count = len(search.data())
    outside = np.ones(row_count, dtype=bool)
    outside[validation] = False
    kept, predictions = [], []
    for record in records:
        if "error" in record:
            continue
        if len(record.get("folds", [])) != len(splits):
            raise ValueError(f"Incomplete fold metrics: {record['id']}")
        path = HERE / "results" / phase / f"{record['id']}.npz"
        with np.load(path, allow_pickle=False) as saved:
            scores = np.asarray(saved["scores"], dtype=float)
        if scores.shape != (row_count,) or not np.isfinite(scores[validation]).all():
            raise ValueError(f"Invalid OOF predictions: {record['id']}")
        if not np.isnan(scores[outside]).all():
            raise ValueError(f"Predictions exist outside the allowed phase rows: {record['id']}")
        if record.get("score_kind") == "probability" and (
                np.any(scores[validation] < 0) or np.any(scores[validation] > 1)):
            raise ValueError(f"Probability outside [0, 1]: {record['id']}")
        kept.append(record)
        predictions.append(scores)
    if not kept:
        raise ValueError(f"No successful models in {phase}")
    return kept, np.asarray(predictions)


def phase_audit(phase, plan=None):
    plan = protocol() if plan is None else plan
    records = _read_phase(phase, plan)
    failures = [{"id": r["id"], "spec": r["spec"], "error": r["error"]}
                for r in records if "error" in r]
    warning_records = [{"id": r["id"], "spec": r["spec"], "warnings": r["warnings"]}
                       for r in records if r.get("warnings")]
    return {"expected_models": len(records), "successful_models": len(records) - len(failures),
            "failed_models": len(failures), "failures": failures,
            "models_with_warnings": len(warning_records), "warnings": warning_records,
            "warning_counts": dict(Counter(w for r in records for w in set(r.get("warnings", []))))}


def _select(phase, records, scores, labels, plan):
    selections = {}
    for name in PROCEDURES:
        positions = [i for i, r in enumerate(records)
                     if name == "all_libraries" or r["spec"]["source"] != "external"]
        subset = [records[i] for i in positions]
        result = ensemble_search.run(subset, scores[positions], labels,
                                     plan["phases"][phase]["splits"])
        # The complete recipe manifest is represented exactly by these arrays:
        # every singleton, then each row of weights in probability and logit space.
        weights = ensemble_search.weight_grid(
            len(result["pool_model_ids"]),
            random_count=result["protocol"]["random_sparse_draws"],
            seed=result["protocol"]["random_seed"],
        )
        filename = f"recipes_{phase}_{name}.npz"
        np.savez_compressed(
            HERE / filename, weights=weights,
            pool_model_ids=np.array(result["pool_model_ids"]),
            singleton_model_ids=np.array([r["id"] for r in subset]),
            singleton_methods=np.array([r["score_kind"] for r in subset]),
            blend_methods=np.array(["probability", "logit"]),
        )
        result["recipe_manifest"] = {
            "file": filename,
            "description": "Every singleton ID/method; each weight row applied to "
                           "pool_model_ids in each listed blend method. Full specs in protocol.json.",
            "model_scope": "No external libraries" if name == "compatible" else "All available libraries",
        }
        selections[name] = result
    artifact = {"phase": phase, "procedures": selections,
                "audit": phase_audit(phase, plan),
                "protocol_sha256": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
                "selector_sha256": hashlib.sha256((HERE / "ensemble_search.py").read_bytes()).hexdigest(),
                "interpretation": "Scores in these recipes are inner/development selection scores, "
                                  "not evaluation on the outer fold or final hidden test."}
    search.write_json(HERE / f"selection_{phase}.json", artifact)
    return selections


def screen():
    plan = protocol()
    records, scores = load_phase("screen")
    raw = search.data()
    selection = _select("screen", records, scores, raw[:, -1], plan)
    families = defaultdict(list)
    for record in _read_phase("screen", plan):
        families[ensemble_search.family_key(record)].append(record)
    summary = []
    for family, rows in sorted(families.items()):
        valid = [r for r in rows if "error" not in r]
        item = {"family": family, "attempted": len(rows), "successful": len(valid),
                "failed": len(rows) - len(valid),
                "warnings": dict(Counter(w for r in rows for w in set(r.get("warnings", []))))}
        if valid:
            for metric in ("auc", "tpr"):
                best = max(valid, key=lambda r: (r[metric], r["auc"], r["tpr"], r["id"]))
                item[f"best_{metric}"] = {"id": best["id"], "spec": best["spec"],
                                         **{m: best[m] for m in METRICS},
                                         "warnings": best.get("warnings", [])}
        summary.append(item)
    search.write_json(HERE / "screen_family_summary.json", {
        "families": summary, "audit": phase_audit("screen", plan),
        "interpretation": "Exploratory maxima selected on the same development folds; "
                          "these maxima are not independent performance estimates.",
    })
    print(json.dumps({"phase": "screen", "selected": {
        name: {key: result["chosen"][key] for key in ("method", "model_ids", "auc", "tpr")}
        for name, result in selection.items()}}), flush=True)


def blend(recipe, predictions):
    """Apply a frozen recipe without labels or fitting any calibration."""
    rows = np.asarray([predictions[model_id] for model_id in recipe["model_ids"]])
    weights = np.asarray(recipe["weights"], dtype=float)
    if len(rows) != len(weights) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid recipe weights")
    if recipe["method"] == "logit":
        # Keep the exact score space used by ensemble_search. A final sigmoid
        # would preserve ordering mathematically but can introduce float ties.
        return weights @ logit(np.clip(rows, 1e-6, 1 - 1e-6))
    if recipe["method"] == "decision":
        if len(rows) != 1 or weights[0] != 1:
            raise ValueError("Raw-margin models must remain singletons")
        return rows[0]
    if recipe["method"] != "probability":
        raise ValueError(f"Unknown method: {recipe['method']}")
    return weights @ rows


def nested_fold(index):
    phase = f"nested_{index}"
    plan = protocol()
    records, scores = load_phase(phase)
    outer = plan["outer_splits"][index]
    train_ids = np.asarray(outer["train"], dtype=int)
    validation_ids = np.asarray(outer["validation"], dtype=int)
    inner_ids = np.concatenate([s["validation"] for s in plan["phases"][phase]["splits"]])
    if set(inner_ids) != set(train_ids) or set(train_ids) & set(validation_ids):
        raise ValueError("Inner folds do not partition the outer training rows")
    raw = search.data()
    # The selector receives no outer labels, even though arrays retain global IDs.
    inner_labels = np.full(len(raw), np.nan)
    inner_labels[train_ids] = raw[train_ids, -1]
    selections = _select(phase, records, scores, inner_labels, plan)
    # Both selections are now frozen and saved. Only their members are refitted.
    current = next(r["spec"] for r in records if r["spec"]["source"] == "current")
    members = {current["id"]: current}
    for result in selections.values():
        for spec in result["chosen"]["specs"]:
            members[spec["id"]] = spec
    outputs, fits = {}, {}
    for model_id, spec in members.items():
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                values = np.asarray(search.fit_scores(
                    spec, raw[train_ids, :-1], raw[train_ids, -1], raw[validation_ids, :-1]), dtype=float)
            if values.shape != (len(validation_ids),) or not np.isfinite(values).all():
                raise ValueError("Invalid outer prediction shape or nonfinite score")
            outputs[model_id] = values
            fits[model_id] = {"seconds": time.perf_counter() - started,
                              "warnings": sorted({str(w.message) for w in notices
                                                   if not issubclass(w.category, FutureWarning)})}
        except Exception as exc:
            fits[model_id] = {"seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"}
    # First access to the outer labels happens after all choices and predictions.
    outer_labels = raw[validation_ids, -1]
    saved = {"validation_ids": validation_ids}
    report = {"phase": phase, "training_rows": len(train_ids),
              "validation_rows": len(validation_ids), "fitted_members": fits,
              "procedures": {}, "selection_artifact": f"selection_{phase}.json"}
    if current["id"] not in outputs:
        report["baseline_error"] = fits[current["id"]]["error"]
    else:
        baseline_scores = outputs[current["id"]]
        report["baseline"] = search.metrics(outer_labels, baseline_scores)
        saved["baseline"] = np.full(len(raw), np.nan)
        saved["baseline"][validation_ids] = baseline_scores
    for name, result in selections.items():
        recipe = result["chosen"]
        failed = [model_id for model_id in recipe["model_ids"] if model_id not in outputs]
        entry = {"recipe": recipe, "selection_counts": result["counts"]}
        if failed:
            entry["error"] = "Selected members failed outer refit; no fallback was selected."
            entry["failed_members"] = failed
        else:
            values = blend(recipe, outputs)
            entry["outer_metrics"] = search.metrics(outer_labels, values)
            if "baseline" in report:
                entry["delta_from_baseline"] = {
                    key: entry["outer_metrics"][key] - report["baseline"][key] for key in METRICS}
            saved[name] = np.full(len(raw), np.nan)
            saved[name][validation_ids] = values
        report["procedures"][name] = entry
    np.savez_compressed(HERE / f"{phase}_outer_predictions.npz", **saved)
    search.write_json(HERE / f"{phase}_assessment.json", report)
    print(json.dumps({"phase": phase, "baseline": report.get("baseline"),
                      "procedures": {name: {k: v for k, v in entry.items() if k != "recipe"}
                                     for name, entry in report["procedures"].items()}}), flush=True)
    return report


def summarize_nested():
    plan = protocol()
    paths = [HERE / f"nested_{i}_assessment.json" for i in range(len(plan["outer_splits"]))]
    if not all(p.is_file() for p in paths):
        raise RuntimeError("All outer fold assessments must finish before summary")
    reports = [json.loads(p.read_text()) for p in paths]
    expected_protocol_hash = hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    expected_selector_hash = hashlib.sha256((HERE / "ensemble_search.py").read_bytes()).hexdigest()
    for index in range(len(reports)):
        selection = json.loads((HERE / f"selection_nested_{index}.json").read_text())
        if (selection["protocol_sha256"] != expected_protocol_hash or
                selection["selector_sha256"] != expected_selector_hash):
            raise ValueError("Nested assessments refer to different protocol or selector sources")
    raw = search.data()
    outputs = {name: np.full(len(raw), np.nan) for name in ("baseline", *PROCEDURES)}
    filled = np.zeros(len(raw), dtype=bool)
    for index, report in enumerate(reports):
        with np.load(HERE / f"nested_{index}_outer_predictions.npz", allow_pickle=False) as saved:
            ids = saved["validation_ids"]
            if not np.array_equal(ids, plan["outer_splits"][index]["validation"]):
                raise ValueError("Outer prediction IDs differ from frozen protocol")
            if filled[ids].any():
                raise ValueError("Outer validation folds overlap")
            filled[ids] = True
            for name in outputs:
                if name in saved:
                    outputs[name][ids] = saved[name][ids]
    if not filled.all():
        raise ValueError("Outer folds fail to cover all emails")
    summary = {"outer_folds": len(reports), "folds": reports, "procedures": {},
               "interpretation": "Mean fold metrics evaluate the search procedure. Pooled OOF "
                                 "metrics are separately labelled and mix different fitted recipes. "
                                 "All released emails have historical development reuse; this is "
                                 "not a pristine estimate of the final hidden leaderboard."}
    if all("baseline" in report for report in reports):
        summary["baseline_mean_fold_metrics"] = {
            key: float(np.mean([r["baseline"][key] for r in reports])) for key in METRICS}
    for name in PROCEDURES:
        entries = [r["procedures"][name] for r in reports]
        if any("error" in entry for entry in entries):
            summary["procedures"][name] = {"error": "At least one outer refit failed",
                                           "failed_folds": [i for i, e in enumerate(entries) if "error" in e]}
            continue
        means = {key: float(np.mean([e["outer_metrics"][key] for e in entries])) for key in METRICS}
        result = {"mean_fold_metrics": means,
                  "pooled_oof_metrics": search.metrics(raw[:, -1], outputs[name]),
                  "recipe_evaluations": sum(e["selection_counts"]["total_recipe_evaluations"] for e in entries)}
        if "baseline_mean_fold_metrics" in summary:
            result["mean_delta_from_baseline"] = {
                key: means[key] - summary["baseline_mean_fold_metrics"][key] for key in METRICS}
            result["fold_wins"] = {
                key: sum(e["outer_metrics"][key] > r["baseline"][key] for e, r in zip(entries, reports))
                for key in METRICS}
        summary["procedures"][name] = result
    if np.isfinite(outputs["baseline"]).all():
        summary["baseline_pooled_oof_metrics"] = search.metrics(raw[:, -1], outputs["baseline"])
    np.savez_compressed(HERE / "nested_outer_predictions.npz", **outputs)
    search.write_json(HERE / "nested_assessment.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "folds"}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=("screen", "nested", "nested_0", "nested_1", "nested_2", "nested_summary"))
    arguments = parser.parse_args()
    if arguments.phase == "screen":
        screen()
    elif arguments.phase == "nested_summary":
        summarize_nested()
    elif arguments.phase == "nested":
        for fold_index in range(len(protocol()["outer_splits"])):
            nested_fold(fold_index)
        summarize_nested()
    else:
        nested_fold(int(arguments.phase.rsplit("_", 1)[1]))
