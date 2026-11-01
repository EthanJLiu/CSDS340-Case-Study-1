"""Reproducible finite broad search and independently specified nested library.

Run with the local .venv/bin/python. This never edits the submission.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
import time
import warnings

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed, parallel_backend

import archived_models
import additional_models
import external_models
from baseline_classifier import predictTest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def canonical(value):
    if isinstance(value, BaseEstimator):
        return {"class": type(value).__module__ + "." + type(value).__name__,
                "parameters": canonical(value.get_params(deep=False))}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=True).encode()).hexdigest()[:16]


def build(spec):
    source, params = spec["source"], spec["params"]
    if source == "current" or (source == "archive" and params["family"] == "original"):
        return None
    module = {"archive": archived_models, "additional": additional_models,
              "external": external_models}[source]
    return module.build(params)


def data():
    return np.concatenate([np.loadtxt(ROOT / name, delimiter=",")
                           for name in ("spamTrain1.csv", "spamTrain2.csv")])


def metrics(y, scores):
    fpr, tpr, _ = roc_curve(y, scores)
    return {"auc": float(roc_auc_score(y, scores)),
            "tpr": float(np.interp(.01, fpr, tpr)),
            "pauc": float(roc_auc_score(y, scores, max_fpr=.01))}


@contextmanager
def time_limit(seconds):
    def expired(signum, frame):
        raise TimeoutError(f"Exceeded the prespecified {seconds}s per-fit limit")
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def initialize():
    raw = [{"source": "current", "params": {"family": "current_submission"}}]
    for source, configs in (("archive", archived_models.FULL_LIBRARY),
                            ("additional", additional_models.get_configs()),
                            ("external", external_models.get_configs())):
        raw += [{"source": source, "params": c} for c in configs]
    specs, seen, aliases = [], {}, []
    for spec in raw:
        spec["id"] = digest(spec)
        try:
            model = build(spec)
            fingerprint = digest(canonical(model)) if model is not None else spec["id"]
        except Exception:
            # Keep unavailable libraries so their actual errors are recorded.
            fingerprint = spec["id"]
        if fingerprint in seen:
            aliases.append({"alias": spec, "equivalent_to": seen[fingerprint]})
        else:
            seen[fingerprint] = spec["id"]
            specs.append(spec)
    family_groups = defaultdict(list)
    for spec in specs:
        family_groups[(spec["source"], spec["params"]["family"])].append(spec)
    # Prespecified coverage-based nested library, selected BEFORE any scores.
    nested = []
    for group in family_groups.values():
        nested += sorted(group, key=lambda c: c["id"])[:6]
    # Explicitly include every component of the current submission.
    components = [
        {"family": "advanced_bins", "bins": 2, "spacing": "log", "presence": True,
         "missing": "prior", "C": 1},
        {"family": "knn_svm", "transform": "log", "neighbors": 31, "C": 10., "gamma": .003},
        {"family": "interactions", "features": 60, "interaction_only": True, "C": .001},
    ]
    for params in components:
        target = digest(canonical(archived_models.build(params)))
        for spec in specs:
            if spec["source"] == "archive" and spec["params"]["family"] == params["family"]:
                if digest(canonical(build(spec))) == target:
                    nested.append(spec)
                    break
    nested_ids = sorted({s["id"] for s in nested})
    raw_data = data()
    y = raw_data[:, -1]
    source_labels = np.repeat([0, 1], [1500, 1500])
    strata = (2 * source_labels + y).astype(int)
    phases, outer = {}, []
    for name, seed in [("screen", 20261101)]:
        splits = [{"train": tr.tolist(), "validation": va.tolist()} for tr, va in
                  StratifiedKFold(3, shuffle=True, random_state=seed).split(y, strata)]
        phases[name] = {"seed": seed, "splits": splits, "ids": [s["id"] for s in specs]}
    for i, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=9042026).split(y, strata)):
        splits = [{"train": tr[a].tolist(), "validation": tr[b].tolist()} for a, b in
                  StratifiedKFold(3, shuffle=True, random_state=11107 + i).split(tr, strata[tr])]
        phases[f"nested_{i}"] = {"seed": 11107 + i, "splits": splits, "ids": nested_ids}
        outer.append({"train": tr.tolist(), "validation": va.tolist()})
    provenance = {}
    for path in [ROOT / "spamTrain1.csv", ROOT / "spamTrain2.csv", HERE / "archived_models.py",
                 HERE / "additional_models.py", HERE / "external_models.py", HERE / "baseline_classifier.py"]:
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {"version": 1, "specs": specs, "aliases": aliases, "phases": phases,
                "outer_splits": outer, "nested_ids": nested_ids, "sha256": provenance,
                "sklearn": sklearn.__version__, "max_fit_seconds": 45,
                "max_preprocessing_seconds": 120,
                "selector": "Among mean-AUC >= current baseline, maximize mean-AUC + .05*mean-TPR; "
                            "include all singleton models and fixed diverse probability/logit blend grid.",
                "limitations": "All3000 rows now development; nested grid fixed by family/hash before scores; "
                                "historical algorithm development reused these rows. "
                                "Full screen is exploratory and broader than nested library."}
    path = HERE / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Refusing to overwrite a different frozen protocol")
    else:
        write_json(path, protocol)
    print(json.dumps({"raw_configs": len(raw), "unique_pipelines": len(specs),
                      "aliases": len(aliases), "nested_configs": len(nested_ids),
                      "families": len(family_groups)}), flush=True)


def validate_provenance(protocol):
    for relative, expected in protocol["sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input changed: {relative}")


def fit_scores(spec, X, y, Z):
    with threadpool_limits(limits=1), time_limit(45):
        model = build(spec)
        if model is None:
            predictor = predictTest if spec["source"] == "current" else archived_models.original_predict
            return predictor(X, y, Z)
        model.fit(X, y)
        return (model.predict_proba(Z)[:, 1] if hasattr(model, "predict_proba")
                else model.decision_function(Z))


def run_group(specs, phase, phase_spec, raw):
    destination = HERE / "results" / phase
    records = {s["id"]: {"id": s["id"], "spec": s, "folds": [], "warnings": [],
                          "score_kind": "probability", "seconds": 0.} for s in specs}
    outputs = {s["id"]: np.full(len(raw), np.nan) for s in specs}
    for fold_number, fold in enumerate(phase_spec["splits"]):
        train, valid = raw[fold["train"]], raw[fold["validation"]]
        prep_started = time.perf_counter()
        try:
            first_model = build(specs[0])
            if isinstance(first_model, Pipeline):
                with warnings.catch_warnings(record=True) as notices, threadpool_limits(limits=1), time_limit(120):
                    warnings.simplefilter("always")
                    transformer = first_model[:-1]
                    a = transformer.fit_transform(train[:, :-1], train[:, -1])
                    b = transformer.transform(valid[:, :-1])
                prep_warnings = {str(w.message) for w in notices if not issubclass(w.category, FutureWarning)}
            else:
                a, b = train[:, :-1], valid[:, :-1]
                prep_warnings = set()
            prep_seconds = time.perf_counter() - prep_started
        except Exception as exc:
            for record in records.values():
                record["error"] = f"preprocessing: {type(exc).__name__}: {exc}"
            break
        for spec in specs:
            record = records[spec["id"]]
            if "error" in record:
                continue
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as notices, threadpool_limits(limits=1), time_limit(45):
                    warnings.simplefilter("always")
                    model = build(spec)
                    if isinstance(model, Pipeline):
                        estimator = model[-1]
                        estimator.fit(a, train[:, -1])
                        if hasattr(estimator, "predict_proba"):
                            scores = estimator.predict_proba(b)[:, 1]
                        else:
                            scores = estimator.decision_function(b)
                            record["score_kind"] = "decision"
                    elif model is None:
                        predictor = predictTest if spec["source"] == "current" else archived_models.original_predict
                        scores = predictor(a, train[:, -1], b)
                    else:
                        model.fit(a, train[:, -1])
                        scores = model.predict_proba(b)[:, 1]
                scores = np.asarray(scores, dtype=float)
                if scores.shape != (len(valid),) or not np.isfinite(scores).all():
                    raise ValueError("Invalid prediction shape or nonfinite scores")
                outputs[spec["id"]][fold["validation"]] = scores
                record["folds"].append({"fold": fold_number, **metrics(valid[:, -1], scores)})
                record["warnings"] = sorted(set(record["warnings"]) | prep_warnings |
                                             {str(w.message) for w in notices if not issubclass(w.category, FutureWarning)})
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["seconds"] += time.perf_counter() - started + prep_seconds / len(specs)
    for record in records.values():
        if "error" not in record:
            record.update({key: float(np.mean([f[key] for f in record["folds"]])) for key in ("auc", "tpr", "pauc")})
            np.savez_compressed(destination / (record["id"] + ".npz"), scores=outputs[record["id"]])
        write_json(destination / (record["id"] + ".json"), record)
    return {"completed": len(records), "errors": sum("error" in r for r in records.values())}


def run_phase(phase, jobs=4):
    protocol = json.loads((HERE / "protocol.json").read_text())
    validate_provenance(protocol)
    phase_spec = protocol["phases"][phase]
    destination = HERE / "results" / phase
    destination.mkdir(parents=True, exist_ok=True)
    pending = [s for s in protocol["specs"] if s["id"] in phase_spec["ids"]
               and not (destination / (s["id"] + ".json")).exists()]
    groups = defaultdict(list)
    for spec in pending:
        try:
            model = build(spec)
            key = digest(canonical(model[:-1])) if isinstance(model, Pipeline) else spec["id"]
        except Exception:
            key = spec["id"]
        groups[key].append(spec)
    tasks = [group[start:start + 12] for group in groups.values() for start in range(0, len(group), 12)]
    # Long estimators start early, while fixed preprocessing is shared within tasks.
    np.random.default_rng(340).shuffle(tasks)
    raw = data()
    completed = len(phase_spec["ids"]) - len(pending)
    errors = 0
    started = time.perf_counter()
    print(json.dumps({"phase": phase, "pending": len(pending), "groups": len(tasks)}), flush=True)
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(run_group)(group, phase, phase_spec, raw) for group in tasks)
        for result in results:
            completed += result["completed"]
            errors += result["errors"]
            print(json.dumps({"phase": phase, "completed": completed, "total": len(phase_spec["ids"]),
                              "errors_this_run": errors, "seconds": round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("init", "screen", "nested_0", "nested_1", "nested_2"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    initialize() if args.phase == "init" else run_phase(args.phase, args.jobs)
