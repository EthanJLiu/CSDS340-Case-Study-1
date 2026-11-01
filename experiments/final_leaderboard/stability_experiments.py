"""Prespecified stability comparisons on development folds only.

The separate confirmation partition is never evaluated by this script.
"""

import inspect
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from baseline_classifier import predictTest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def make_components():
    source = inspect.getsource(predictTest)
    old = """return (0.50 * binned_probabilities
            + 0.25 * svm_probabilities
            + 0.25 * interaction_probabilities)"""
    assert source.count(old) == 1
    source = source.replace(
        old,
        "return np.column_stack((binned_probabilities, svm_probabilities, interaction_probabilities))",
    )
    namespace = {}
    exec(compile(source, "baseline_components", "exec"), namespace)
    return namespace["predictTest"]


components = make_components()


def binned_probability(x, y, z, bins=2, median=False):
    train = np.asarray(x, dtype=float).copy()
    test = np.asarray(z, dtype=float).copy()
    train[train == -1] = np.nan
    test[test == -1] = np.nan
    transformed_train, transformed_test = [], []
    for j in range(train.shape[1]):
        values = train[:, j]
        observed = values[np.isfinite(values)]
        positive = observed[observed > 0]
        if not len(positive):
            edges = np.array([])
        elif median:
            edges = np.array([np.median(positive)])
        else:
            edges = np.unique(np.expm1(np.linspace(
                np.log1p(1000.0 * positive.min()),
                np.log1p(1000.0 * positive.max()), bins + 1,
            )[1:-1]) / 1000.0)
        category_count = len(edges) + 2
        codes = np.where(observed > 0,
                         1 + np.searchsorted(edges, observed, side="right"), 0)
        counts = np.bincount(codes, minlength=category_count).astype(float)
        if counts.sum():
            prior = counts / counts.sum()
        else:
            prior = np.zeros(category_count)
            prior[0] = 1.0
        for data, output in ((train, transformed_train), (test, transformed_test)):
            v = data[:, j]
            codes = np.where(v > 0, 1 + np.searchsorted(edges, v, side="right"), 0)
            encoded = np.eye(category_count)[codes]
            encoded[np.isnan(v)] = prior
            output.append(np.column_stack((1.0 - encoded[:, 0], encoded)))
    with threadpool_limits(limits=2):
        model = LogisticRegression(C=1., solver="lbfgs", max_iter=3000, random_state=340)
        model.fit(np.hstack(transformed_train), y)
        return model.predict_proba(np.hstack(transformed_test))[:, 1]


def metrics(y, probability):
    fpr, tpr, _ = roc_curve(y, probability)
    i = np.where(fpr <= 0.01)[0][-1]
    value = tpr[i] + (tpr[i + 1] - tpr[i]) / (fpr[i + 1] - fpr[i]) * (0.01 - fpr[i])
    return {"auc": float(roc_auc_score(y, probability)), "tpr": float(value)}


def run():
    plan = json.loads((HERE / "split_plan.json").read_text())
    raw = np.r_[np.loadtxt(ROOT / "spamTrain1.csv", delimiter=","),
                np.loadtxt(ROOT / "spamTrain2.csv", delimiter=",")]
    allowed = set(plan["development"])
    result_path = HERE / "stability_results.json"
    result = (json.loads(result_path.read_text()) if result_path.exists() else
              {"bagging_fraction": .8, "folds": []})
    result["protocol"] = "Ten fixed seed-340/341 development folds; confirmation untouched"
    members = result.get("bagging_members")
    for fold_number, fold in enumerate(plan["folds"]):
        if fold_number < len(result["folds"]):
            previous = result["folds"][fold_number]
            assert previous["seed"] == fold["seed"] and previous["fold"] == fold["fold"]
            assert (HERE / f"stability_fold_{fold_number}.npz").exists()
            continue
        train_ids = np.array(fold["train"])
        valid_ids = np.array(fold["validation"])
        assert set(train_ids) <= allowed and set(valid_ids) <= allowed
        train, valid = raw[train_ids], raw[valid_ids]
        started = time.monotonic()
        p = components(train[:, :-1], train[:, -1], valid[:, :-1])
        baseline_seconds = time.monotonic() - started
        # Runtime-only decision made before the first metric is computed.
        if members is None:
            members = 3 if baseline_seconds > 20 else 5
            result["bagging_members"] = members
            print(f"First baseline time: {baseline_seconds:.2f}s. Bagging members: {members}", flush=True)
        predictions = {"baseline": p @ [.5, .25, .25]}
        bin3 = binned_probability(train[:, :-1], train[:, -1], valid[:, :-1], bins=3)
        bin4 = binned_probability(train[:, :-1], train[:, -1], valid[:, :-1], bins=4)
        binmedian = binned_probability(train[:, :-1], train[:, -1], valid[:, :-1], median=True)
        predictions["bin_average_2_3_4"] = .5 * np.mean([p[:, 0], bin3, bin4], axis=0) + .25 * p[:, 1] + .25 * p[:, 2]
        predictions["positive_median_boundary"] = .5 * binmedian + .25 * p[:, 1] + .25 * p[:, 2]
        bagged = []
        for member in range(members):
            subset, _ = train_test_split(np.arange(len(train)), train_size=.8,
                                        stratify=train[:, -1], random_state=34000 + 100 * fold_number + member)
            bagged.append(predictTest(train[subset, :-1], train[subset, -1], valid[:, :-1]))
        predictions["subsample_bagging"] = np.mean(bagged, axis=0)
        path = HERE / f"stability_fold_{fold_number}.npz"
        np.savez_compressed(path, validation_ids=valid_ids, labels=valid[:, -1],
                            baseline_components=p, bin3=bin3, bin4=bin4, binmedian=binmedian,
                            bagging_members=np.array(bagged), **predictions)
        scores = {name: metrics(valid[:, -1], values) for name, values in predictions.items()}
        row = {"seed": fold["seed"], "fold": fold["fold"], "baseline_seconds": baseline_seconds,
               "total_seconds": time.monotonic() - started, "metrics": scores}
        result["folds"].append(row)
        (HERE / "stability_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    result["means"] = {name: {metric: float(np.mean([row["metrics"][name][metric] for row in result["folds"]]))
                              for metric in ["auc", "tpr"]} for name in result["folds"][0]["metrics"]}
    (HERE / "stability_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("MEANS " + json.dumps(result["means"]), flush=True)


if __name__ == "__main__":
    run()
