"""Limited, reproducible development comparison; never scores confirmation rows.

Run from the repository root with:
    python3 experiments/final_leaderboard/regularization_experiments.py

The baseline source is frozen before experiments. Source substitutions expose its
three scores and C parameters without changing its preprocessing or production.
"""

import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


DIRECTORY = Path(__file__).resolve().parent


def component_predictor():
    source = (DIRECTORY / "baseline_classifier.py").read_text()
    substitutions = {
        "def predictTest(trainFeatures,trainLabels,testFeatures):":
            "def predictTest(trainFeatures,trainLabels,testFeatures, "
            "bin_c=1.0, svm_c=10.0, interaction_c=0.001):",
        'C=1.0, solver="lbfgs"': 'C=bin_c, solver="lbfgs"',
        'C=10.0, kernel="rbf"': 'C=svm_c, kernel="rbf"',
        'C=0.001, solver="lbfgs"': 'C=interaction_c, solver="lbfgs"',
        "return (0.50 * binned_probabilities\n"
        "            + 0.25 * svm_probabilities\n"
        "            + 0.25 * interaction_probabilities)":
            "return np.column_stack((binned_probabilities, svm_probabilities, "
            "interaction_probabilities))",
    }
    for old, new in substitutions.items():
        if source.count(old) != 1:
            raise ValueError(f"Frozen source no longer matches: {old!r}")
        source = source.replace(old, new)
    namespace = {"__name__": "frozen_components"}
    exec(compile(source, str(DIRECTORY / "baseline_classifier.py"), "exec"), namespace)
    return namespace["predictTest"]


def metrics(labels, predictions):
    fpr, tpr, _ = roc_curve(labels, predictions)
    # Equivalent to the instructor's interpolation, including repeated FPRs.
    return {"auc": float(roc_auc_score(labels, predictions)),
            "tpr": float(np.interp(0.01, fpr, tpr))}


def main():
    warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")
    data = np.concatenate([np.loadtxt(DIRECTORY.parents[1] / name, delimiter=",")
                           for name in ("spamTrain1.csv", "spamTrain2.csv")])
    plan = json.loads((DIRECTORY / "split_plan.json").read_text())
    predict = component_predictor()
    rows = []
    saved = {}
    start = time.perf_counter()
    for index, fold in enumerate(plan["folds"]):
        train = data[fold["train"]]
        validation = data[fold["validation"]]
        assert not set(fold["train"] + fold["validation"]) & set(plan["confirmation"])
        factor = 1500 / len(train)
        parameters = {
            "baseline": {},
            "size_scaled_C": {"bin_c": factor, "svm_c": 10 * factor,
                              "interaction_c": 0.001 * factor},
            "interaction_half_C": {"interaction_c": 0.0005},
            "interaction_double_C": {"interaction_c": 0.002},
        }
        saved[f"indices_{index}"] = np.array(fold["validation"])
        for name, kwargs in parameters.items():
            components = predict(train[:, :-1], train[:, -1], validation[:, :-1], **kwargs)
            scores = components @ np.array([0.5, 0.25, 0.25])
            saved[f"{name}_{index}"] = scores
            if name == "baseline":
                saved[f"components_{index}"] = components
            row = dict(index=index, seed=fold["seed"], fold=fold["fold"],
                       model=name, **metrics(validation[:, -1], scores))
            rows.append(row)
        print(json.dumps({"completed_fold": index, "seconds": time.perf_counter() - start}), flush=True)
        np.savez_compressed(DIRECTORY / "regularization_predictions.npz", **saved)
        (DIRECTORY / "regularization_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    for name in parameters:
        subset = [row for row in rows if row["model"] == name]
        print(json.dumps({"model": name, **{
            key: float(np.mean([row[key] for row in subset])) for key in ("auc", "tpr")}}))


if __name__ == "__main__":
    main()
