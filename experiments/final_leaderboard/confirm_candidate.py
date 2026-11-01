"""One confirmation gate for the candidate frozen after development comparison.

These rows come from released training data, NOT the hidden final test. They are
used once for the retention decision, so this is not an independent final-score
estimate. Rerunning reproduces the gate; do not use its errors to tune candidates.
"""

import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import sklearn

from stability_experiments import binned_probability, components, metrics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")
    # Written before accessing confirmation predictions or metrics.
    recipe = {
        "candidate": "bin_average_2_3_4",
        "description": "Average binned logistic probabilities using 2, 3, and 4 "
                       "positive-frequency bins; keep original C and 50/25/25 weights.",
        "gate": "Retain baseline if candidate AUC OR TPR at FPR=.01 declines.",
        "bootstrap": "2000 paired stratified resamples of fixed confirmation "
                     "predictions; describes row uncertainty, not model-selection uncertainty.",
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
            HERE / "baseline_classifier.py", HERE / "split_plan.json",
            HERE / "stability_experiments.py", HERE / "regularization_experiments.py",
            ROOT / "spamTrain1.csv", ROOT / "spamTrain2.csv",
        ]},
    }
    recipe_path = HERE / "frozen_candidate.json"
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            raise ValueError("Confirmation recipe or inputs changed after freezing.")
    else:
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    plan = json.loads((HERE / "split_plan.json").read_text())
    data = np.concatenate([np.loadtxt(ROOT / filename, delimiter=",")
                           for filename in ("spamTrain1.csv", "spamTrain2.csv")])
    train_ids = np.array(plan["development"])
    validation_ids = np.array(plan["confirmation"])
    assert len(train_ids) == 2500 and len(validation_ids) == 500
    assert not set(train_ids) & set(validation_ids)
    train, validation = data[train_ids], data[validation_ids]
    outputs = components(train[:, :-1], train[:, -1], validation[:, :-1])
    baseline = outputs @ [.5, .25, .25]
    bins = [outputs[:, 0]] + [binned_probability(
        train[:, :-1], train[:, -1], validation[:, :-1], bins=count)
        for count in (3, 4)]
    candidate = .5 * np.mean(bins, axis=0) + .25 * outputs[:, 1] + .25 * outputs[:, 2]
    labels = validation[:, -1]
    scores = {"baseline": metrics(labels, baseline), "candidate": metrics(labels, candidate)}
    delta = {key: scores["candidate"][key] - scores["baseline"][key] for key in ("auc", "tpr")}
    rng = np.random.default_rng(20261101)
    ham, spam = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    differences = []
    for _ in range(2000):
        ids = np.r_[rng.choice(ham, len(ham), replace=True),
                    rng.choice(spam, len(spam), replace=True)]
        b, c = metrics(labels[ids], baseline[ids]), metrics(labels[ids], candidate[ids])
        differences.append([c[key] - b[key] for key in ("auc", "tpr")])
    intervals = np.quantile(differences, [.025, .975], axis=0)
    report = {
        "training_rows": len(train), "confirmation_rows": len(validation),
        "ham_rows": len(ham), "spam_rows": len(spam),
        "metrics": scores, "candidate_minus_baseline": delta,
        "paired_bootstrap_95_percent_intervals": {
            key: intervals[:, i].tolist() for i, key in enumerate(("auc", "tpr"))},
        "decision": "retain_baseline" if any(value < 0 for value in delta.values()) else "candidate_passes_gate",
        "limitation": "Internal comparison on released data; cannot establish final leaderboard rank. "
                      "The gate uses these labels for a retention decision. Bootstrap intervals "
                      "do not account for model selection, refitting, or dataset shift.",
    }
    np.savez_compressed(HERE / "confirmation_predictions.npz", validation_ids=validation_ids,
                        labels=labels, baseline=baseline, candidate=candidate)
    (HERE / "confirmation_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
