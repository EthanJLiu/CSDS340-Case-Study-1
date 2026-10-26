"""CSDS 340 Case Study 1: personalized email spam scoring.

Requires NumPy and scikit-learn.  Each row is one email and each of the
30 columns is a word-frequency feature.  A value of -1 denotes missing data.

The grader imports predictTest(trainFeatures,trainLabels,testFeatures) and
expects a continuous spam score for every test email.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def predictTest(trainFeatures,trainLabels,testFeatures):
    """Fit on the supplied training set and return test spam scores.

    The combination puts 60% weight on a regularized histogram gradient
    boosting classifier and 40% on a random forest.  The former handles
    NaN values natively; the latter uses training-only median imputation
    and explicit missingness indicators.  No fitting uses test examples.

    Returns
    -------
    np.ndarray, shape (n_test,)
        Floating-point spam scores in [0, 1], with larger meaning more
        likely to be spam.  Scores, not hard labels, support ROC evaluation.
    """
    trainFeatures = np.asarray(trainFeatures, dtype=float)
    trainLabels = np.asarray(trainLabels).ravel()
    testFeatures = np.asarray(testFeatures, dtype=float)

    # The sentinel -1 is a missing value, never a legitimate word frequency.
    # This conversion uses no statistics from the test data.
    train_nan = np.where(trainFeatures == -1, np.nan, trainFeatures)
    test_nan = np.where(testFeatures == -1, np.nan, testFeatures)

    boosted = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=340,
    )
    boosted.fit(train_nan, trainLabels)
    boosted_scores = boosted.predict_proba(test_nan)[:, 1]

    forest = make_pipeline(
        SimpleImputer(missing_values=-1, strategy="median", add_indicator=True),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=1,
            random_state=340,
        ),
    )
    forest.fit(trainFeatures, trainLabels)
    forest_scores = forest.predict_proba(testFeatures)[:, 1]

    return 0.60 * boosted_scores + 0.40 * forest_scores


if __name__ == "__main__":
    # Optional local self-check: this block is NOT run when the grader imports
    # predictTest.  Run from a directory containing spamTrain1.csv.
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold

    data = np.loadtxt("spamTrain1.csv", delimiter=",")
    X, y = data[:, :-1], data[:, -1]
    scores = np.empty(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=340)
    for train_idx, test_idx in cv.split(X, y):
        scores[test_idx] = predictTest(X[train_idx], y[train_idx], X[test_idx])
    fpr, tpr, _ = roc_curve(y, scores)
    print(f"Out-of-fold ROC-AUC: {roc_auc_score(y, scores):.4f}")
    print(f"Out-of-fold TPR at 1% FPR: {np.interp(0.01, fpr, tpr):.4f}")
