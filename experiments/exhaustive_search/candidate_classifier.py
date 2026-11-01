"""Frozen screen-selected candidate; not the active submission.

Weighted log-odds of three probabilities:
  25% logistic regression on zero-aware positive-frequency quantile bins;
  25% histogram gradient boosting with native missing-value routing;
  50% regularized logistic regression on selected quadratic word features.

This standalone implementation contains the fixed recipe from the broad search.
Its promotion requires the separate prespecified stability and nested gates.
"""


def predictTest(trainFeatures,trainLabels,testFeatures):
    import numpy as np
    from scipy.special import logit
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from threadpoolctl import threadpool_limits

    train = np.array(trainFeatures, dtype=float, copy=True)
    test = np.array(testFeatures, dtype=float, copy=True)
    labels = np.asarray(trainLabels).reshape(-1)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("Training and test features must be two-dimensional.")
    if train.shape[1] != test.shape[1] or train.shape[1] == 0:
        raise ValueError("Training and test features must have matching columns.")
    if len(train) == 0 or len(labels) != len(train):
        raise ValueError("Each training email must have one label.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must be 0 (ham) or 1 (spam).")
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError("Infinite feature values are not supported.")
    train[train == -1] = np.nan
    test[test == -1] = np.nan
    if (train < 0).any() or (test < 0).any():
        raise ValueError("Features must be nonnegative, -1, or NaN.")
    if len(test) == 0:
        return np.empty(0, dtype=float)
    if len(np.unique(labels)) == 1:
        return np.full(len(test), float(labels[0]))

    # Keep observed absence separate from twelve positive-frequency quantile
    # bins. Missing measurements receive TRAINING category proportions.
    binned_train = []
    binned_test = []
    for column_index in range(train.shape[1]):
        column = train[:, column_index]
        observed = column[np.isfinite(column)]
        positive = observed[observed > 0]
        edges = (np.unique(np.quantile(positive, np.arange(1, 12) / 12))
                 if len(positive) else np.array([]))
        category_count = len(edges) + 2
        codes = np.where(observed > 0,
                         1 + np.searchsorted(edges, observed, side="right"), 0)
        counts = np.bincount(codes, minlength=category_count).astype(float)
        prior = (counts / counts.sum() if counts.sum()
                 else np.r_[1.0, np.zeros(category_count - 1)])
        for data, output in ((train, binned_train), (test, binned_test)):
            values = data[:, column_index]
            codes = np.where(values > 0,
                             1 + np.searchsorted(edges, values, side="right"), 0)
            encoded = np.eye(category_count)[codes]
            encoded[np.isnan(values)] = prior
            output.append(np.column_stack((1.0 - encoded[:, 0], encoded)))

    # All fits, including supervised feature selection and imputation, receive
    # only training rows. Limit native libraries locally for repeatability.
    with threadpool_limits(limits=1):
        binned_model = LogisticRegression(
            C=0.3, max_iter=3000, class_weight={0: 1.0, 1: 1.0},
        )
        binned_model.fit(np.hstack(binned_train), labels)
        binned_probability = binned_model.predict_proba(np.hstack(binned_test))[:, 1]

        histogram_model = HistGradientBoostingClassifier(
            early_stopping=False, random_state=340,
            l2_regularization=0.1, learning_rate=0.05, max_iter=400,
            max_leaf_nodes=7, min_samples_leaf=10,
        )
        # A wholly unobserved TRAINING column cannot supply a learned split.
        # Give only those columns a constant representation to avoid an empty
        # threshold-array error in some sklearn histogram-binning versions.
        empty_columns = np.isnan(train).all(axis=0)
        histogram_train, histogram_test = train, test
        if empty_columns.any():
            histogram_train, histogram_test = train.copy(), test.copy()
            histogram_train[:, empty_columns] = 0.0
            histogram_test[:, empty_columns] = 0.0
        histogram_model.fit(histogram_train, labels)
        histogram_probability = histogram_model.predict_proba(histogram_test)[:, 1]

        transformed_train = np.hstack((
            np.log1p(1000.0 * train),
            np.where(np.isnan(train), np.nan, (train > 0).astype(float)),
        ))
        transformed_test = np.hstack((
            np.log1p(1000.0 * test),
            np.where(np.isnan(test), np.nan, (test > 0).astype(float)),
        ))
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        interaction_train = imputer.fit_transform(transformed_train)
        interaction_test = imputer.transform(transformed_test)
        selector = SelectKBest(f_classif, k=30)
        interaction_train = selector.fit_transform(interaction_train, labels)
        interaction_test = selector.transform(interaction_test)
        quadratic = PolynomialFeatures(
            degree=2, interaction_only=False, include_bias=False,
        )
        interaction_train = quadratic.fit_transform(interaction_train)
        interaction_test = quadratic.transform(interaction_test)
        scaler = StandardScaler()
        interaction_train = scaler.fit_transform(interaction_train)
        interaction_test = scaler.transform(interaction_test)
        interaction_model = LogisticRegression(C=0.001, max_iter=3000)
        interaction_model.fit(interaction_train, labels)
        interaction_probability = interaction_model.predict_proba(interaction_test)[:, 1]

    probabilities = np.array((binned_probability, histogram_probability,
                              interaction_probability))
    return np.array([0.25, 0.25, 0.5]) @ logit(np.clip(probabilities, 1e-6, 1 - 1e-6))


if __name__ == "__main__":
    print("Frozen research candidate; import predictTest to evaluate it.")
