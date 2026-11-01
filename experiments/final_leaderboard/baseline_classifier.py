"""CSDS 340 spam classifier.

The 3 spam scores:
  1. Zero-aware frequency bins + logistic regression (weight 0.50).
  2. Nearest-neighbor imputation + radial-basis SVM (weight 0.25).
  3. Log frequencies, presence, and pairwise interactions + logistic regression
     (weight 0.25).
"""

def predictTest(trainFeatures,trainLabels,testFeatures):
    import numpy as np
    from sklearn.impute import KNNImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVC
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

    # Representation 1: absence plus two positive-frequency categories.
    # Unobserved values get TRAINING category proportions, never test statistics.
    binned_train = []
    binned_test = []
    for column_index in range(train.shape[1]):
        column = train[:, column_index]
        observed = column[np.isfinite(column)]
        positive = observed[observed > 0]
        if len(positive):
            edges = np.unique(np.expm1(np.linspace(
                np.log1p(1000.0 * positive.min()),
                np.log1p(1000.0 * positive.max()), 3,
            )[1:-1]) / 1000.0)
        else:
            edges = np.array([])
        category_count = len(edges) + 2
        observed_codes = np.where(
            observed > 0,
            1 + np.searchsorted(edges, observed, side="right"),
            0,
        )
        counts = np.bincount(observed_codes, minlength=category_count).astype(float)
        if counts.sum():
            prior = counts / counts.sum()
        else:
            prior = np.zeros(category_count)
            prior[0] = 1.0
        for data, output in ((train, binned_train), (test, binned_test)):
            values = data[:, column_index]
            codes = np.where(
                values > 0,
                1 + np.searchsorted(edges, values, side="right"),
                0,
            )
            encoded = np.eye(category_count)[codes]
            encoded[np.isnan(values)] = prior
            output.append(np.column_stack((1.0 - encoded[:, 0], encoded)))

    # Representation 2: compress skewed frequencies, while preserving absence.
    # Missing means unknown; observed zero remains a valid absence measurement.
    log_train = np.log1p(1000.0 * train)
    log_test = np.log1p(1000.0 * test)
    transformed_train = np.hstack((
        log_train,
        np.where(np.isnan(train), np.nan, (train > 0).astype(float)),
    ))
    transformed_test = np.hstack((
        log_test,
        np.where(np.isnan(test), np.nan, (test > 0).astype(float)),
    ))
    observed_counts = np.isfinite(transformed_train).sum(axis=0)
    means = np.divide(
        np.nansum(transformed_train, axis=0), observed_counts,
        out=np.zeros(transformed_train.shape[1]), where=observed_counts > 0,
    )
    transformed_train = np.where(np.isnan(transformed_train), means, transformed_train)
    transformed_test = np.where(np.isnan(transformed_test), means, transformed_test)
    # Restrict native thread pools locally, without changing the caller's settings.
    with threadpool_limits(limits=2):
        binned_model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=3000, random_state=340,
        )
        binned_model.fit(np.hstack(binned_train), labels)
        binned_probabilities = binned_model.predict_proba(np.hstack(binned_test))[:, 1]

        # Imputation uses distances between observed log frequencies. Only
        # training emails can supply replacements for a test email.
        imputer = KNNImputer(n_neighbors=31, weights="uniform", keep_empty_features=True)
        imputed_train = imputer.fit_transform(log_train)
        imputed_test = imputer.transform(log_test)
        svm_scaler = StandardScaler()
        svm_train = svm_scaler.fit_transform(imputed_train)
        svm_test = svm_scaler.transform(imputed_test)
        svm_model = SVC(
            C=10.0, kernel="rbf", gamma=0.003, probability=True,
            random_state=340, cache_size=256,
        )
        svm_model.fit(svm_train, labels)
        svm_probabilities = svm_model.predict_proba(svm_test)[:, 1]

        # Strong regularization controls the 1,830 original/pairwise features
        # formed from 30 log frequencies and 30 presence measurements.
        interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interaction_train = interactions.fit_transform(transformed_train)
        interaction_test = interactions.transform(transformed_test)
        interaction_scaler = StandardScaler()
        interaction_train = interaction_scaler.fit_transform(interaction_train)
        interaction_test = interaction_scaler.transform(interaction_test)
        interaction_model = LogisticRegression(
            C=0.001, solver="lbfgs", max_iter=3000, random_state=340,
        )
        interaction_model.fit(interaction_train, labels)
        interaction_probabilities = interaction_model.predict_proba(interaction_test)[:, 1]

    return (0.50 * binned_probabilities
            + 0.25 * svm_probabilities
            + 0.25 * interaction_probabilities)


if __name__ == "__main__":
    print("Import predictTest(trainFeatures, trainLabels, testFeatures) to use this classifier.")
