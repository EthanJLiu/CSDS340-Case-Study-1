"""Prespecified sklearn families beyond the archived 1,841 configurations.

Every pipeline accepts raw frequencies with -1 marking missingness. All learned
transformations fit inside the caller's training split. Configurations are fixed
before evaluating scores; this module contains no data-dependent selection.
"""

from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import NuSVC, SVC


MODEL_SEED = 340


def frequencies(x, transform="log"):
    """Stateless transformation; learned replacement values come from the imputer."""
    values = np.asarray(x, dtype=float).copy()
    values[values == -1] = np.nan
    if transform == "raw":
        return values
    if transform == "binary":
        return np.where(np.isnan(values), np.nan, (values > 0).astype(float))
    log_values = np.log1p(1000.0 * values)
    if transform == "log":
        return log_values
    if transform == "log_binary":
        return np.column_stack((log_values, np.where(
            np.isnan(values), np.nan, (values > 0).astype(float))))
    raise ValueError(transform)


def preprocessing(transform, scale=True):
    steps = [
        ("features", FunctionTransformer(frequencies, kw_args={"transform": transform})),
        ("impute", SimpleImputer(strategy="mean", keep_empty_features=True)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


class MissingnessAugmentedLogistic(ClassifierMixin, BaseEstimator):
    """Fit on original rows plus one independently masked training copy.

    This creates no extra independent observations. It tests a training-time
    regularizer; held-out features and labels never influence the augmentation.
    """

    def __init__(self, C=0.1, fraction=0.1, random_state=MODEL_SEED):
        self.C = C
        self.fraction = fraction
        self.random_state = random_state

    def fit(self, X, y):
        original = np.asarray(X, dtype=float)
        altered = original.copy()
        rng = np.random.default_rng(self.random_state)
        mask = (rng.random(altered.shape) < self.fraction) & np.isfinite(altered) & (altered >= 0)
        altered[mask] = -1
        self.model_ = Pipeline(preprocessing("log_binary") + [
            ("model", LogisticRegression(C=self.C, max_iter=2000,
                                         random_state=self.random_state)),
        ])
        self.model_.fit(np.concatenate((original, altered)), np.tile(np.asarray(y), 2))
        self.classes_ = self.model_.classes_
        self.n_features_in_ = original.shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def get_configs():
    """Return the entire fixed Cartesian library, not a performance shortlist."""
    configurations = []
    for transform, reg in product(("log", "binary", "log_binary"), (.1, .3, .7, .95)):
        configurations.append(dict(family="regularized_qda", transform=transform, reg_param=reg))
    for family, transform, alpha in product(
            ("multinomial_nb", "complement_nb"), ("raw", "log", "binary"), (.01, .1, 1., 10.)):
        configurations.append(dict(family=family, transform=transform, alpha=alpha))
    for transform, c, ratio in product(("log", "log_binary"), (.01, .1, 1., 10.), (.1, .5, .9)):
        configurations.append(dict(family="elastic_logistic", transform=transform, C=c, l1_ratio=ratio))
    for transform, degree, c, gamma in product(
            ("log", "log_binary"), (2, 3), (.1, 1., 10.), (.003, .01)):
        configurations.append(dict(family="polynomial_svm", transform=transform,
                                   degree=degree, C=c, gamma=gamma, coef0=1.))
    for transform, nu, gamma in product(("log", "log_binary"), (.1, .3, .5), (.003, .01)):
        configurations.append(dict(family="nu_svm", transform=transform, nu=nu, gamma=gamma))
    for family, transform, gamma, c in product(
            ("random_fourier", "nystroem_rbf"), ("log", "log_binary"), (.003, .01), (.1, 1.)):
        configurations.append(dict(family=family, transform=transform, gamma=gamma,
                                   C=c, n_components=256))
    for transform, layers, alpha in product(("log", "log_binary"), ([32], [64, 32]), (.1, 1., 10.)):
        configurations.append(dict(family="adam_mlp", transform=transform,
                                   hidden_layer_sizes=layers, alpha=alpha))
    for fraction, c in product((.05, .15), (.01, .1, 1.)):
        configurations.append(dict(family="missingness_augmented_logistic", fraction=fraction, C=c))
    assert len(configurations) == 130
    return configurations


def build(config):
    c = dict(config)
    family = c.pop("family")
    transform = c.pop("transform", "log")
    if family == "missingness_augmented_logistic":
        return MissingnessAugmentedLogistic(**c)
    steps = preprocessing(transform, scale=family not in ("multinomial_nb", "complement_nb"))
    if family == "regularized_qda":
        estimator = QuadraticDiscriminantAnalysis(**c)
    elif family == "multinomial_nb":
        estimator = MultinomialNB(**c)
    elif family == "complement_nb":
        estimator = ComplementNB(**c)
    elif family == "elastic_logistic":
        estimator = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=1500,
                                       tol=1e-4, random_state=MODEL_SEED, **c)
    elif family == "polynomial_svm":
        estimator = SVC(kernel="poly", probability=True, cache_size=256,
                        random_state=MODEL_SEED, **c)
    elif family == "nu_svm":
        estimator = NuSVC(kernel="rbf", probability=True, cache_size=256,
                          random_state=MODEL_SEED, **c)
    elif family in ("random_fourier", "nystroem_rbf"):
        c_value = c.pop("C")
        mapper = (RBFSampler(random_state=MODEL_SEED, **c) if family == "random_fourier"
                  else Nystroem(kernel="rbf", random_state=MODEL_SEED, n_jobs=1, **c))
        steps.append(("kernel_map", mapper))
        estimator = LogisticRegression(C=c_value, max_iter=2000, random_state=MODEL_SEED)
    elif family == "adam_mlp":
        estimator = MLPClassifier(solver="adam", early_stopping=True,
                                  validation_fraction=.15, n_iter_no_change=30,
                                  max_iter=600, learning_rate_init=.001,
                                  random_state=MODEL_SEED, **c)
    else:
        raise ValueError(f"Unknown additional family: {family}")
    return Pipeline(steps + [("model", estimator)])
