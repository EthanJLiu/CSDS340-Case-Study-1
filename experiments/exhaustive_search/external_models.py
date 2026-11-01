"""Prespecified optional boosting models for the broad spam comparison.

All preprocessing learns from the training fold only. The module does not read
data, choose validation folds, evaluate metrics, or install dependencies. Optional
packages are imported only when a model is built so the runner can record missing
dependencies as failures rather than silently dropping model families.
"""

from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


SEED = 340


class MissingToNaN(TransformerMixin, BaseEstimator):
    """Convert the assignment's missing sentinel while retaining observed zeros."""

    def fit(self, X, y=None):
        data = np.asarray(X)
        if data.ndim != 2:
            raise ValueError("Features must be a two-dimensional array.")
        self.n_features_in_ = data.shape[1]
        return self

    def transform(self, X):
        data = np.array(X, dtype=float, copy=True)
        if data.ndim != 2 or data.shape[1] != self.n_features_in_:
            raise ValueError("Training and prediction feature columns must match.")
        data[data == -1] = np.nan
        if np.isinf(data).any() or (data < 0).any():
            raise ValueError("Features must be nonnegative, -1, or NaN.")
        return data


def get_configs():
    """Return all 144 unique, finite configurations before seeing their scores."""
    configs = []
    # Native missing routing and mean-plus-indicator preprocessing are genuinely
    # different representations. A raw/log-only distinction is intentionally
    # omitted because monotone transforms largely duplicate tree partitions.
    for depth, iterations, l2, randomness, representation in product(
        (2, 4, 6), (200, 500), (3.0, 15.0), (0.0, 1.0),
        ("native", "mean_indicator"),
    ):
        configs.append({
            "family": "catboost", "depth": depth, "iterations": iterations,
            "l2_leaf_reg": l2, "random_strength": randomness,
            "representation": representation,
        })
    for depth, trees, l2, child_weight, column_fraction in product(
        (2, 4, 6), (200, 500), (1.0, 10.0), (1.0, 10.0), (0.7, 1.0),
    ):
        configs.append({
            "family": "xgboost", "max_depth": depth, "n_estimators": trees,
            "reg_lambda": l2, "min_child_weight": child_weight,
            "colsample_bytree": column_fraction, "representation": "native",
        })
    for leaves, trees, l2, child_samples, representation in product(
        (7, 15, 31), (200, 500), (1.0, 10.0), (10, 30),
        ("native", "mean_indicator"),
    ):
        configs.append({
            "family": "lightgbm", "num_leaves": leaves, "n_estimators": trees,
            "reg_lambda": l2, "min_child_samples": child_samples,
            "representation": representation,
        })
    return configs


def build(config):
    """Construct an unfitted model; no validation labels or features are used."""
    parameters = dict(config)
    family = parameters.pop("family")
    representation = parameters.pop("representation", "native")
    steps = [("missing", MissingToNaN())]
    if representation == "mean_indicator":
        steps.append(("impute", SimpleImputer(
            strategy="mean", add_indicator=True, keep_empty_features=True,
        )))
    elif representation != "native":
        raise ValueError(f"Unknown representation: {representation!r}")

    if family == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            loss_function="Logloss", learning_rate=0.05, random_seed=SEED,
            thread_count=1, verbose=False, allow_writing_files=False,
            task_type="CPU", nan_mode="Min", **parameters,
        )
    elif family == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            learning_rate=0.05, tree_method="hist", n_jobs=1,
            random_state=SEED, verbosity=0, subsample=1.0,
            missing=np.nan, **parameters,
        )
    elif family == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary", learning_rate=0.05, random_state=SEED,
            n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True,
            colsample_bytree=0.8, subsample=1.0, **parameters,
        )
    else:
        raise ValueError(f"Unknown external model family: {family!r}")
    return Pipeline(steps + [("model", model)])
