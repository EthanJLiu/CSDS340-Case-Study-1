"""Frozen model definitions extracted from archive/exploration_full.ipynb.
No notebook experiments execute when this module is imported.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')

from pathlib import Path
import hashlib
import itertools
import json
import time
import warnings
import numpy as np
import pandas as pd
import sklearn
import scipy.special as special
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesRegressor)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.linear_model import LogisticRegression, Perceptron, SGDRegressor, BayesianRidge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import chi2_kernel, laplacian_kernel
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     train_test_split, ParameterGrid)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, StandardScaler,
                                    SplineTransformer, PolynomialFeatures)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed, parallel_backend


MODEL_SEED = 340


def original_predict(trainFeatures,trainLabels,testFeatures):
    """Return a one-dimensional spam probability array in test-row order."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
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

    # Representation 1: absence plus up to five positive-frequency categories.
    # Unobserved values get TRAINING category proportions, never test statistics.
    binned_train = []
    binned_test = []
    for column_index in range(train.shape[1]):
        column = train[:, column_index]
        observed = column[np.isfinite(column)]
        positive = observed[observed > 0]
        if len(positive):
            edges = np.unique(np.quantile(positive, [0.2, 0.4, 0.6, 0.8]))
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
            output.append(encoded)

    # Representation 2: compress skewed frequencies, while preserving absence.
    transformed_train = np.hstack((
        np.log1p(1000.0 * train),
        np.where(np.isnan(train), np.nan, (train > 0).astype(float)),
    ))
    transformed_test = np.hstack((
        np.log1p(1000.0 * test),
        np.where(np.isnan(test), np.nan, (test > 0).astype(float)),
    ))
    observed_counts = np.isfinite(transformed_train).sum(axis=0)
    means = np.divide(
        np.nansum(transformed_train, axis=0), observed_counts,
        out=np.zeros(transformed_train.shape[1]), where=observed_counts > 0,
    )
    transformed_train = np.where(np.isnan(transformed_train), means, transformed_train)
    transformed_test = np.where(np.isnan(transformed_test), means, transformed_test)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(transformed_train)
    scaled_test = scaler.transform(transformed_test)

    # Restrict native thread pools locally, without changing the caller's settings.
    with threadpool_limits(limits=2):
        binned_model = LogisticRegression(
            C=0.3, solver="lbfgs", max_iter=3000, random_state=340,
        )
        binned_model.fit(np.hstack(binned_train), labels)
        binned_probabilities = binned_model.predict_proba(np.hstack(binned_test))[:, 1]

        log_model = LogisticRegression(
            C=0.01, solver="lbfgs", max_iter=3000, random_state=340,
        )
        log_model.fit(scaled_train, labels)
        log_probabilities = log_model.predict_proba(scaled_test)[:, 1]

        # Some sklearn versions cannot bin an entirely unobserved column.
        # Such a column contains no training information; make it constant.
        empty_columns = np.isnan(train).all(axis=0)
        train[:, empty_columns] = 0.0
        test[:, empty_columns] = 0.0
        boosted_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=7,
            min_samples_leaf=15,
            l2_regularization=1.0,
            categorical_features=None,
            early_stopping=False,
            random_state=340,
        )
        boosted_model.fit(train, labels)
        boosted_probabilities = boosted_model.predict_proba(test)[:, 1]

    return (binned_probabilities + log_probabilities + boosted_probabilities) / 3.0

def metrics(y,s):
    f,t,_=roc_curve(y,s);i=np.flatnonzero(f<=.01)[-1]
    tpr=t[i]+(t[i+1]-t[i])*(.01-f[i])/(f[i+1]-f[i])
    return {'auc':float(roc_auc_score(y,s)),'tpr':float(tpr),
            'pauc':float(roc_auc_score(y,s,max_fpr=.01))}

def identity(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]

class FrequencyBinner(TransformerMixin, BaseEstimator):
    """Keep absence separate from positive-frequency bins; learn bins on train only.

    Missing entries become training-set category proportions, not zero/absence.
    """
    def __init__(self, bins=3, presence=False):
        self.bins = bins
        self.presence = presence

    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float)
        self.edges_ = []
        self.priors_ = []
        for column in x.T:
            observed = column[np.isfinite(column) & (column >= 0)]
            positive = observed[observed > 0]
            edges = np.unique(np.quantile(positive, np.arange(1, self.bins) / self.bins)) if len(positive) else np.array([])
            codes = np.where(observed > 0, 1 + np.searchsorted(edges, observed, side="right"), 0)
            counts = np.bincount(codes, minlength=len(edges) + 2).astype(float)
            self.edges_.append(edges)
            self.priors_.append(counts / counts.sum() if counts.sum() else np.r_[1., np.zeros(len(edges) + 1)])
        return self

    def transform(self, x):
        pieces = []
        for column, edges, prior in zip(np.asarray(x, dtype=float).T, self.edges_, self.priors_):
            missing = ~np.isfinite(column) | (column < 0)
            codes = np.where(column > 0, 1 + np.searchsorted(edges, column, side="right"), 0)
            encoded = np.eye(len(prior))[codes]
            encoded[missing] = prior
            if self.presence:
                encoded = np.column_stack((1.0 - encoded[:, 0], encoded))
            pieces.append(encoded)
        return np.hstack(pieces)

def missing_to_nan(x):
    x = np.asarray(x, dtype=float).copy()
    x[x == -1] = np.nan
    return x

def word_transform(x, kind="raw", scale=1000.0):
    if kind == "raw":
        return x
    if kind == "log":
        return np.log1p(scale * x)
    if kind == "sqrt":
        return np.sqrt(x)
    if kind == "binary":
        return np.where(np.isnan(x), np.nan, (x > 0).astype(float))
    if kind == "log_binary":
        return np.hstack((np.log1p(scale * x),
            np.where(np.isnan(x), np.nan, (x > 0).astype(float))))
    raise ValueError(kind)

def make_model(config):
    c = dict(config)
    family = c.pop("family")
    transform = c.pop("transform", "raw")
    scale = c.pop("scale", 1000.0)
    impute = c.pop("impute", "median")
    indicators = c.pop("indicators", False)
    standardize = c.pop("standardize", True)
    bins = c.pop("bins", 3)
    presence = c.pop("presence", False)
    steps = [("missing", FunctionTransformer(missing_to_nan)),
             ("transform", FunctionTransformer(word_transform,
                kw_args={"kind": transform, "scale": scale}))]
    if impute != "native":
        steps.append(("imputer", SimpleImputer(strategy=impute,
                     add_indicator=indicators, keep_empty_features=True)))
    if family == "binned":
        steps = [("frequency_bins", FrequencyBinner(bins=bins, presence=presence))]
    if family in ("svm", "logistic", "binned") and standardize:
        steps.append(("scaler", StandardScaler()))
    if family == "baseline":
        estimator = GaussianNB(**c)
    elif family == "bernoulli":
        estimator = BernoulliNB(**c)
    elif family in ("logistic", "binned"):
        estimator = LogisticRegression(max_iter=3000, random_state=MODEL_SEED, **c)
    elif family == "svm":
        estimator = SVC(kernel="rbf", probability=True, cache_size=256,
                        random_state=MODEL_SEED, **c)
    elif family == "gb":
        estimator = GradientBoostingClassifier(random_state=MODEL_SEED, **c)
    elif family == "hist":
        estimator = HistGradientBoostingClassifier(early_stopping=False,
                    random_state=MODEL_SEED, **c)
    elif family == "extra":
        estimator = ExtraTreesClassifier(n_jobs=2, random_state=MODEL_SEED, **c)
    elif family == "forest":
        estimator = RandomForestClassifier(n_jobs=2, random_state=MODEL_SEED, **c)
    else:
        raise ValueError(family)
    return Pipeline(steps + [("model", estimator)])

class AdalineSGD(ClassifierMixin, BaseEstimator):
    """Adaline-style linear squared-error classifier with averaged SGD.

    Encode ham/spam as -1/+1, fit a linear output with squared-error loss, and
    use that continuous output to rank emails. L2 regularization and averaging
    are practical extensions to textbook Adaline. This is not a probability model.
    """
    def __init__(self, alpha=0.0001, eta0=0.001):
        self.alpha = alpha
        self.eta0 = eta0

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.model_ = SGDRegressor(loss="squared_error", penalty="l2", alpha=self.alpha,
            learning_rate="invscaling", eta0=self.eta0, average=True,
            max_iter=3000, tol=1e-5, random_state=MODEL_SEED)
        self.model_.fit(X, 2 * np.asarray(y) - 1)
        return self

    def decision_function(self, X):
        return self.model_.predict(X)

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

def make_course_model(config):
    c = dict(config)
    if c["family"] in {"baseline", "logistic", "binned", "svm", "gb", "hist", "extra", "forest", "bernoulli"}:
        return make_model(c)
    family = c.pop("family")
    transform = c.pop("transform", "log_binary")
    impute = c.pop("impute", "mean")
    indicators = c.pop("indicators", False)
    standardize = c.pop("standardize", family not in {"tree", "adaboost"})
    components = c.pop("components", None)
    selected_features = c.pop("selected_features", None)
    steps = [("missing", FunctionTransformer(missing_to_nan)),
             ("transform", FunctionTransformer(word_transform, kw_args={"kind": transform})),
             ("imputer", SimpleImputer(strategy=impute, add_indicator=indicators,
                                        keep_empty_features=True))]
    if standardize:
        steps.append(("scaler", StandardScaler()))
    if selected_features is not None:
        steps.append(("feature_selection", SelectKBest(f_classif, k=selected_features)))
    if components is not None:
        steps.append(("pca", PCA(n_components=components, svd_solver="full")))
    if family == "perceptron":
        estimator = Perceptron(penalty="l2", max_iter=3000, tol=1e-4,
                               random_state=MODEL_SEED, **c)
    elif family == "adaline":
        estimator = AdalineSGD(**c)
    elif family == "linear_svm":
        estimator = LinearSVC(dual=False, max_iter=10000, random_state=MODEL_SEED, **c)
    elif family == "tree":
        estimator = DecisionTreeClassifier(random_state=MODEL_SEED, **c)
    elif family == "knn":
        estimator = KNeighborsClassifier(n_jobs=2, **c)
    elif family == "adaboost":
        depth = c.pop("depth")
        base_tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=MODEL_SEED)
        estimator = AdaBoostClassifier(estimator=base_tree, random_state=MODEL_SEED, **c)
    elif family == "lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", **c)
    elif family == "lda_projection":
        # With two classes LDA offers at most one discriminant direction.
        steps.append(("lda", LinearDiscriminantAnalysis(solver="svd", n_components=1)))
        estimator = LogisticRegression(max_iter=3000, **c)
    elif family in {"pca_logistic", "selected_logistic"}:
        estimator = LogisticRegression(max_iter=3000, random_state=MODEL_SEED, **c)
    elif family == "pca_svm":
        estimator = SVC(kernel="rbf", probability=True, random_state=MODEL_SEED, **c)
    elif family == "mlp":
        c.setdefault("max_iter", 1500)
        c.setdefault("max_fun", 25000)
        estimator = MLPClassifier(solver="lbfgs", random_state=MODEL_SEED, **c)
    else:
        raise ValueError(family)
    return Pipeline(steps + [("model", estimator)])

class AdvancedBinner(TransformerMixin, BaseEstimator):
    def __init__(self,bins=5,spacing='quantile',presence=False,missing='prior',smooth=0.):
        self.bins=bins; self.spacing=spacing; self.presence=presence
        self.missing=missing; self.smooth=smooth
    def fit(self,X,y=None):
        self.edges_=[]; self.priors_=[]
        for col in np.asarray(X).T:
            obs=col[np.isfinite(col)&(col>=0)]; pos=obs[obs>0]
            if not len(pos):edges=np.array([])
            elif self.spacing=='quantile':edges=np.unique(np.quantile(pos,np.arange(1,self.bins)/self.bins))
            else:edges=np.unique(np.expm1(np.linspace(np.log1p(1000*pos.min()),np.log1p(1000*pos.max()),self.bins+1)[1:-1])/1000)
            codes=np.where(obs>0,1+np.searchsorted(edges,obs,side='right'),0)
            counts=np.bincount(codes,minlength=len(edges)+2).astype(float)+self.smooth
            prior=counts/counts.sum() if counts.sum() else np.r_[1.,np.zeros(len(edges)+1)]
            self.edges_.append(edges);self.priors_.append(prior)
        return self
    def transform(self,X):
        blocks=[]
        for col,edges,prior in zip(np.asarray(X).T,self.edges_,self.priors_):
            miss=~np.isfinite(col)|(col<0)
            codes=np.where(col>0,1+np.searchsorted(edges,col,side='right'),0)
            encoded=np.eye(len(prior))[codes]
            encoded[miss]=prior if self.missing in ('prior','indicator') else 0.
            if self.presence:encoded=np.c_[1-encoded[:,0],encoded]
            if self.missing=='indicator':encoded=np.c_[encoded,miss.astype(float)]
            blocks.append(encoded)
        return np.hstack(blocks)

class PowerPresence(TransformerMixin,BaseEstimator):
    def __init__(self,power=.5):self.power=power
    def fit(self,X,y=None):return self
    def transform(self,X):
        X=missing_to_nan(X)
        return np.c_[np.power(X,self.power),np.where(np.isnan(X),np.nan,(X>0).astype(float))]

class KernelSVC(ClassifierMixin,BaseEstimator):
    def __init__(self,kernel='laplacian',C=1.,gamma=.01):self.kernel=kernel;self.C=C;self.gamma=gamma
    def _kernel(self,X,Y):
        return (laplacian_kernel if self.kernel=='laplacian' else chi2_kernel)(X,Y,gamma=self.gamma)
    def fit(self,X,y):
        self.train_=np.asarray(X).copy();self.classes_=np.unique(y)
        self.model_=SVC(kernel='precomputed',C=self.C,probability=True,random_state=MODEL_SEED)
        self.model_.fit(self._kernel(self.train_,self.train_),y)
        return self
    def predict_proba(self,X):return self.model_.predict_proba(self._kernel(X,self.train_))

class PresenceWeight(TransformerMixin,BaseEstimator):
    def __init__(self,weight=1.):self.weight=weight
    def fit(self,X,y=None):return self
    def transform(self,X):
        X=np.asarray(X).copy();X[:,X.shape[1]//2:]*=self.weight;return X

def build(config):
    c=dict(config);family=c.pop('family')
    if family in ('knn_binned','knn_hist','knn_svm'):
        kind=c.pop('transform','log');neighbors=c.pop('neighbors',15)
        weights=c.pop('weights','uniform');indicators=c.pop('indicators',False)
        steps=[('missing',FunctionTransformer(missing_to_nan)),
               ('features',FunctionTransformer(word_transform,kw_args={'kind':kind})),
               ('impute',KNNImputer(n_neighbors=neighbors,weights=weights,add_indicator=indicators,keep_empty_features=True))]
        if family=='knn_binned':
            C=c.pop('C');steps.append(('bins',AdvancedBinner(bins=c.pop('bins',5),presence=c.pop('presence',False))))
            estimator=LogisticRegression(C=C,max_iter=3000,**c)
        elif family=='knn_hist':estimator=HistGradientBoostingClassifier(early_stopping=False,random_state=MODEL_SEED,**c)
        else:
            steps.append(('scale',StandardScaler()));estimator=SVC(probability=True,random_state=MODEL_SEED,**c)
        return Pipeline(steps+[('model',estimator)])
    if family=='advanced_bins':
        C=c.pop('C');standardize=c.pop('standardize',False);weight=c.pop('ham_weight',1.)
        steps=[('bins',AdvancedBinner(**c))]
        if standardize:steps.append(('scale',StandardScaler()))
        return Pipeline(steps+[('model',LogisticRegression(C=C,max_iter=3000,class_weight={0:weight,1:1.}))])
    if family in ('spline','interactions','power','knn_logistic','iterative_logistic','chi2_logistic','kernel_svm'):
        C=c.pop('C');kind=c.pop('transform','log_binary');impute=c.pop('impute','mean')
        scale=c.pop('scale',1000.);indicators=c.pop('indicators',False)
        steps=[('missing',FunctionTransformer(missing_to_nan)),('features',FunctionTransformer(word_transform,kw_args={'kind':kind,'scale':scale}))]
        if family=='power':steps=[('features',PowerPresence(c.pop('power')))]
        if family=='knn_logistic':
            presence_weight=c.pop('presence_weight',1.);pre_scale=c.pop('pre_scale',False)
            if presence_weight!=1.:steps.append(('presence_weight',PresenceWeight(presence_weight)))
            if pre_scale:steps.append(('pre_scale',StandardScaler()))
            steps.append(('impute',KNNImputer(n_neighbors=c.pop('neighbors'),weights=c.pop('weights','distance'),add_indicator=indicators,keep_empty_features=True)))
        elif family=='iterative_logistic':
            mode=c.pop('estimator','bayesian');max_iter=c.pop('impute_iter',10)
            estimator=BayesianRidge() if mode=='bayesian' else ExtraTreesRegressor(n_estimators=30,max_depth=5,min_samples_leaf=5,n_jobs=1,random_state=MODEL_SEED)
            steps.append(('impute',IterativeImputer(estimator=estimator,max_iter=max_iter,initial_strategy='mean',min_value=0,add_indicator=indicators,random_state=MODEL_SEED,keep_empty_features=True,n_nearest_features=c.pop('nearest_features',None))))
        else:steps.append(('impute',SimpleImputer(strategy=impute,add_indicator=indicators,keep_empty_features=True)))
        if family=='spline':steps.append(('spline',SplineTransformer(n_knots=c.pop('knots'),degree=c.pop('degree'),include_bias=False,knots='quantile',extrapolation='constant')))
        if family=='interactions':
            selected=c.pop('features',30)
            steps += [('select',SelectKBest(f_classif,k=selected)),('poly',PolynomialFeatures(degree=2,interaction_only=c.pop('interaction_only',True),include_bias=False))]
        if family=='chi2_logistic':steps.append(('chi2',AdditiveChi2Sampler(sample_steps=c.pop('steps',2))))
        if family=='kernel_svm':
            kernel=c.pop('kernel');gamma=c.pop('gamma')
            if kernel=='laplacian':steps.append(('scale',StandardScaler()))
            return Pipeline(steps+[('model',KernelSVC(C=C,kernel=kernel,gamma=gamma))])
        steps.append(('scale',StandardScaler()))
        return Pipeline(steps+[('model',LogisticRegression(C=C,max_iter=3000,**c))])
    ham_weight=c.pop('ham_weight',None)
    model=make_course_model({'family':family,**c})
    if family in ('forest','extra'):model.set_params(model__n_jobs=1)
    if ham_weight is not None and family in ('logistic','hist','svm','forest','extra'):
        model.set_params(model__class_weight={0:ham_weight,1:1.})
    return model

MODEL_GRIDS = [
    {'C': [0.03, 0.1, 0.3, 1.0], 'bins': [3, 5, 8], 'family': ['binned'], 'presence': [True], 'standardize': [False]},
    {'family': ['baseline'], 'impute': ['mean']},
    {'family': ['original']},
    {'C': [0.01, 0.1, 1.0, 10.0], 'family': ['logistic'], 'impute': ['mean'], 'standardize': [False, True], 'transform': ['binary', 'log', 'log_binary']},
    {'family': ['hist'], 'impute': ['median', 'native'], 'l2_regularization': [1.0], 'learning_rate': [0.05], 'max_iter': [250], 'max_leaf_nodes': [15, 7], 'min_samples_leaf': [15]},
    {'family': ['hist'], 'impute': ['native'], 'l2_regularization': [1.0, 10.0], 'learning_rate': [0.05], 'max_iter': [150, 400], 'max_leaf_nodes': [15, 3, 7], 'min_samples_leaf': [20]},
    {'family': ['hist'], 'impute': ['native'], 'l2_regularization': [0.1, 1.0, 10.0, 100.0], 'learning_rate': [0.05], 'max_iter': [150, 400], 'max_leaf_nodes': [15, 3, 31, 7], 'min_samples_leaf': [10, 30]},
    {'C': [0.3, 10.0, 3.0, 30.0], 'family': ['svm'], 'gamma': [0.01], 'impute': ['mean'], 'standardize': [False], 'transform': ['binary']},
    {'C': [0.3, 1.0, 10.0, 100.0, 3.0, 30.0], 'family': ['svm'], 'gamma': [0.001, 0.003, 0.01, 0.03, 0.1], 'impute': ['mean'], 'standardize': [True], 'transform': ['binary', 'log', 'log_binary']},
    {'C': [10.0, 3.0, 30.0], 'family': ['svm'], 'gamma': [0.03, 0.1], 'impute': ['mean'], 'standardize': [False], 'transform': ['binary']},
    {'C': [10.0, 3.0, 30.0], 'family': ['svm'], 'gamma': [0.01, 0.03, 0.1], 'impute': ['mean'], 'standardize': [False], 'transform': ['log_binary']},
    {'C': [0.3, 3.0, 30.0], 'family': ['svm'], 'gamma': [0.001, 0.01], 'impute': ['mean'], 'standardize': [False], 'transform': ['log']},
    {'C': [0.3, 3.0, 30.0], 'family': ['svm'], 'gamma': [0.001], 'impute': ['mean'], 'standardize': [False], 'transform': ['binary']},
    {'family': ['gb'], 'impute': ['mean', 'median'], 'indicators': [False, True], 'learning_rate': [0.05], 'max_depth': [2], 'min_samples_leaf': [8], 'n_estimators': [300]},
    {'family': ['gb'], 'impute': ['mean', 'median'], 'indicators': [False, True], 'learning_rate': [0.03, 0.1], 'max_depth': [1, 2, 3], 'min_samples_leaf': [10], 'n_estimators': [150, 500]},
    {'C': [0.03, 0.3, 3.0], 'bins': [2, 3, 5], 'family': ['binned'], 'standardize': [False, True]},
    {'alpha': [0.0001, 0.01], 'family': ['perceptron'], 'transform': ['binary', 'log', 'log_binary']},
    {'C': [0.01, 0.1, 1.0, 10.0], 'family': ['linear_svm'], 'transform': ['binary', 'log_binary']},
    {'C': [1.0], 'family': ['lda_projection'], 'transform': ['log', 'log_binary']},
    {'C': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 'family': ['chi2_logistic'], 'transform': ['binary', 'log', 'log_binary']},
    {'family': ['tree'], 'impute': ['median'], 'indicators': [True], 'max_depth': [2, 4, 6, None], 'min_samples_leaf': [15, 5], 'transform': ['raw']},
    {'alpha': [0.0001, 0.01], 'eta0': [0.001], 'family': ['adaline'], 'transform': ['binary', 'log', 'log_binary']},
    {'family': ['knn'], 'n_neighbors': [15, 31, 5], 'p': [1, 2], 'transform': ['binary', 'log'], 'weights': ['distance', 'uniform']},
    {'family': ['extra', 'forest'], 'impute': ['mean'], 'max_features': [0.7], 'min_samples_leaf': [2], 'n_estimators': [400], 'transform': ['log', 'raw']},
    {'family': ['lda'], 'transform': ['binary', 'log', 'log_binary', 'raw']},
    {'C': [3.0], 'components': [0.95, 10, 20], 'family': ['pca_svm'], 'gamma': [0.01], 'transform': ['log']},
    {'C': [0.01], 'components': [0.95, 10, 20], 'family': ['pca_logistic'], 'transform': ['log_binary']},
    {'C': [0.01], 'family': ['selected_logistic'], 'selected_features': [10, 20, 40], 'transform': ['log_binary']},
    {'depth': [1, 2], 'family': ['adaboost'], 'learning_rate': [0.1, 1.0], 'n_estimators': [100, 300], 'transform': ['raw']},
    {'family': ['gb'], 'impute': ['mean'], 'learning_rate': [0.05], 'max_depth': [1, 2, 3], 'min_samples_leaf': [10], 'n_estimators': [100, 600]},
    {'alpha': [1.0, 10.0], 'family': ['mlp'], 'hidden_layer_sizes': [[16]], 'transform': ['binary', 'log_binary']},
    {'C': [0.03, 0.1, 0.3], 'family': ['logistic'], 'impute': ['mean'], 'indicators': [True], 'transform': ['binary', 'log_binary']},
    {'C': [1.0], 'family': ['logistic'], 'impute': ['mean'], 'transform': ['binary', 'log', 'raw']},
    {'C': [3.0], 'family': ['svm'], 'gamma': ['scale'], 'impute': ['mean', 'median'], 'transform': ['log', 'raw', 'sqrt']},
    {'C': [3.0], 'family': ['svm'], 'gamma': ['scale'], 'impute': ['mean'], 'indicators': [True], 'transform': ['log']},
    {'alpha': [1.0], 'family': ['bernoulli'], 'impute': ['median'], 'transform': ['binary']},
    {'C': [0.01, 0.03, 0.1, 0.3, 1, 3], 'bins': [12, 2, 3, 5, 8], 'family': ['advanced_bins'], 'missing': ['indicator', 'prior'], 'presence': [False, True], 'spacing': ['log', 'quantile']},
    {'C': [0.03, 0.1, 0.3, 1], 'bins': [3, 5, 8], 'family': ['advanced_bins'], 'missing': ['prior'], 'presence': [True], 'smooth': [0.5, 10.0, 2.0], 'spacing': ['quantile']},
    {'C': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 'family': ['logistic'], 'impute': ['mean', 'median'], 'indicators': [False, True], 'scale': [10.0, 100.0, 1000.0, 10000.0], 'transform': ['log_binary']},
    {'C': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 'family': ['power'], 'power': [0.1, 0.25, 0.5, 0.75, 1.0]},
    {'C': [0.003, 0.03, 0.3, 3.0], 'degree': [1, 2, 3], 'family': ['spline'], 'indicators': [False, True], 'knots': [3, 5, 8], 'transform': ['log']},
    {'C': [0.001, 0.003, 0.01, 0.03, 0.1], 'family': ['interactions'], 'features': [15, 30, 60], 'interaction_only': [False, True]},
    {'C': [0.003, 0.01, 0.03, 0.1], 'family': ['knn_logistic'], 'neighbors': [15, 3, 31, 7], 'weights': ['distance', 'uniform']},
    {'C': [1.0, 10.0, 3.0, 30.0], 'family': ['kernel_svm'], 'gamma': [0.003, 0.01, 0.03, 0.1], 'kernel': ['laplacian'], 'transform': ['binary', 'log']},
    {'C': [0.1, 1.0, 10.0, 100.0], 'family': ['kernel_svm'], 'gamma': [0.01, 0.1, 1.0], 'kernel': ['chi2'], 'transform': ['binary', 'log']},
    {'family': ['extra', 'forest'], 'impute': ['mean'], 'max_features': [0.3, 0.7, 1.0], 'min_samples_leaf': [1, 10, 2, 5], 'n_estimators': [300]},
    {'alpha': [0.1, 1.0, 10.0, 100.0], 'family': ['mlp'], 'hidden_layer_sizes': [[16], [32, 16], [32], [8]], 'max_fun': [50000], 'max_iter': [3000], 'transform': ['binary', 'log_binary']},
    {'C': [0.01, 0.1, 0.3, 1.0], 'family': ['logistic'], 'ham_weight': [0.25, 0.5, 2.0, 4.0], 'impute': ['mean'], 'transform': ['log_binary']},
    {'C': [0.01, 0.1, 0.3, 1.0], 'bins': [5], 'family': ['advanced_bins'], 'ham_weight': [0.25, 0.5, 2.0, 4.0], 'presence': [True]},
    {'C': [0.003, 0.01, 0.03, 0.1, 0.3], 'family': ['knn_logistic'], 'indicators': [False, True], 'neighbors': [1, 15, 3, 31, 63, 7], 'pre_scale': [False, True], 'transform': ['log', 'log_binary'], 'weights': ['distance', 'uniform']},
    {'C': [0.01, 0.03, 0.1], 'family': ['knn_logistic'], 'neighbors': [15, 3, 31], 'presence_weight': [0.25, 2.0, 5.0], 'scale': [100.0, 1000.0, 10000.0], 'transform': ['log_binary'], 'weights': ['uniform']},
    {'C': [0.003, 0.01, 0.03, 0.1, 0.3], 'estimator': ['bayesian', 'trees'], 'family': ['iterative_logistic'], 'impute_iter': [5], 'indicators': [False, True], 'nearest_features': [15], 'transform': ['log', 'log_binary']},
    {'C': [0.03, 0.1, 0.3, 1.0], 'family': ['knn_binned'], 'indicators': [False, True], 'neighbors': [15, 3, 31, 7], 'presence': [False, True]},
    {'family': ['knn_hist'], 'l2_regularization': [0.1, 1.0, 10.0], 'learning_rate': [0.05], 'max_iter': [250], 'max_leaf_nodes': [15, 3, 7], 'min_samples_leaf': [15], 'neighbors': [31, 7]},
    {'C': [1.0, 10.0, 3.0, 30.0], 'family': ['knn_svm'], 'gamma': [0.003, 0.01, 0.03], 'neighbors': [31, 7], 'transform': ['log', 'log_binary']},
]

FULL_LIBRARY = list(ParameterGrid(MODEL_GRIDS))
assert len(FULL_LIBRARY) == 1841
assert len({identity(c) for c in FULL_LIBRARY}) == 1841
