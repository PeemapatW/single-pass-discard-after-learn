"""
Hyperparameter variation tests for all spdal classifiers.

For each classifier, tests exercise key hyperparameter values other than
the defaults. Every test asserts that the new list-of-dicts implementation
produces identical predictions and neuron values to the deprecated
DataFrame implementation, matching the same verification angles used in
test_classifiers.py (predictions, neuron values, chunked partial_fit).

TRACED-specific notes
---------------------
The production TRACED dropped four research parameters
(variance_threshold, components, min_dims, threshold_percentile).
Tests for TRACED use the remaining parameters.  When calling the
deprecated TRACED the removed parameters are left at their defaults
(variance_threshold=1, components=None, threshold_percentile=100,
min_dims=None) so the two code paths are equivalent.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score

from spdal import LRHE, VEBF, SCIL, SHEF, D4, TRACED


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(n_samples=300, random_state=42)
    return X[:200], X[200:], y[:200]


@pytest.fixture(scope="module")
def binary_chunks():
    X, y = make_classification(n_samples=300, random_state=42)
    chunks = [(X[i:i+50], y[i:i+50]) for i in range(0, 200, 50)]
    return chunks, X[200:]


@pytest.fixture(scope="module")
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X[:120], X[120:], y[:120]


@pytest.fixture(scope="module")
def iris_chunks():
    X, y = load_iris(return_X_y=True)
    chunks = [(X[i:i+30], y[i:i+30]) for i in range(0, 120, 30)]
    return chunks, X[120:], np.unique(y)


@pytest.fixture(scope="module")
def dep_module():
    repo_root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        "spdal_deprecated", repo_root / "deprecated" / "spdal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_classifiers.py)
# ---------------------------------------------------------------------------

def _sorted_neurons(neuron_list):
    if hasattr(neuron_list, 'iterrows'):
        neurons = [row.to_dict() for _, row in neuron_list.iterrows()]
    else:
        neurons = list(neuron_list)
    return sorted(neurons, key=lambda n: (n['y'], float(np.linalg.norm(n['center']))))


def _assert_neurons_close(new_neurons, dep_neurons, fields, rtol=1e-10):
    assert len(new_neurons) == len(dep_neurons), (
        f"neuron count mismatch: new={len(new_neurons)}, dep={len(dep_neurons)}"
    )
    for i, (n, d) in enumerate(zip(new_neurons, dep_neurons)):
        assert n['y'] == d['y'], f"neuron {i}: y mismatch {n['y']} != {d['y']}"
        assert n['n'] == d['n'], f"neuron {i}: n mismatch {n['n']} != {d['n']}"
        for field in fields:
            np.testing.assert_allclose(
                n[field], d[field], rtol=rtol,
                err_msg=f"neuron {i}: field '{field}' differs"
            )


def _assert_preds_match(new, dep, X):
    np.testing.assert_array_equal(
        new.predict(X), dep.predict(X),
        err_msg="predictions differ between new and deprecated"
    )


def _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes, fit_kwargs=None):
    """Fit both classifiers chunk by chunk and assert predictions match after each chunk."""
    fit_kwargs = fit_kwargs or {}
    for Xc, yc in chunks:
        clf_new.partial_fit(Xc, yc, classes=classes, **fit_kwargs)
        clf_dep.partial_fit(Xc, yc, classes=classes, **fit_kwargs)
        _assert_preds_match(clf_new, clf_dep, X_test)


# ---------------------------------------------------------------------------
# LRHE hyperparameter tests
# Params: alpha=0.5, theta=0, delta=1, epsilon=1e-10
# ---------------------------------------------------------------------------

LRHE_PARAM_CASES = [
    pytest.param({"alpha": 0.2},              id="alpha=0.2"),
    pytest.param({"alpha": 0.8},              id="alpha=0.8"),
    pytest.param({"theta": -0.5},             id="theta=-0.5"),
    pytest.param({"theta": 0.5},              id="theta=0.5"),
    pytest.param({"delta": 2},                id="delta=2"),
    pytest.param({"alpha": 0.3, "theta": -0.3, "delta": 2}, id="combined"),
]


class TestLRHEHyperparams:

    @pytest.mark.parametrize("kwargs", LRHE_PARAM_CASES)
    def test_binary_predictions_match_deprecated(self, binary_data, dep_module, kwargs):
        X_train, X_test, y_train = binary_data
        clf_new = LRHE(**kwargs)
        clf_dep = dep_module.LRHE(**kwargs)
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", LRHE_PARAM_CASES)
    def test_iris_predictions_match_deprecated(self, iris_data, dep_module, kwargs):
        X_train, X_test, y_train = iris_data
        clf_new = LRHE(**kwargs)
        clf_dep = dep_module.LRHE(**kwargs)
        clf_new.fit(X_train, y_train)
        clf_dep.fit(X_train, y_train)
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", LRHE_PARAM_CASES)
    def test_binary_neuron_values_match_deprecated(self, binary_data, dep_module, kwargs):
        X_train, _, y_train = binary_data
        clf_new = LRHE(**kwargs)
        clf_dep = dep_module.LRHE(**kwargs)
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component'],
        )

    @pytest.mark.parametrize("kwargs", LRHE_PARAM_CASES)
    def test_binary_chunked_matches_deprecated(self, binary_chunks, dep_module, kwargs):
        chunks, X_test = binary_chunks
        clf_new = LRHE(**kwargs)
        clf_dep = dep_module.LRHE(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=[0, 1])

    @pytest.mark.parametrize("kwargs", LRHE_PARAM_CASES)
    def test_iris_chunked_matches_deprecated(self, iris_chunks, dep_module, kwargs):
        chunks, X_test, classes = iris_chunks
        clf_new = LRHE(**kwargs)
        clf_dep = dep_module.LRHE(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=classes)


# ---------------------------------------------------------------------------
# VEBF hyperparameter tests
# Params: theta=0, delta=1, epsilon=1e-10
# ---------------------------------------------------------------------------

VEBF_PARAM_CASES = [
    pytest.param({"theta": -0.5},  id="theta=-0.5"),
    pytest.param({"theta": 0.5},   id="theta=0.5"),
    pytest.param({"delta": 2},     id="delta=2"),
    pytest.param({"delta": 3},     id="delta=3"),
    pytest.param({"theta": -0.3, "delta": 2}, id="combined"),
]


class TestVEBFHyperparams:

    @pytest.mark.parametrize("kwargs", VEBF_PARAM_CASES)
    def test_binary_predictions_match_deprecated(self, binary_data, dep_module, kwargs):
        X_train, X_test, y_train = binary_data
        clf_new = VEBF(**kwargs)
        clf_dep = dep_module.VEBF(**kwargs)
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", VEBF_PARAM_CASES)
    def test_iris_predictions_match_deprecated(self, iris_data, dep_module, kwargs):
        X_train, X_test, y_train = iris_data
        clf_new = VEBF(**kwargs)
        clf_dep = dep_module.VEBF(**kwargs)
        clf_new.fit(X_train, y_train)
        clf_dep.fit(X_train, y_train)
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", VEBF_PARAM_CASES)
    def test_binary_neuron_values_match_deprecated(self, binary_data, dep_module, kwargs):
        X_train, _, y_train = binary_data
        clf_new = VEBF(**kwargs)
        clf_dep = dep_module.VEBF(**kwargs)
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component'],
        )

    @pytest.mark.parametrize("kwargs", VEBF_PARAM_CASES)
    def test_binary_chunked_matches_deprecated(self, binary_chunks, dep_module, kwargs):
        chunks, X_test = binary_chunks
        clf_new = VEBF(**kwargs)
        clf_dep = dep_module.VEBF(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=[0, 1])

    @pytest.mark.parametrize("kwargs", VEBF_PARAM_CASES)
    def test_iris_chunked_matches_deprecated(self, iris_chunks, dep_module, kwargs):
        chunks, X_test, classes = iris_chunks
        clf_new = VEBF(**kwargs)
        clf_dep = dep_module.VEBF(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=classes)


# ---------------------------------------------------------------------------
# SCIL hyperparameter tests
# Params: N0=3, eta=2, delta=1, epsilon=1e-10, theta=0
# ---------------------------------------------------------------------------

SCIL_PARAM_CASES = [
    pytest.param({"N0": 1},              id="N0=1"),
    pytest.param({"N0": 5},              id="N0=5"),
    pytest.param({"eta": 3},             id="eta=3"),
    pytest.param({"theta": -0.5},        id="theta=-0.5"),
    pytest.param({"delta": 2},           id="delta=2"),
    pytest.param({"N0": 1, "eta": 3, "delta": 2}, id="combined"),
]


class TestSCILHyperparams:

    @pytest.mark.parametrize("kwargs", SCIL_PARAM_CASES)
    def test_binary_predictions_match_deprecated(self, binary_data, dep_module, kwargs):
        X_train, X_test, y_train = binary_data
        clf_new = SCIL(**kwargs)
        clf_dep = dep_module.SCIL(**kwargs)
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", SCIL_PARAM_CASES)
    def test_iris_predictions_match_deprecated(self, iris_data, dep_module, kwargs):
        X_train, X_test, y_train = iris_data
        clf_new = SCIL(**kwargs)
        clf_dep = dep_module.SCIL(**kwargs)
        clf_new.fit(X_train, y_train)
        clf_dep.fit(X_train, y_train)
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", SCIL_PARAM_CASES)
    def test_binary_neuron_values_match_deprecated(self, binary_data, dep_module, kwargs):
        X_train, _, y_train = binary_data
        clf_new = SCIL(**kwargs)
        clf_dep = dep_module.SCIL(**kwargs)
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component', 'variance'],
        )

    @pytest.mark.parametrize("kwargs", SCIL_PARAM_CASES)
    def test_binary_chunked_matches_deprecated(self, binary_chunks, dep_module, kwargs):
        chunks, X_test = binary_chunks
        clf_new = SCIL(**kwargs)
        clf_dep = dep_module.SCIL(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=[0, 1])

    @pytest.mark.parametrize("kwargs", SCIL_PARAM_CASES)
    def test_iris_chunked_matches_deprecated(self, iris_chunks, dep_module, kwargs):
        chunks, X_test, classes = iris_chunks
        clf_new = SCIL(**kwargs)
        clf_dep = dep_module.SCIL(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=classes)


# ---------------------------------------------------------------------------
# SHEF hyperparameter tests
# Params: M=3, r=1.5, epsilon=1e-10
# ---------------------------------------------------------------------------

SHEF_PARAM_CASES = [
    pytest.param({"M": 1},        id="M=1"),
    pytest.param({"M": 5},        id="M=5"),
    pytest.param({"r": 1.0},      id="r=1.0"),
    pytest.param({"r": 2.0},      id="r=2.0"),
    pytest.param({"M": 2, "r": 2.0}, id="combined"),
]


class TestSHEFHyperparams:

    @pytest.mark.parametrize("kwargs", SHEF_PARAM_CASES)
    def test_binary_predictions_match_deprecated(self, binary_data, dep_module, kwargs):
        X_train, X_test, y_train = binary_data
        clf_new = SHEF(**kwargs)
        clf_dep = dep_module.SHEF(**kwargs)
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", SHEF_PARAM_CASES)
    def test_iris_predictions_match_deprecated(self, iris_data, dep_module, kwargs):
        X_train, X_test, y_train = iris_data
        clf_new = SHEF(**kwargs)
        clf_dep = dep_module.SHEF(**kwargs)
        clf_new.fit(X_train, y_train)
        clf_dep.fit(X_train, y_train)
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", SHEF_PARAM_CASES)
    def test_binary_neuron_values_match_deprecated(self, binary_data, dep_module, kwargs):
        X_train, _, y_train = binary_data
        clf_new = SHEF(**kwargs)
        clf_dep = dep_module.SHEF(**kwargs)
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov'],
        )

    @pytest.mark.parametrize("kwargs", SHEF_PARAM_CASES)
    def test_binary_chunked_matches_deprecated(self, binary_chunks, dep_module, kwargs):
        chunks, X_test = binary_chunks
        clf_new = SHEF(**kwargs)
        clf_dep = dep_module.SHEF(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=[0, 1])

    @pytest.mark.parametrize("kwargs", SHEF_PARAM_CASES)
    def test_iris_chunked_matches_deprecated(self, iris_chunks, dep_module, kwargs):
        chunks, X_test, classes = iris_chunks
        clf_new = SHEF(**kwargs)
        clf_dep = dep_module.SHEF(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=classes)


# ---------------------------------------------------------------------------
# D4 hyperparameter tests
# Params: norm=2, delta=1, width_parameter=1, reduce_dims=0, epsilon=1e-10, r=1.5, threshold=15
# ---------------------------------------------------------------------------

D4_PARAM_CASES = [
    pytest.param({"norm": 1},                  id="norm=1"),
    pytest.param({"delta": 2},                 id="delta=2"),
    pytest.param({"width_parameter": 0.5},     id="width_parameter=0.5"),
    pytest.param({"r": 1.0},                   id="r=1.0"),
    pytest.param({"r": 2.0},                   id="r=2.0"),
    pytest.param({"threshold": 10},            id="threshold=10"),
    pytest.param({"threshold": 30},            id="threshold=30"),
    pytest.param({"reduce_dims": 2},           id="reduce_dims=2"),
    pytest.param({"width_parameter": 0.5, "r": 2.0, "threshold": 20}, id="combined"),
]


class TestD4Hyperparams:

    @pytest.mark.parametrize("kwargs", D4_PARAM_CASES)
    def test_iris_predictions_match_deprecated(self, iris_data, dep_module, kwargs):
        X_train, X_test, y_train = iris_data
        clf_new = D4(**kwargs)
        clf_dep = dep_module.D4(**kwargs)
        clf_new.fit(X_train, y_train)
        clf_dep.fit(X_train, y_train)
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", D4_PARAM_CASES)
    def test_iris_neuron_values_match_deprecated(self, iris_data, dep_module, kwargs):
        X_train, _, y_train = iris_data
        clf_new = D4(**kwargs)
        clf_dep = dep_module.D4(**kwargs)
        clf_new.fit(X_train, y_train)
        clf_dep.fit(X_train, y_train)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component', 'variance'],
        )

    @pytest.mark.parametrize("kwargs", D4_PARAM_CASES)
    def test_iris_chunked_matches_deprecated(self, iris_chunks, dep_module, kwargs):
        chunks, X_test, classes = iris_chunks
        clf_new = D4(**kwargs)
        clf_dep = dep_module.D4(**kwargs)
        _assert_chunk_preds_match(clf_new, clf_dep, chunks, X_test, classes=classes)


# ---------------------------------------------------------------------------
# TRACED hyperparameter tests
# Production params: norm=2, method="overlap-outside", r=_SQRT_2PI, N0=3,
#   delta=2, alpha=0.5, beta=0.01, distance_metric='boundary',
#   width_parameter=1, reduce_dims=1, threshold=15
#
# Removed params (deprecated defaults used for equivalence):
#   variance_threshold=1, components=None, min_dims=None,
#   threshold_percentile=100, pca_strategy='bottom',
#   overlap_selection='parallel'
# ---------------------------------------------------------------------------

# Keyword arguments to pass only to the deprecated classifier to maintain equivalence
_DEP_TRACED_DEFAULTS = dict(
    variance_threshold=1, components=None, threshold_percentile=100, min_dims=None,
    pca_strategy='bottom', overlap_selection='parallel',
)


def _make_traced_pair(dep_module, kwargs):
    """Return (clf_new, clf_dep) for TRACED with given kwargs."""
    clf_new = TRACED(**kwargs)
    clf_dep = dep_module.TRACED(**kwargs, **_DEP_TRACED_DEFAULTS)
    return clf_new, clf_dep


TRACED_PARAM_CASES = [
    pytest.param({"norm": 1},                           id="norm=1"),
    pytest.param({"method": "overlap"},                  id="method=overlap"),
    pytest.param({"method": "outside"},                  id="method=outside"),
    pytest.param({"r": 1.0},                             id="r=1.0"),
    pytest.param({"r": 2.0},                             id="r=2.0"),
    pytest.param({"N0": 1},                              id="N0=1"),
    pytest.param({"N0": 5},                              id="N0=5"),
    pytest.param({"delta": 1},                           id="delta=1"),
    pytest.param({"delta": 3},                           id="delta=3"),
    pytest.param({"alpha": 0.3},                         id="alpha=0.3"),
    pytest.param({"beta": 0.3},                          id="beta=0.3"),
    pytest.param({"distance_metric": "center"},          id="distance_metric=center"),
    pytest.param({"width_parameter": 2},                 id="width_parameter=2"),
    pytest.param({"reduce_dims": 1},                     id="reduce_dims=1"),
    pytest.param({"threshold": 10},                      id="threshold=10"),
    pytest.param({"threshold": 30},                      id="threshold=30"),
    pytest.param(
        {"alpha": 0.3, "beta": 0.3, "reduce_dims": 1, "method": "overlap-outside"},
        id="combined_adaptive",
    ),
    pytest.param(
        {"N0": 1, "delta": 1, "distance_metric": "center", "width_parameter": 2},
        id="combined_geometry",
    ),
]


class TestTRACEDHyperparams:

    @pytest.mark.parametrize("kwargs", TRACED_PARAM_CASES)
    def test_iris_predictions_match_deprecated(self, iris_data, dep_module, kwargs):
        X_train, X_test, y_train = iris_data
        clf_new, clf_dep = _make_traced_pair(dep_module, kwargs)
        clf_new.fit(X_train, y_train)
        clf_dep.fit(X_train, y_train)
        _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", TRACED_PARAM_CASES)
    def test_iris_neuron_values_match_deprecated(self, iris_data, dep_module, kwargs):
        X_train, _, y_train = iris_data
        clf_new, clf_dep = _make_traced_pair(dep_module, kwargs)
        clf_new.fit(X_train, y_train)
        clf_dep.fit(X_train, y_train)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'variance', 'width', 'displacement', 'expansion'],
        )

    @pytest.mark.parametrize("kwargs", TRACED_PARAM_CASES)
    def test_iris_chunked_matches_deprecated(self, iris_chunks, dep_module, kwargs):
        chunks, X_test, classes = iris_chunks
        clf_new, clf_dep = _make_traced_pair(dep_module, kwargs)
        for Xc, yc in chunks:
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_preds_match(clf_new, clf_dep, X_test)

    @pytest.mark.parametrize("kwargs", TRACED_PARAM_CASES)
    def test_iris_accuracy_reasonable(self, iris_data, kwargs):
        """Classifier reaches at least 50% accuracy — sanity check for each config."""
        X_train, X_test, y_train = iris_data
        y_test = load_iris(return_X_y=True)[1][120:]
        clf = TRACED(**kwargs)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        assert acc >= 0.5, f"accuracy {acc:.3f} too low for kwargs={kwargs}"
