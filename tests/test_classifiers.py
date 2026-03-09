"""
Regression tests for all spdal classifiers.

Baseline predictions were captured from the original monolithic src/spdal.py
using make_classification(n_samples=300, random_state=42) for binary tasks
and load_iris() for multiclass tasks. Tests verify that refactoring has not
changed classification outputs or neuron counts.

Chunked partial_fit baselines were captured from deprecated/spdal.py using
4 chunks of 50 samples for binary classifiers, and 2 chunks of 50 samples
for multiclass classifiers.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification

from spdal import LRHE, VEBF, SCIL, SHEF, PPVH, TRACED


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(n_samples=300, random_state=42)
    return X[:200], X[200:], y[:200]


@pytest.fixture(scope="module")
def multiclass_data():
    X, y = load_iris(return_X_y=True)
    return X[:100], X[100:], y[:100], np.unique(y)


@pytest.fixture(scope="module")
def binary_chunks():
    """4 chunks of 50 samples each from the binary training set."""
    X, y = make_classification(n_samples=300, random_state=42)
    X_train, X_test, y_train = X[:200], X[200:], y[:200]
    chunks = [(X_train[i:i+50], y_train[i:i+50]) for i in range(0, 200, 50)]
    return chunks, X_test


@pytest.fixture(scope="module")
def multiclass_chunks():
    """2 chunks of 50 samples each from the multiclass training set."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train = X[:100], X[100:], y[:100]
    classes = np.unique(y)
    chunks = [(X_train[i:i+50], y_train[i:i+50]) for i in range(0, 100, 50)]
    return chunks, X_test, classes


# ---------------------------------------------------------------------------
# Expected baselines (captured from original src/spdal.py)
# ---------------------------------------------------------------------------

LRHE_EXPECTED_N_NEURONS = 3
LRHE_EXPECTED_PREDS = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]

VEBF_EXPECTED_N_NEURONS = 2
VEBF_EXPECTED_PREDS = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]

SCIL_EXPECTED_N_NEURONS = 2
SCIL_EXPECTED_PREDS = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]

SHEF_EXPECTED_N_NEURONS = 2
SHEF_EXPECTED_PREDS = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]

PPVH_EXPECTED_N_NEURONS = 2
PPVH_EXPECTED_PREDS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

TRACED_EXPECTED_N_NEURONS = 5
TRACED_EXPECTED_PREDS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Chunked partial_fit baselines (captured from deprecated/spdal.py)
# Binary: 4 chunks of 50 samples; Multiclass: 2 chunks of 50 samples
LRHE_CHUNK_N_NEURONS = 5
LRHE_CHUNK_PREDS = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]

VEBF_CHUNK_N_NEURONS = 2
VEBF_CHUNK_PREDS = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1]

SCIL_CHUNK_N_NEURONS = 2
SCIL_CHUNK_PREDS = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]

SHEF_CHUNK_N_NEURONS = 2
SHEF_CHUNK_PREDS = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]

PPVH_CHUNK_N_NEURONS = 2
PPVH_CHUNK_PREDS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

TRACED_CHUNK_N_NEURONS = 5
TRACED_CHUNK_PREDS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLRHE:
    def test_neuron_count(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = LRHE()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert clf.neuron_list.shape[0] == LRHE_EXPECTED_N_NEURONS

    def test_predictions(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = LRHE()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert list(preds) == LRHE_EXPECTED_PREDS

    def test_neuron_columns(self, binary_data):
        X_train, _, y_train = binary_data
        clf = LRHE()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        for col in ['y', 'center', 'cov', 'eig_component', 'width', 'n']:
            assert col in clf.neuron_list.columns

    def test_sklearn_interface(self, binary_data):
        X_train, _, y_train = binary_data
        clf = LRHE()
        assert hasattr(clf, 'partial_fit')
        assert hasattr(clf, 'predict')
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert hasattr(clf, 'classes_')


class TestVEBF:
    def test_neuron_count(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = VEBF()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert clf.neuron_list.shape[0] == VEBF_EXPECTED_N_NEURONS

    def test_predictions(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = VEBF()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert list(preds) == VEBF_EXPECTED_PREDS

    def test_neuron_columns(self, binary_data):
        X_train, _, y_train = binary_data
        clf = VEBF()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        for col in ['y', 'center', 'cov', 'eig_component', 'width', 'n']:
            assert col in clf.neuron_list.columns


class TestSCIL:
    def test_neuron_count(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = SCIL()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert clf.neuron_list.shape[0] == SCIL_EXPECTED_N_NEURONS

    def test_predictions(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = SCIL()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert list(preds) == SCIL_EXPECTED_PREDS

    def test_neuron_columns(self, binary_data):
        X_train, _, y_train = binary_data
        clf = SCIL()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        for col in ['y', 'center', 'cov', 'eig_component', 'width', 'n', 'variance']:
            assert col in clf.neuron_list.columns


class TestSHEF:
    def test_neuron_count(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = SHEF()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert clf.neuron_list.shape[0] == SHEF_EXPECTED_N_NEURONS

    def test_predictions(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = SHEF()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        preds = clf.predict(X_test)
        assert list(preds) == SHEF_EXPECTED_PREDS

    def test_neuron_columns(self, binary_data):
        X_train, _, y_train = binary_data
        clf = SHEF()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        for col in ['y', 'center', 'cov', 'n']:
            assert col in clf.neuron_list.columns


class TestPPVH:
    def test_neuron_count(self, multiclass_data):
        X_train, X_test, y_train, classes = multiclass_data
        clf = PPVH()
        clf.partial_fit(X_train, y_train, classes=classes)
        assert clf.neuron_list.shape[0] == PPVH_EXPECTED_N_NEURONS

    def test_predictions(self, multiclass_data):
        X_train, X_test, y_train, classes = multiclass_data
        clf = PPVH()
        clf.partial_fit(X_train, y_train, classes=classes)
        preds = clf.predict(X_test)
        assert list(preds) == PPVH_EXPECTED_PREDS

    def test_neuron_columns(self, multiclass_data):
        X_train, _, y_train, classes = multiclass_data
        clf = PPVH()
        clf.partial_fit(X_train, y_train, classes=classes)
        for col in ['y', 'center', 'cov', 'eig_component', 'width', 'n', 'variance']:
            assert col in clf.neuron_list.columns


class TestTRACED:
    def test_neuron_count(self, multiclass_data):
        X_train, X_test, y_train, classes = multiclass_data
        clf = TRACED()
        clf.partial_fit(X_train, y_train, classes=classes)
        assert clf.neuron_list.shape[0] == TRACED_EXPECTED_N_NEURONS

    def test_predictions(self, multiclass_data):
        X_train, X_test, y_train, classes = multiclass_data
        clf = TRACED()
        clf.partial_fit(X_train, y_train, classes=classes)
        preds = clf.predict(X_test)
        assert list(preds) == TRACED_EXPECTED_PREDS

    def test_neuron_columns(self, multiclass_data):
        X_train, _, y_train, classes = multiclass_data
        clf = TRACED()
        clf.partial_fit(X_train, y_train, classes=classes)
        for col in ['y', 'center', 'cov', 'eig_component', 'width', 'n', 'variance', 'displacement', 'expansion']:
            assert col in clf.neuron_list.columns


# ---------------------------------------------------------------------------
# Chunked partial_fit tests (4 chunks of 50 for binary, 2 chunks of 50 for multiclass)
# ---------------------------------------------------------------------------

class TestLRHEChunked:
    def test_neuron_count(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = LRHE()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert clf.neuron_list.shape[0] == LRHE_CHUNK_N_NEURONS

    def test_predictions(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = LRHE()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert list(clf.predict(X_test)) == LRHE_CHUNK_PREDS


class TestVEBFChunked:
    def test_neuron_count(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = VEBF()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert clf.neuron_list.shape[0] == VEBF_CHUNK_N_NEURONS

    def test_predictions(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = VEBF()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert list(clf.predict(X_test)) == VEBF_CHUNK_PREDS


class TestSCILChunked:
    def test_neuron_count(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = SCIL()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert clf.neuron_list.shape[0] == SCIL_CHUNK_N_NEURONS

    def test_predictions(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = SCIL()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert list(clf.predict(X_test)) == SCIL_CHUNK_PREDS


class TestSHEFChunked:
    def test_neuron_count(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = SHEF()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert clf.neuron_list.shape[0] == SHEF_CHUNK_N_NEURONS

    def test_predictions(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = SHEF()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert list(clf.predict(X_test)) == SHEF_CHUNK_PREDS


class TestPPVHChunked:
    def test_neuron_count(self, multiclass_chunks):
        chunks, X_test, classes = multiclass_chunks
        clf = PPVH()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=classes)
        assert clf.neuron_list.shape[0] == PPVH_CHUNK_N_NEURONS

    def test_predictions(self, multiclass_chunks):
        chunks, X_test, classes = multiclass_chunks
        clf = PPVH()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=classes)
        assert list(clf.predict(X_test)) == PPVH_CHUNK_PREDS


class TestTRACEDChunked:
    def test_neuron_count(self, multiclass_chunks):
        chunks, X_test, classes = multiclass_chunks
        clf = TRACED()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=classes)
        assert clf.neuron_list.shape[0] == TRACED_CHUNK_N_NEURONS

    def test_predictions(self, multiclass_chunks):
        chunks, X_test, classes = multiclass_chunks
        clf = TRACED()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=classes)
        assert list(clf.predict(X_test)) == TRACED_CHUNK_PREDS
