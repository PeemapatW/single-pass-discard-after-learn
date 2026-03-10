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
def iris_data():
    """Full Iris: train 120, test 30."""
    X, y = load_iris(return_X_y=True)
    return X[:120], X[120:], y[:120]


@pytest.fixture(scope="module")
def digits_data():
    """Digits: train 400, test 100."""
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    return X[:400], X[400:500], y[:400]


@pytest.fixture(scope="module")
def multiclass_chunks():
    """2 chunks of 50 samples each from the multiclass training set."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train = X[:100], X[100:], y[:100]
    classes = np.unique(y)
    chunks = [(X_train[i:i+50], y_train[i:i+50]) for i in range(0, 100, 50)]
    return chunks, X_test, classes


@pytest.fixture(scope="module")
def iris_chunks():
    """4 chunks of 30 samples each from the full Iris training set (120 train, 30 test)."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train = X[:120], X[120:], y[:120]
    classes = np.unique(y)
    chunks = [(X_train[i:i+30], y_train[i:i+30]) for i in range(0, 120, 30)]
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

D4_EXPECTED_N_NEURONS = 2
D4_EXPECTED_PREDS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

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

D4_CHUNK_N_NEURONS = 2
D4_CHUNK_PREDS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

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
        assert len(clf.neuron_list) == LRHE_EXPECTED_N_NEURONS

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
            assert col in clf.neuron_list[0]


class TestVEBF:
    def test_neuron_count(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = VEBF()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert len(clf.neuron_list) == VEBF_EXPECTED_N_NEURONS

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
            assert col in clf.neuron_list[0]


class TestSCIL:
    def test_neuron_count(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = SCIL()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert len(clf.neuron_list) == SCIL_EXPECTED_N_NEURONS

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
            assert col in clf.neuron_list[0]


class TestSHEF:
    def test_neuron_count(self, binary_data):
        X_train, X_test, y_train = binary_data
        clf = SHEF()
        clf.partial_fit(X_train, y_train, classes=[0, 1])
        assert len(clf.neuron_list) == SHEF_EXPECTED_N_NEURONS

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
            assert col in clf.neuron_list[0]


class TestD4:
    def test_neuron_count(self, multiclass_data):
        X_train, X_test, y_train, classes = multiclass_data
        clf = D4()
        clf.partial_fit(X_train, y_train, classes=classes)
        assert len(clf.neuron_list) == D4_EXPECTED_N_NEURONS

    def test_predictions(self, multiclass_data):
        X_train, X_test, y_train, classes = multiclass_data
        clf = D4()
        clf.partial_fit(X_train, y_train, classes=classes)
        preds = clf.predict(X_test)
        assert list(preds) == D4_EXPECTED_PREDS

    def test_neuron_columns(self, multiclass_data):
        X_train, _, y_train, classes = multiclass_data
        clf = D4()
        clf.partial_fit(X_train, y_train, classes=classes)
        for col in ['y', 'center', 'cov', 'eig_component', 'width', 'n', 'variance']:
            assert col in clf.neuron_list[0]


class TestTRACED:
    def test_neuron_count(self, multiclass_data):
        X_train, X_test, y_train, classes = multiclass_data
        clf = TRACED()
        clf.partial_fit(X_train, y_train, classes=classes)
        assert len(clf.neuron_list) == TRACED_EXPECTED_N_NEURONS

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
            assert col in clf.neuron_list[0]


# ---------------------------------------------------------------------------
# Chunked partial_fit tests (4 chunks of 50 for binary, 2 chunks of 50 for multiclass)
# ---------------------------------------------------------------------------

class TestLRHEChunked:
    def test_neuron_count(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = LRHE()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert len(clf.neuron_list) == LRHE_CHUNK_N_NEURONS

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
        assert len(clf.neuron_list) == VEBF_CHUNK_N_NEURONS

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
        assert len(clf.neuron_list) == SCIL_CHUNK_N_NEURONS

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
        assert len(clf.neuron_list) == SHEF_CHUNK_N_NEURONS

    def test_predictions(self, binary_chunks):
        chunks, X_test = binary_chunks
        clf = SHEF()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=[0, 1])
        assert list(clf.predict(X_test)) == SHEF_CHUNK_PREDS


class TestD4Chunked:
    def test_neuron_count(self, multiclass_chunks):
        chunks, X_test, classes = multiclass_chunks
        clf = D4()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=classes)
        assert len(clf.neuron_list) == D4_CHUNK_N_NEURONS

    def test_predictions(self, multiclass_chunks):
        chunks, X_test, classes = multiclass_chunks
        clf = D4()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=classes)
        assert list(clf.predict(X_test)) == D4_CHUNK_PREDS


class TestTRACEDChunked:
    def test_neuron_count(self, multiclass_chunks):
        chunks, X_test, classes = multiclass_chunks
        clf = TRACED()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=classes)
        assert len(clf.neuron_list) == TRACED_CHUNK_N_NEURONS

    def test_predictions(self, multiclass_chunks):
        chunks, X_test, classes = multiclass_chunks
        clf = TRACED()
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=classes)
        assert list(clf.predict(X_test)) == TRACED_CHUNK_PREDS


# ---------------------------------------------------------------------------
# Iris (multiclass, train 120, test 30) baselines
# ---------------------------------------------------------------------------

IRIS_LRHE_N_NEURONS = 3
IRIS_LRHE_PREDS = [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

IRIS_VEBF_N_NEURONS = 3
IRIS_VEBF_PREDS = [2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

IRIS_SCIL_N_NEURONS = 3
IRIS_SCIL_PREDS = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

IRIS_SHEF_N_NEURONS = 8
IRIS_SHEF_PREDS = [2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

IRIS_D4_N_NEURONS = 3
IRIS_D4_PREDS = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

IRIS_TRACED_N_NEURONS = 7
IRIS_TRACED_PREDS = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


# ---------------------------------------------------------------------------
# Digits (train 400, test 100) baselines
# ---------------------------------------------------------------------------

DIGITS_LRHE_N_NEURONS = 18
DIGITS_LRHE_PREDS = [1, 1, 6, 7, 1, 3, 0, 7, 2, 3, 1, 1, 6, 7, 6, 3, 0, 3, 3, 1, 6, 3, 0, 3, 3, 3, 3, 7, 2, 7, 7, 3, 7, 2, 0, 0, 2, 2, 7, 3, 2, 0, 2, 2, 6, 3, 3, 7, 3, 3, 1, 6, 6, 6, 6, 3, 2, 0, 0, 3, 3, 2, 3, 2, 0, 0, 2, 7, 6, 3, 2, 6, 7, 6, 6, 3, 2, 3, 3, 2, 7, 6, 6, 1, 3, 2, 1, 0, 1, 3, 6, 2, 1, 2, 7, 1, 6, 6, 7, 2]

DIGITS_VEBF_N_NEURONS = 10
DIGITS_VEBF_PREDS = [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 9, 5, 5, 8, 9, 0, 9, 8, 9, 8, 4, 1, 7, 7, 3, 5, 1, 0, 0, 2, 2, 7, 9, 2, 0, 2, 2, 6, 3, 3, 7, 3, 9, 4, 6, 6, 6, 4, 9, 1, 9, 0, 9, 5, 2, 8, 2, 0, 0, 1, 7, 6, 3, 2, 1, 7, 4, 6, 3, 1, 3, 9, 1, 9, 6, 8, 4, 3, 1, 4, 0, 5, 3, 6, 9, 8, 1, 7, 5, 4, 4, 9, 2]

DIGITS_SCIL_N_NEURONS = 10
DIGITS_SCIL_PREDS = [8, 8, 5, 7, 8, 8, 0, 1, 2, 1, 1, 1, 8, 7, 8, 9, 0, 1, 5, 8, 8, 1, 0, 8, 8, 9, 8, 4, 1, 1, 7, 3, 8, 1, 0, 0, 8, 8, 7, 8, 2, 0, 8, 3, 6, 8, 1, 7, 8, 1, 4, 6, 1, 6, 4, 8, 1, 1, 0, 8, 8, 8, 8, 8, 0, 0, 1, 8, 6, 8, 8, 1, 1, 8, 6, 8, 1, 8, 1, 1, 2, 8, 8, 4, 8, 1, 1, 0, 8, 8, 6, 1, 8, 1, 7, 8, 1, 1, 1, 8]

DIGITS_SHEF_N_NEURONS = 10
DIGITS_SHEF_PREDS = [4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 9, 9, 0, 9, 5, 5, 6, 3, 0, 9, 8, 9, 8, 4, 1, 7, 1, 3, 5, 1, 0, 0, 2, 2, 7, 3, 3, 0, 2, 1, 6, 3, 3, 7, 3, 3, 4, 6, 6, 6, 4, 9, 1, 5, 0, 9, 5, 1, 1, 2, 0, 0, 1, 7, 6, 3, 2, 1, 3, 4, 6, 3, 1, 3, 9, 1, 9, 6, 8, 4, 3, 1, 4, 0, 5, 3, 6, 9, 5, 1, 7, 5, 4, 4, 7, 2]

DIGITS_D4_N_NEURONS = 10
DIGITS_D4_PREDS = [4.0, 3.0, 4.0, 3.0, 8.0, 3.0, 0.0, 4.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0, 8.0, 3.0, 4.0, 3.0, 8.0, 3.0, 8.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 4.0, 0.0, 0.0, 3.0, 3.0, 3.0, 8.0, 3.0, 0.0, 3.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0, 4.0, 6.0, 4.0, 3.0, 4.0, 3.0, 1.0, 3.0, 0.0, 3.0, 5.0, 3.0, 8.0, 3.0, 0.0, 0.0, 4.0, 3.0, 6.0, 3.0, 3.0, 1.0, 3.0, 4.0, 8.0, 3.0, 4.0, 3.0, 3.0, 1.0, 3.0, 8.0, 8.0, 4.0, 3.0, 4.0, 4.0, 4.0, 8.0, 3.0, 6.0, 3.0, 4.0, 1.0, 3.0, 5.0, 4.0, 4.0, 3.0, 3.0]

DIGITS_TRACED_N_NEURONS = 10
DIGITS_TRACED_PREDS = [4, 5, 6, 7, 8, 9, 0, 1, 2, 9, 4, 5, 6, 5, 8, 9, 0, 9, 5, 5, 0, 9, 0, 9, 8, 9, 8, 4, 1, 7, 7, 3, 5, 1, 0, 0, 2, 2, 7, 9, 2, 0, 2, 3, 6, 3, 9, 5, 9, 9, 4, 6, 6, 6, 4, 9, 1, 9, 0, 9, 5, 2, 8, 2, 0, 0, 1, 5, 6, 9, 2, 1, 7, 4, 6, 3, 1, 3, 9, 1, 9, 6, 8, 4, 3, 1, 4, 0, 5, 9, 6, 9, 8, 1, 7, 5, 1, 4, 5, 2]


# ---------------------------------------------------------------------------
# Iris tests (all classifiers)
# ---------------------------------------------------------------------------

class TestLRHEIris:
    def test_neuron_count(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = LRHE()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == IRIS_LRHE_N_NEURONS

    def test_predictions(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = LRHE()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == IRIS_LRHE_PREDS


class TestVEBFIris:
    def test_neuron_count(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = VEBF()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == IRIS_VEBF_N_NEURONS

    def test_predictions(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = VEBF()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == IRIS_VEBF_PREDS


class TestSCILIris:
    def test_neuron_count(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = SCIL()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == IRIS_SCIL_N_NEURONS

    def test_predictions(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = SCIL()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == IRIS_SCIL_PREDS


class TestSHEFIris:
    def test_neuron_count(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = SHEF()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == IRIS_SHEF_N_NEURONS

    def test_predictions(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = SHEF()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == IRIS_SHEF_PREDS


class TestD4Iris:
    def test_neuron_count(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = D4()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == IRIS_D4_N_NEURONS

    def test_predictions(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = D4()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == IRIS_D4_PREDS


class TestTRACEDIris:
    def test_neuron_count(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = TRACED()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == IRIS_TRACED_N_NEURONS

    def test_predictions(self, iris_data):
        X_train, X_test, y_train = iris_data
        clf = TRACED()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == IRIS_TRACED_PREDS


# ---------------------------------------------------------------------------
# Digits tests (all classifiers)
# ---------------------------------------------------------------------------

class TestLRHEDigits:
    def test_neuron_count(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = LRHE()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == DIGITS_LRHE_N_NEURONS

    def test_predictions(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = LRHE()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == DIGITS_LRHE_PREDS


class TestVEBFDigits:
    def test_neuron_count(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = VEBF()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == DIGITS_VEBF_N_NEURONS

    def test_predictions(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = VEBF()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == DIGITS_VEBF_PREDS


class TestSCILDigits:
    def test_neuron_count(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = SCIL()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == DIGITS_SCIL_N_NEURONS

    def test_predictions(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = SCIL()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == DIGITS_SCIL_PREDS


class TestSHEFDigits:
    def test_neuron_count(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = SHEF()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == DIGITS_SHEF_N_NEURONS

    def test_predictions(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = SHEF()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == DIGITS_SHEF_PREDS


class TestD4Digits:
    def test_neuron_count(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = D4()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == DIGITS_D4_N_NEURONS

    def test_predictions(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = D4()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == DIGITS_D4_PREDS


class TestTRACEDDigits:
    def test_neuron_count(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = TRACED()
        clf.fit(X_train, y_train)
        assert len(clf.neuron_list) == DIGITS_TRACED_N_NEURONS

    def test_predictions(self, digits_data):
        X_train, X_test, y_train = digits_data
        clf = TRACED()
        clf.fit(X_train, y_train)
        assert list(clf.predict(X_test)) == DIGITS_TRACED_PREDS


# ---------------------------------------------------------------------------
# Numeric parity tests: list-of-dicts vs deprecated DataFrame
# ---------------------------------------------------------------------------
# These tests verify that converting neuron_list storage from DataFrame to
# list-of-dicts did not change any internal neuron values (center, cov, width,
# eig_component, variance, n).

@pytest.fixture(scope="module")
def dep_module():
    """Load deprecated DataFrame-based classifiers from deprecated/spdal.py."""
    repo_root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        "spdal_deprecated", repo_root / "deprecated" / "spdal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sorted_neurons(neuron_list):
    """Normalise to list of dicts, sorted by (y, norm of center) for stable comparison."""
    if hasattr(neuron_list, 'iterrows'):  # DataFrame
        neurons = [row.to_dict() for _, row in neuron_list.iterrows()]
    else:
        neurons = list(neuron_list)
    return sorted(neurons, key=lambda n: (n['y'], float(np.linalg.norm(n['center']))))


def _assert_neurons_close(new_neurons, dep_neurons, fields, rtol=1e-10):
    """Assert that each neuron matches between new (list) and deprecated (DataFrame) versions."""
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


class TestLRHENumerics:
    """LRHE: list-of-dicts neuron values must match deprecated DataFrame values."""

    def test_neuron_values_binary(self, binary_data, dep_module):
        X_train, _, y_train = binary_data
        clf_new = LRHE()
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep = dep_module.LRHE()
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component'],
        )

    def test_neuron_values_iris(self, iris_data, dep_module):
        X_train, _, y_train = iris_data
        clf_new = LRHE()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.LRHE()
        clf_dep.fit(X_train, y_train)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component'],
        )

    def test_accuracy_binary(self, binary_data, dep_module):
        X_train, X_test, y_train = binary_data
        clf_new = LRHE()
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep = dep_module.LRHE()
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        preds_new = clf_new.predict(X_test)
        preds_dep = clf_dep.predict(X_test)
        np.testing.assert_array_equal(preds_new, preds_dep)

    def test_accuracy_iris(self, iris_data, dep_module):
        X_train, X_test, y_train = iris_data
        clf_new = LRHE()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.LRHE()
        clf_dep.fit(X_train, y_train)
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))


class TestVEBFNumerics:
    """VEBF: list-of-dicts neuron values must match deprecated DataFrame values."""

    def test_neuron_values_binary(self, binary_data, dep_module):
        X_train, _, y_train = binary_data
        clf_new = VEBF()
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep = dep_module.VEBF()
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component'],
        )

    def test_neuron_values_iris(self, iris_data, dep_module):
        X_train, _, y_train = iris_data
        clf_new = VEBF()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.VEBF()
        clf_dep.fit(X_train, y_train)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component'],
        )

    def test_accuracy_binary(self, binary_data, dep_module):
        X_train, X_test, y_train = binary_data
        clf_new = VEBF()
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep = dep_module.VEBF()
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))

    def test_accuracy_iris(self, iris_data, dep_module):
        X_train, X_test, y_train = iris_data
        clf_new = VEBF()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.VEBF()
        clf_dep.fit(X_train, y_train)
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))


class TestSCILNumerics:
    """SCIL: list-of-dicts neuron values must match deprecated DataFrame values."""

    def test_neuron_values_binary(self, binary_data, dep_module):
        X_train, _, y_train = binary_data
        clf_new = SCIL()
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep = dep_module.SCIL()
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component', 'variance'],
        )

    def test_neuron_values_iris(self, iris_data, dep_module):
        X_train, _, y_train = iris_data
        clf_new = SCIL()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.SCIL()
        clf_dep.fit(X_train, y_train)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component', 'variance'],
        )

    def test_accuracy_binary(self, binary_data, dep_module):
        X_train, X_test, y_train = binary_data
        clf_new = SCIL()
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep = dep_module.SCIL()
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))

    def test_accuracy_iris(self, iris_data, dep_module):
        X_train, X_test, y_train = iris_data
        clf_new = SCIL()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.SCIL()
        clf_dep.fit(X_train, y_train)
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))


class TestSHEFNumerics:
    """SHEF: list-of-dicts neuron values must match deprecated DataFrame values."""

    def test_neuron_values_binary(self, binary_data, dep_module):
        X_train, _, y_train = binary_data
        clf_new = SHEF()
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep = dep_module.SHEF()
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov'],
        )

    def test_neuron_values_iris(self, iris_data, dep_module):
        X_train, _, y_train = iris_data
        clf_new = SHEF()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.SHEF()
        clf_dep.fit(X_train, y_train)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov'],
        )

    def test_accuracy_binary(self, binary_data, dep_module):
        X_train, X_test, y_train = binary_data
        clf_new = SHEF()
        clf_new.partial_fit(X_train, y_train, classes=[0, 1])
        clf_dep = dep_module.SHEF()
        clf_dep.partial_fit(X_train, y_train, classes=[0, 1])
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))

    def test_accuracy_iris(self, iris_data, dep_module):
        X_train, X_test, y_train = iris_data
        clf_new = SHEF()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.SHEF()
        clf_dep.fit(X_train, y_train)
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))


# ---------------------------------------------------------------------------
# Chunk-by-chunk monitoring tests: track neuron params + predictions + metrics
# after every partial_fit call
# ---------------------------------------------------------------------------

def _assert_chunk_state(clf_new, clf_dep, chunk_idx, X_test, y_test, fields):
    """
    After fitting one chunk, assert that new and deprecated produce identical:
      - neuron count
      - neuron parameter values (fields)
      - predictions on X_test
      - accuracy score on X_test
    """
    new_neurons = _sorted_neurons(clf_new.neuron_list)
    dep_neurons = _sorted_neurons(clf_dep.neuron_list)

    assert len(new_neurons) == len(dep_neurons), (
        f"chunk {chunk_idx}: neuron count mismatch "
        f"new={len(new_neurons)} dep={len(dep_neurons)}"
    )

    _assert_neurons_close(new_neurons, dep_neurons, fields=fields, rtol=1e-10)

    preds_new = clf_new.predict(X_test)
    preds_dep = clf_dep.predict(X_test)
    np.testing.assert_array_equal(
        preds_new, preds_dep,
        err_msg=f"chunk {chunk_idx}: predictions differ",
    )

    acc_new = accuracy_score(y_test, preds_new)
    acc_dep = accuracy_score(y_test, preds_dep)
    assert acc_new == acc_dep, (
        f"chunk {chunk_idx}: accuracy mismatch new={acc_new:.4f} dep={acc_dep:.4f}"
    )


class TestLRHEChunkedMonitor:
    """Track LRHE neuron params, predictions, and accuracy after every chunk."""

    def test_binary_chunks(self, binary_chunks, dep_module):
        chunks, X_test = binary_chunks
        y_test = make_classification(n_samples=300, random_state=42)[1][200:]
        clf_new = LRHE()
        clf_dep = dep_module.LRHE()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=[0, 1])
            clf_dep.partial_fit(Xc, yc, classes=[0, 1])
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'width', 'eig_component'])

    def test_iris_chunks(self, iris_chunks, dep_module):
        chunks, X_test, classes = iris_chunks
        y_test = load_iris(return_X_y=True)[1][120:]
        clf_new = LRHE()
        clf_dep = dep_module.LRHE()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'width', 'eig_component'])


class TestVEBFChunkedMonitor:
    """Track VEBF neuron params, predictions, and accuracy after every chunk."""

    def test_binary_chunks(self, binary_chunks, dep_module):
        chunks, X_test = binary_chunks
        y_test = make_classification(n_samples=300, random_state=42)[1][200:]
        clf_new = VEBF()
        clf_dep = dep_module.VEBF()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=[0, 1])
            clf_dep.partial_fit(Xc, yc, classes=[0, 1])
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'width', 'eig_component'])

    def test_iris_chunks(self, iris_chunks, dep_module):
        chunks, X_test, classes = iris_chunks
        y_test = load_iris(return_X_y=True)[1][120:]
        clf_new = VEBF()
        clf_dep = dep_module.VEBF()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'width', 'eig_component'])


class TestSCILChunkedMonitor:
    """Track SCIL neuron params, predictions, and accuracy after every chunk."""

    def test_binary_chunks(self, binary_chunks, dep_module):
        chunks, X_test = binary_chunks
        y_test = make_classification(n_samples=300, random_state=42)[1][200:]
        clf_new = SCIL()
        clf_dep = dep_module.SCIL()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=[0, 1])
            clf_dep.partial_fit(Xc, yc, classes=[0, 1])
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'width', 'eig_component', 'variance'])

    def test_iris_chunks(self, iris_chunks, dep_module):
        chunks, X_test, classes = iris_chunks
        y_test = load_iris(return_X_y=True)[1][120:]
        clf_new = SCIL()
        clf_dep = dep_module.SCIL()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'width', 'eig_component', 'variance'])


class TestSHEFChunkedMonitor:
    """Track SHEF neuron params, predictions, and accuracy after every chunk."""

    def test_binary_chunks(self, binary_chunks, dep_module):
        chunks, X_test = binary_chunks
        y_test = make_classification(n_samples=300, random_state=42)[1][200:]
        clf_new = SHEF()
        clf_dep = dep_module.SHEF()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=[0, 1])
            clf_dep.partial_fit(Xc, yc, classes=[0, 1])
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov'])

    def test_iris_chunks(self, iris_chunks, dep_module):
        chunks, X_test, classes = iris_chunks
        y_test = load_iris(return_X_y=True)[1][120:]
        clf_new = SHEF()
        clf_dep = dep_module.SHEF()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov'])


class TestD4Numerics:
    """D4: list-of-dicts neuron values must match deprecated DataFrame values."""

    def test_neuron_values_multiclass(self, multiclass_data, dep_module):
        X_train, _, y_train, classes = multiclass_data
        clf_new = D4()
        clf_new.partial_fit(X_train, y_train, classes=classes)
        clf_dep = dep_module.D4()
        clf_dep.partial_fit(X_train, y_train, classes=classes)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component', 'variance'],
        )

    def test_neuron_values_iris(self, iris_data, dep_module):
        X_train, _, y_train = iris_data
        clf_new = D4()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.D4()
        clf_dep.fit(X_train, y_train)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'width', 'eig_component', 'variance'],
        )

    def test_accuracy_multiclass(self, multiclass_data, dep_module):
        X_train, X_test, y_train, classes = multiclass_data
        clf_new = D4()
        clf_new.partial_fit(X_train, y_train, classes=classes)
        clf_dep = dep_module.D4()
        clf_dep.partial_fit(X_train, y_train, classes=classes)
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))

    def test_accuracy_iris(self, iris_data, dep_module):
        X_train, X_test, y_train = iris_data
        clf_new = D4()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.D4()
        clf_dep.fit(X_train, y_train)
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))


class TestD4ChunkedMonitor:
    """Track D4 neuron params, predictions, and accuracy after every chunk."""

    def test_multiclass_chunks(self, multiclass_chunks, dep_module):
        chunks, X_test, classes = multiclass_chunks
        y_test = load_iris(return_X_y=True)[1][100:]
        clf_new = D4()
        clf_dep = dep_module.D4()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'width', 'eig_component', 'variance'])

    def test_iris_chunks(self, iris_chunks, dep_module):
        chunks, X_test, classes = iris_chunks
        y_test = load_iris(return_X_y=True)[1][120:]
        clf_new = D4()
        clf_dep = dep_module.D4()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'width', 'eig_component', 'variance'])


class TestTRACEDNumerics:
    """TRACED: list-of-dicts neuron values must match deprecated DataFrame values."""

    def test_neuron_values_multiclass(self, multiclass_data, dep_module):
        X_train, _, y_train, classes = multiclass_data
        clf_new = TRACED()
        clf_new.partial_fit(X_train, y_train, classes=classes)
        clf_dep = dep_module.TRACED()
        clf_dep.partial_fit(X_train, y_train, classes=classes)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'variance', 'width', 'displacement', 'expansion'],
        )

    def test_neuron_values_iris(self, iris_data, dep_module):
        X_train, _, y_train = iris_data
        clf_new = TRACED()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.TRACED()
        clf_dep.fit(X_train, y_train)
        _assert_neurons_close(
            _sorted_neurons(clf_new.neuron_list),
            _sorted_neurons(clf_dep.neuron_list),
            fields=['center', 'cov', 'variance', 'width', 'displacement', 'expansion'],
        )

    def test_accuracy_multiclass(self, multiclass_data, dep_module):
        X_train, X_test, y_train, classes = multiclass_data
        clf_new = TRACED()
        clf_new.partial_fit(X_train, y_train, classes=classes)
        clf_dep = dep_module.TRACED()
        clf_dep.partial_fit(X_train, y_train, classes=classes)
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))

    def test_accuracy_iris(self, iris_data, dep_module):
        X_train, X_test, y_train = iris_data
        clf_new = TRACED()
        clf_new.fit(X_train, y_train)
        clf_dep = dep_module.TRACED()
        clf_dep.fit(X_train, y_train)
        np.testing.assert_array_equal(clf_new.predict(X_test), clf_dep.predict(X_test))


class TestTRACEDChunkedMonitor:
    """Track TRACED neuron params, predictions, and accuracy after every chunk."""

    def test_multiclass_chunks(self, multiclass_chunks, dep_module):
        chunks, X_test, classes = multiclass_chunks
        y_test = load_iris(return_X_y=True)[1][100:]
        clf_new = TRACED()
        clf_dep = dep_module.TRACED()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'variance', 'width', 'displacement', 'expansion'])

    def test_iris_chunks(self, iris_chunks, dep_module):
        chunks, X_test, classes = iris_chunks
        y_test = load_iris(return_X_y=True)[1][120:]
        clf_new = TRACED()
        clf_dep = dep_module.TRACED()
        for i, (Xc, yc) in enumerate(chunks):
            clf_new.partial_fit(Xc, yc, classes=classes)
            clf_dep.partial_fit(Xc, yc, classes=classes)
            _assert_chunk_state(clf_new, clf_dep, i, X_test, y_test,
                                 fields=['center', 'cov', 'variance', 'width', 'displacement', 'expansion'])
