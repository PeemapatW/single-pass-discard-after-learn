"""Property/behaviour tests for the package's DEFAULT eigensolver (eigh).

The regression suites (test_classifiers / test_hyperparams / test_phishing)
pin eig_solver='eig' to reproduce the paper code (deprecated/spdal.py). This
module instead exercises the package default — the symmetric solver eigh — and
asserts the mathematical guarantees that motivate it, plus the sklearn-style
API contract and determinism. No exact baselines: eigh's eigenvector basis is
defined only up to sign / degenerate-subspace rotation, so we assert
properties (orthonormality, symmetry, PSD, accuracy range) rather than frozen
values.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from spdal import LRHE, VEBF, SCIL, SHEF, D4, TRACED
from spdal._base import HyperellipsoidBaseClassifier

CLASSIFIERS = [LRHE, VEBF, SCIL, SHEF, D4, TRACED]


@pytest.fixture(autouse=True)
def _force_default_eigh():
    """Override conftest's eig pin: this module tests the default eigh path."""
    HyperellipsoidBaseClassifier.eig_solver = 'eigh'
    yield
    HyperellipsoidBaseClassifier.eig_solver = 'eigh'


@pytest.fixture(scope="module")
def iris_scaled():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = X[:120], X[120:], y[:120], y[120:]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


@pytest.fixture(scope="module")
def digits_scaled():
    X, y = load_digits(return_X_y=True)
    sc = StandardScaler().fit(X[:1000])
    return sc.transform(X[:1000]), sc.transform(X[1000:]), y[:1000], y[1000:]


@pytest.mark.parametrize("Cls", CLASSIFIERS, ids=[c.__name__ for c in CLASSIFIERS])
def test_eig_component_orthonormal(Cls, iris_scaled):
    """eigh guarantees orthonormal eigenvectors: U U^T == I for square U."""
    Xtr, _, ytr, _ = iris_scaled
    clf = Cls(); clf.fit(Xtr, ytr)
    checked = 0
    for nrn in clf.neuron_list:
        if 'eig_component' not in nrn:    # e.g. SHEF stores no eigenbasis
            continue
        U = np.asarray(nrn['eig_component'])
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            continue                      # projection-reduced rows: skip
        np.testing.assert_allclose(U @ U.T, np.eye(U.shape[0]), atol=1e-8)
        checked += 1
    if not any('eig_component' in n for n in clf.neuron_list):
        pytest.skip(f"{Cls.__name__} stores no eig_component")


@pytest.mark.parametrize("Cls", CLASSIFIERS, ids=[c.__name__ for c in CLASSIFIERS])
def test_cov_symmetric_psd(Cls, iris_scaled):
    """Neuron covariances stay symmetric with (essentially) non-negative spectrum."""
    Xtr, _, ytr, _ = iris_scaled
    clf = Cls(); clf.fit(Xtr, ytr)
    for nrn in clf.neuron_list:
        cov = np.asarray(nrn['cov'])
        if cov.size == 0 or cov.ndim != 2:
            continue
        np.testing.assert_allclose(cov, cov.T, atol=1e-8)
        w = np.linalg.eigvalsh((cov + cov.T) / 2)
        assert w.min() >= -1e-6, f"{Cls.__name__}: negative eigenvalue {w.min()}"


@pytest.mark.parametrize("Cls", CLASSIFIERS, ids=[c.__name__ for c in CLASSIFIERS])
def test_accuracy_reasonable_iris(Cls, iris_scaled):
    """Default eigh path classifies standardized iris well above chance (1/3)."""
    Xtr, Xte, ytr, yte = iris_scaled
    clf = Cls(); clf.fit(Xtr, ytr)
    acc = (clf.predict(Xte) == yte).mean()
    assert acc >= 0.7, f"{Cls.__name__}: iris acc {acc:.3f} too low"


@pytest.mark.parametrize("Cls", CLASSIFIERS, ids=[c.__name__ for c in CLASSIFIERS])
def test_api_contract(Cls, iris_scaled):
    """partial_fit returns self, sets classes_, predict has the right shape."""
    Xtr, Xte, ytr, yte = iris_scaled
    clf = Cls()
    clf.partial_fit(Xtr, ytr, classes=np.unique(ytr))
    assert set(np.unique(ytr)).issubset(set(clf.classes_))
    preds = clf.predict(Xte)
    assert len(preds) == len(Xte)


@pytest.mark.parametrize("Cls", CLASSIFIERS, ids=[c.__name__ for c in CLASSIFIERS])
def test_determinism(Cls, iris_scaled):
    """Same data + same solver => identical predictions across runs."""
    Xtr, Xte, ytr, _ = iris_scaled
    c1 = Cls(); c1.fit(Xtr, ytr); p1 = c1.predict(Xte)
    c2 = Cls(); c2.fit(Xtr, ytr); p2 = c2.predict(Xte)
    np.testing.assert_array_equal(p1, p2)


@pytest.mark.parametrize("Cls", CLASSIFIERS, ids=[c.__name__ for c in CLASSIFIERS])
def test_rank_deficient_no_crash(Cls, digits_scaled):
    """High-dim / rank-deficient input (digits, d=64) must not raise."""
    Xtr, Xte, ytr, _ = digits_scaled
    clf = Cls(); clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    assert len(preds) == len(Xte)


@pytest.mark.parametrize("Cls", CLASSIFIERS, ids=[c.__name__ for c in CLASSIFIERS])
def test_eig_and_eigh_agree_on_well_conditioned_data(Cls):
    """Safety of the eig -> eigh swap: on full-rank, standardized data, with the
    package defaults, the two solvers give identical predictions (they diverge
    only on rank-deficient / degenerate covariances). A stratified, shuffled
    split keeps the covariances well-conditioned and representative."""
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0,
                                        stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    c_eig = Cls(); c_eig.eig_solver = 'eig'; c_eig.fit(Xtr, ytr)
    c_eigh = Cls(); c_eigh.eig_solver = 'eigh'; c_eigh.fit(Xtr, ytr)
    np.testing.assert_array_equal(c_eig.predict(Xte), c_eigh.predict(Xte))
