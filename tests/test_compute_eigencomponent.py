"""Unit tests for HyperellipsoidBaseClassifier.compute_sorted_eigencomponent.

This is the one method the eig -> eigh migration changed, but until now it was
covered only indirectly (through full classifier fits). These tests pin its
contract directly for both solver paths:

  eigh (default): real eigenvalues, descending order, orthonormal eigenvectors,
                  exact spectral reconstruction on a PD matrix, (S+S^T)/2
                  symmetrization, non-positive eigenvalue flooring.
  eig  (legacy):  real, descending output (reproduces the paper code).

Plus the concrete advantage that motivated the switch: eigh returns real
eigenvalues on a rank-deficient covariance where the general solver np.linalg.eig
returns a complex dtype.

eig_solver is set per-instance (shadows the class attribute that conftest pins
to 'eig'), so each test controls its own solver regardless of the suite default.
"""
import numpy as np
from numpy import linalg as LA

from spdal import LRHE

# eig_c rows are eigenvectors; pca_var are eigenvalues, descending.
# Reconstruction: cov == eig_c.T @ diag(pca_var) @ eig_c.


def _eig(cov, solver):
    clf = LRHE()                 # supplies self.epsilon = 1e-10
    clf.eig_solver = solver      # instance attr overrides conftest's class pin
    return clf.compute_sorted_eigencomponent(np.asarray(cov, float))


def _pd_cov():
    """A well-conditioned symmetric positive-definite 3x3 (no flooring)."""
    return np.array([[4.0, 1.0, 0.2],
                     [1.0, 3.0, 0.5],
                     [0.2, 0.5, 2.0]])


def test_eigh_eigenvalues_real_and_descending():
    eig_c, pca_var = _eig(_pd_cov(), 'eigh')
    assert np.isrealobj(pca_var)
    assert np.all(np.diff(pca_var) <= 1e-12)      # non-increasing
    assert pca_var.min() > 0


def test_eigh_eigenvectors_orthonormal():
    eig_c, _ = _eig(_pd_cov(), 'eigh')
    np.testing.assert_allclose(eig_c @ eig_c.T, np.eye(eig_c.shape[0]), atol=1e-10)


def test_eigh_reconstructs_covariance():
    cov = _pd_cov()
    eig_c, pca_var = _eig(cov, 'eigh')            # PD -> no flooring
    recon = eig_c.T @ np.diag(pca_var) @ eig_c
    np.testing.assert_allclose(recon, cov, atol=1e-10)


def test_eigh_symmetrization_is_transpose_invariant():
    """(S + S^T)/2 makes the result independent of an asymmetric input's
    triangle: compute(M) must equal compute(M^T)."""
    M = _pd_cov().copy()
    M[0, 1] += 0.3                                # break symmetry (upper only)
    ec1, pv1 = _eig(M, 'eigh')
    ec2, pv2 = _eig(M.T, 'eigh')
    np.testing.assert_allclose(pv1, pv2, atol=1e-10)
    # eigenvectors equal up to per-axis sign
    np.testing.assert_allclose(np.abs(ec1 @ ec2.T), np.eye(ec1.shape[0]), atol=1e-8)


def test_eigh_floors_nonpositive_eigenvalues():
    """A covariance with a zero eigenvalue comes back strictly positive."""
    # build cov = V diag([5, 2, 0]) V^T with V orthonormal
    V, _ = LA.qr(np.array([[1.0, 0.3, 0.0],
                           [0.0, 1.0, 0.4],
                           [0.2, 0.0, 1.0]]))
    cov = V @ np.diag([5.0, 2.0, 0.0]) @ V.T
    _, pca_var = _eig(cov, 'eigh')
    assert pca_var.min() > 0                      # the 0 axis was floored


def test_eigh_returns_real_where_eig_returns_complex():
    """The migration's motivation: on a rank-deficient covariance the general
    solver yields a complex dtype; the symmetric solver stays real."""
    rng = np.random.RandomState(0)
    A = rng.rand(20, 3)
    cov = A @ A.T                                 # 20x20, rank 3
    assert np.iscomplexobj(LA.eig(cov)[0])        # documents eig's behaviour
    _, pca_var = _eig(cov, 'eigh')
    assert np.isrealobj(pca_var)
    assert np.all(np.isfinite(pca_var))


def test_eig_legacy_path_real_and_descending():
    """Legacy 'eig' path (reproduces deprecated/spdal.py) returns real,
    descending eigenvalues."""
    eig_c, pca_var = _eig(_pd_cov(), 'eig')
    assert np.isrealobj(pca_var) and np.isrealobj(eig_c)
    assert np.all(np.diff(pca_var) <= 1e-12)
