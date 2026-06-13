"""Shared pytest configuration for the spdal test suite.

The regression/parity tests in this directory verify that the refactored
package reproduces the original paper code (deprecated/spdal.py), which uses
the general eigensolver ``numpy.linalg.eig``. The package itself now defaults
to the symmetric solver ``eigh`` (correct + robust; see
HyperellipsoidBaseClassifier.compute_sorted_eigencomponent), which produces
different — both valid — eigenvectors on rank-deficient / degenerate data.

To keep these "reproduce the paper code" tests meaningful and deterministic,
this autouse fixture pins ``eig_solver='eig'`` for the whole suite. Tests that
exercise the default ``eigh`` path (test_eigh_default.py) set it back to
'eigh' explicitly inside their own bodies.
"""
import pytest

from spdal._base import HyperellipsoidBaseClassifier


@pytest.fixture(autouse=True)
def _use_legacy_eig_solver():
    old = HyperellipsoidBaseClassifier.eig_solver
    HyperellipsoidBaseClassifier.eig_solver = 'eig'
    yield
    HyperellipsoidBaseClassifier.eig_solver = old
