# Changelog

All notable changes to `spdal` are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semantic
versioning (pre-1.0: minor bumps may change default behaviour).

## [0.3.0]

### Fixed (alters TRACED results vs 0.2.0)
- `TRACED` now rebuilds its ellipsoid shape matrix as `Pᵀ D P` (was `P D Pᵀ`) in
  `merge_neuron` and `update_parameter`, matching the row-major eigenvector
  convention of `compute_sorted_eigencomponent`. Neuron counts can shift slightly;
  accuracy is largely unchanged. The paper code (`deprecated/spdal.py`) uses the
  old form, so `TRACED` no longer reproduces it. Other classifiers do not use
  this reconstruction.

### Changed
- `spdal[dev]` now installs `matplotlib`, which the test suite needs:
  `deprecated/spdal.py` imports `pyplot` at module level.

## [0.2.0]

### Changed (default behaviour — may alter results vs 0.1.1)
- **Eigendecomposition now uses the symmetric solver `numpy.linalg.eigh` by
  default** (was the general `numpy.linalg.eig`). Covariance matrices are
  symmetric positive semi-definite by construction, so `eigh` is the correct,
  faster and more robust choice: it returns real eigenvalues and orthonormal
  eigenvectors and does not fail to converge on ill-conditioned input. The
  inputs are symmetrised (`(S + Sᵀ)/2`) before decomposition. For full-rank,
  well-conditioned (e.g. standardized) data the two solvers give identical
  predictions; results can differ on rank-deficient / degenerate covariances.
- **`LRHE` default `alpha` changed `0.5 → 0.99`** to match the paper's
  recommended value (gradual shrinking; LRHE paper p.11). Calls to `LRHE()`
  with no arguments now use `alpha=0.99`.

### Added
- `eig_solver` attribute on the hyperellipsoid base class (default `'eigh'`).
  Set to `'eig'` to reproduce the original paper-code eigendecomposition
  bit-for-bit (used by the regression/parity tests against `deprecated/spdal.py`).
- Test coverage for the eigensolver: direct unit tests of
  `compute_sorted_eigencomponent` (real/descending eigenvalues, orthonormal
  eigenvectors, spectral reconstruction, symmetrisation, eigenvalue flooring)
  and default-path property tests (orthonormality, symmetric-PSD covariances,
  accuracy, determinism, rank-deficient robustness, eig/eigh agreement on
  well-conditioned data).

## [0.1.1]
- Initial published release: LRHE, VEBF, SCIL, SHEF, D4, TRACED online
  hyperellipsoid classifiers.
