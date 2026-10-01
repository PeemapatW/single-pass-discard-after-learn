# Changelog

All notable changes to `spdal` are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semantic
versioning (pre-1.0: minor bumps may change default behaviour).

## [0.4.0]

This release brings several classifiers closer to the algorithms as described in their papers, and
adds two small numerical safeguards. Results for `VEBF`, `LRHE`, `SHEF` and `TRACED` may differ from
0.3.0; `SCIL` differs only in the rare cases covered by the new safeguard, and `D4` is unchanged.
To reproduce earlier numbers, 0.3.0 remains available (`pip install spdal==0.3.0`).

### Changed (may alter results vs 0.3.0)
- `VEBF`
  - A new neuron now starts from a zero covariance matrix, following Algorithm step 6 ('Else' b) of
    the paper; previously it started from the identity.
  - When testing whether a sample can be absorbed, the eigenvectors of the tentative covariance are now
    used (Algorithm steps 4-5), rather than the neuron's current ones.
  - Widths are updated once a neuron holds more than N0 = 2 samples (step 6e), in line with the paper's
    setting.
- `LRHE`
  - The same two adjustments to the absorption test and the N0 = 2 width update (Algorithm 1 steps
    12-15; p. 4, above Eq. 6).
  - When a neuron is created or updated, every other neuron of the same class that satisfies the
    merging condition is now merged into it (Algorithm 1 steps 18-22 and 25-29); previously a single
    merge was performed. A new neuron continues to start from the identity covariance (step 17).
- `SHEF`
  - Covariance matrices are inverted with the regularisation (S + εI)⁻¹ given in Eq. 17.
  - The merge test scales the merging neuron by r²S, following Eq. 37 and the proof in Appendix B.
  - A neuron with exactly M members is now eligible for merging (Algorithm 1 step 15, n ≥ M).
  - A class whose first chunk contains a single sample starts at √ε (Eq. 25, Lemma 1), and a class first
    seen after the first chunk also starts at √ε (p. 6).
- `TRACED`: a class whose first chunk contains a single sample starts at √ε, consistent with SHEF, on
  whose threshold TRACED's is based.

### Fixed
- `LRHE`: when a class first appeared as a single sample, its initial width could be 0. Such a class
  now uses the chunk's mean pairwise distance until it is seen in a chunk of two or more samples.
- `LRHE.predict` now adds ε to the widths, as the membership function already does, to avoid division
  by zero.
- `SCIL.predict` now clips eigenvalues at 0 and adds ε, to avoid division by zero or NaN from round-off.
- These safeguards only take effect in the edge cases described; other predictions are unchanged.

### Removed
- The `r` argument of `D4`. Eq. 7 of the D4 paper subtracts 1, and since `r` shifted every projected
  distance by the same amount, it did not affect predictions. Please remove `r=` from existing `D4(...)`
  calls.

### Tests
- The parity tests against `deprecated/spdal.py` now use copies of the 0.3.0 classes, kept in
  `tests/legacy_0_3_0/`, since the paper code predates these changes.
- `tests/test_paper_rules.py` adds a small test for each of the changes above.

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
