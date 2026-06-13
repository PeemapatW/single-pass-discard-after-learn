# spdal

**Single-Pass Discard-After-Learn** — hyperellipsoid classifiers for online streaming data.

Each training sample is processed once and then discarded. No full dataset is ever stored. All classifiers implement scikit-learn's `partial_fit` / `predict` interface.


---

## Installation

```bash
pip install spdal
```

Development mode:

```bash
git clone https://github.com/PeemapatW/single-pass-discard-after-learn
cd single-pass-discard-after-learn
pip install -e ".[dev]"
```

---

## Quick Start

```python
from sklearn.datasets import make_classification
from spdal import LRHE

X, y = make_classification(n_samples=500, random_state=42)

clf = LRHE()
clf.fit(X[:400], y[:400])
print(clf.predict(X[400:]))          # array of class labels
print(len(clf.neuron_list))          # number of learned prototypes
```

### Incremental learning

```python
from spdal import TRACED
import numpy as np

X, y = make_classification(n_samples=500, random_state=42)
classes = np.unique(y)

clf = TRACED()
for i in range(0, 400, 50):
    clf.partial_fit(X[i:i+50], y[i:i+50], classes=classes)

print(clf.predict(X[400:]))
```

---

## Classifiers

| Class | Full name | Year |
|-------|-----------|------|
| `VEBF` | Versatile Elliptic Basis Function | 2010 |
| `SCIL` | Streaming Chunk Incremental Learning | 2019 |
| `LRHE` | Learning with Recoil in Hyperellipsoidal Structure | 2020 |
| `SHEF` | Scalable Hyper-Ellipsoidal Function | 2020 |
| `D4`  | Diversion of Data Distribution Direction | 2026 |
| `TRACED` | Trend-Adaptive Classification with Ellipsoidal Disambiguation | 2026 |

### Comparison

| Feature | VEBF | SCIL | LRHE | SHEF | D4 | TRACED |
|---------|------|------|------|------|-----|--------|
| Learning mode | Single | Chunk | Single | Single | Chunk | Chunk |
| Neurons per class | Multiple | Multiple | Multiple | Multiple | One | Multiple |
| Width formula | Init: avg pairwise dist; Update: width + center shift; Merge: sqrt(2π\|λ\|) | Update: width + center shift; Expand: sqrt(1+η·max\_psi)·w; Merge: 1.96 sqrt(\|λ\|/n) | Init: avg pairwise dist; Update: width + center shift; Merge: sqrt(2π\|λ\|) | Update: recursive covariance; Width = r·sqrt(λ) | Init: avg pairwise dist; Update: α·sqrt(2πλ) + (1-α)·(width + center shift) | Init: mean NN dist; Update: γ·r·sqrt(λ) + (1-γ)·(width + center shift); Merge: r·sqrt(λ) |
| Creation threshold | Hyperellipsoidal test | Hyperellipsoidal test | Hyperellipsoidal test | Euclidean distance (dynamic threshold) | N/A (always merges) | Euclidean distance (dynamic threshold) |
| Ambiguity resolution | — | — | Shift-and-shrink | Discriminant projection | Coincident: principal-axis subspace | Coincident: principal-axis subspace; Exterior: EMA displacement + expansion |

---

## Parameters



### VEBF

```python
from spdal import VEBF
clf = VEBF(theta=0, delta=1)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | `0` | Overlap threshold for neuron merging |
| `delta` | `1` | Width scaling from pairwise distances for initial width [$\delta > 0$]|

### SCIL

```python
from spdal import SCIL
clf = SCIL(N0=3, eta=2, delta=1, theta=0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N0` | `3` | Min samples for an active neuron in prediction|
| `eta` | `2` | Width expansion scaling factor |
| `delta` | `1` | Width scaling from pairwise distances for initial width [$\delta > 0$]|
| `theta` | `0` | Overlap threshold for neuron merging |

### LRHE

```python
from spdal import LRHE
clf = LRHE(alpha=0.99, theta=0, delta=1)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.99` | Shrink multiplier during recoil [$\alpha \in [0,1]$]; paper-recommended (gradual shrinking)|
| `theta` | `0` | Overlap threshold for neuron merging|
| `delta` | `1` | Width scaling from pairwise distances for initial width [$\delta > 0$]|

### SHEF

```python
from spdal import SHEF
clf = SHEF(M=3, r=1.5)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | `3` | Min samples before adaptive threshold triggers |
| `r` | `1.5` | Ellipsoid radius scaling constant [$r > 0$]|

### D4

```python
from spdal import D4
clf = D4(width_parameter=1, reduce_dims=0, delta=1, norm=2, r=1.5, threshold=15)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width_parameter` | `1` | Blend: `1` = pure statistical width, `0` = pure expansion-based [$\text{width\_parameter} \in [0,1]$]|
| `reduce_dims` | `0` | Axes to drop in coincident-region disambiguation |
| `delta` | `1` | Width scaling from pairwise distances for initial width [$\delta > 0$]|
| `norm` | `2` | Lp norm for distance calculation  |
| `r` | `1.5` | Statistical ellipsoid radius width scaling factor [$r > 0$] |
| `threshold` | `15` | Angle threshold (degrees) for axis pairing |

D4 maintains **one neuron per class**. When two nearest neurons belong to different classes, it select their axes using parallel and compactness criteria and assigns the class with the smaller projected distance in that subspace.

> **Note:** Theorem 2 of the D4 paper contains a sign error in the proof. This does not affect the algorithm or experimental results. See [technical note](https://github.com/PeemapatW/single-pass-discard-after-learn/blob/main/docs/D4_theorem2_note.md) for details.

### TRACED

```python
from spdal import TRACED
clf = TRACED(
    alpha=0.5, beta=0.01, delta=2, width_parameter=1,
    reduce_dims=1, N0=3, r=2.507, norm=2,
    method='overlap-outside', distance_metric='boundary',threshold=15
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | EMA weight for displacement smoothing (`0` = disabled) [$\alpha \in [0,1]$]|
| `beta` | `0.01` | EMA weight for expansion-rate smoothing (`0` = disabled) [$\beta \in [0,1]$]|
| `delta` | `2` | Initial dynamic threshold scaling (mean NN distance × delta) [$\delta > 0$]|
| `width_parameter` | `1` | Blend: `1` = pure statistical width, `0` = pure expansion-based [$\text{width\_parameter} \in [0,1]$] |
| `reduce_dims` | `1` | Axes to drop in coincident-region disambiguation |
| `N0` | `3` | Min samples for an active neuron in prediction and adaptive threshold triggers |
| `r` | `sqrt(2π)` | Statistical ellipsoid radius width scaling factor [$r > 0$]|
| `norm` | `2` | Lp norm for distance calculation |
| `method` | `'overlap-outside'` | Corrections to apply: `'overlap'`, `'outside'`, or both |
| `distance_metric` | `'boundary'` | `'boundary'` or `'center'` |
| `threshold` | `15` | Angle threshold (degrees) for axis pairing |

TRACED resolves two ambiguous regions:
- **Coincident** (x inside multiple classes) — principal-axis subspace projection (like D4)
- **Exterior** (x outside all neurons) — predicts using EMA-smoothed displacement and expansion as a trend model

---

## sklearn Interface

All classifiers are `sklearn.base.BaseEstimator` subclasses and support:

```python
clf.fit(X, y)                              # full batch training
clf.partial_fit(X, y, classes=classes)     # incremental update
clf.predict(X)                             # returns array of class labels
clf.classes_                               # array of known class labels
clf.neuron_list                            # list of neuron dicts
```

Compatible with scikit-learn pipelines and cross-validation tools that support `partial_fit`.

---

## Neuron Schema

Learned prototypes are stored in `clf.neuron_list` as a list of dicts:

```python
{
    'y':             class_label,
    'center':        np.ndarray,     # prototype position
    'cov':           np.ndarray,     # covariance matrix
    'eig_component': np.ndarray,     # PCA eigenvectors
    'width':         np.ndarray,     # semi-axis lengths
    'n':             int,            # sample count
    # SCIL, D4, TRACED only:
    'variance':      np.ndarray,     # eigenvalues
    # TRACED only:
    'displacement':  np.ndarray,     # EMA displacement vector
    'expansion':     np.ndarray,     # EMA per-axis expansion rates
}
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run a single test class
pytest tests/test_classifiers.py::TestTRACED -v

# Build for PyPI
pip install build && python -m build
```

---

## References

1. **VEBF** — Jaiyen, S., Lursinsap, C., & Phimoltares, S. (2010). A Very Fast Neural Learning for Classification Using Only New Incoming Datum. *IEEE Transactions on Neural Networks*, 21(3), 381–392. [[paper]](https://ieeexplore.ieee.org/document/5382496)
2. **SCIL** — Junsawang, P., Phimoltares, S., & Lursinsap, C. (2019). Streaming chunk incremental learning for class-wise data stream classification with fast learning speed and low structural complexity. *PLOS ONE*, 14(9), e0220624. [[paper]](https://doi.org/10.1371/journal.pone.0220624)
3. **LRHE** — Jindadoungrut, K., Phimoltares, S., & Lursinsap, C. (2020). Neural Learning With Recoil Behavior in Hyperellipsoidal Structure. *IEEE Access*, 8, 114643–114655. [[paper]](https://ieeexplore.ieee.org/document/9120020)
4. **SHEF** — Rungcharassang, P., & Lursinsap, C. (2020). Scalable Hyper-Ellipsoidal Function with Projection Ratio for Local Distributed Streaming Data Classification. *IEEE Access*, 8, 105460–105474. [[paper]](https://ieeexplore.ieee.org/document/9102265)
5. **D4** — Wongsriphisant, P., Plaimas, K., & Lursinsap, C. (2026). Markov-based continuous learning with diversion of data distribution direction for streaming data in limited memory. *Expert Systems With Applications*, 298, 129818. [[paper]](https://doi.org/10.1016/j.eswa.2025.129818)
   - Technical note (Theorem 2): [docs/D4_theorem2_note.md](https://github.com/PeemapatW/single-pass-discard-after-learn/blob/main/docs/D4_theorem2_note.md)
6. **TRACED** — Wongsriphisant, P., Plaimas, K., & Lursinsap, C. (2026). TRACED: Trend-Adaptive Classification with Ellipsoidal Disambiguation for Resolving Exterior and Coincident Regions in Data Streams. *Information Sciences*, 743, 123338. [[paper]](https://doi.org/10.1016/j.ins.2026.123338)

---

## Citation

If you use this library in your research, please cite the relevant paper(s):

```bibtex
@article{WONGSRIPHISANT2026D4,
    title = {Markov-based continuous learning with diversion of data distribution direction for streaming data in limited memory},
    journal = {Expert Systems with Applications},
    volume = {298},
    pages = {129818},
    year = {2026},
    issn = {0957-4174},
    doi = {https://doi.org/10.1016/j.eswa.2025.129818},
    url = {https://www.sciencedirect.com/science/article/pii/S0957417425034335},
    author = {Peemapat Wongsriphisant and Kitiporn Plaimas and Chidchanok Lursinsap},
}

@article{WONGSRIPHISANT2026TRACED,
    title = {TRACED: Trend-adaptive classification with ellipsoidal disambiguation for resolving exterior and coincident regions in data streams},
    journal = {Information Sciences},
    volume = {743},
    pages = {123338},
    year = {2026},
    issn = {0020-0255},
    doi = {https://doi.org/10.1016/j.ins.2026.123338},
    url = {https://www.sciencedirect.com/science/article/pii/S0020025526002690},
    author = {Peemapat Wongsriphisant and Kitiporn Plaimas and Chidchanok Lursinsap},
}
```

---

## Notes

- The original monolithic implementation is preserved at [`deprecated/spdal.py`](https://github.com/PeemapatW/single-pass-discard-after-learn/blob/main/deprecated/spdal.py) for reference.
- Refactoring into the modular `src/spdal/` package structure, docstrings, and parameter naming were performed by Claude (Anthropic) and reviewed by the project owner.

## Changelog

See [CHANGELOG.md](https://github.com/PeemapatW/single-pass-discard-after-learn/blob/main/CHANGELOG.md). Latest — **0.2.0**: default eigensolver is now the symmetric `eigh` (was `eig`); `LRHE` default `alpha` is `0.99` (was `0.5`). Both may change results vs 0.1.1 — see the changelog.
