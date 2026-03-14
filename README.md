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
git clone https://github.com/your-org/single-pass-discard-after-learn
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

### Incremental (chunk) learning

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

| Class | Full name | Year | Key idea |
|-------|-----------|------|----------|
| `VEBF` | Versatile Elliptic Basis Function | 2010 | Foundation: PCA-axis hyperellipsoids, single-datum online learning |
| `LRHE` | Learning with Recoil in Hyperellipsoidal Structure | 2020 | Shrink-and-shift recoil to handle noisy boundary data |
| `SCIL` | Streaming Chunk Incremental Learning | 2019 | Neuron merging with parallel-axis covariance pooling |
| `SHEF` | Scalable Hyper-Ellipsoidal Function | 2020 | Regularized covariance + Mahalanobis-based prediction |
| `D4`  | Diversion of Data Distribution Direction | 2026 | Hybrid width formula; principal-axis projection for coincident regions |
| `TRACED` | Trend-Adaptive Classification with Ellipsoidal Disambiguation | TBD | Adds EMA displacement/expansion tracking for exterior-region prediction |

### VEBF

```python
from spdal import VEBF
clf = VEBF(theta=0, delta=1, epsilon=1e-10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | `0` | Overlap threshold for neuron merging |
| `delta` | `1` | Width scaling from pairwise distances |
| `epsilon` | `1e-10` | Numerical floor |

### LRHE

```python
from spdal import LRHE
clf = LRHE(alpha=0.5, theta=0, delta=1, epsilon=1e-10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | Shrink multiplier during recoil (0–1) |
| `theta` | `0` | Overlap threshold for merging |
| `delta` | `1` | Width scaling from pairwise distances |
| `epsilon` | `1e-10` | Numerical floor |

### SCIL

```python
from spdal import SCIL
clf = SCIL(N0=3, eta=2, delta=1, theta=0, epsilon=1e-10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N0` | `3` | Min samples for an active neuron |
| `eta` | `2` | Width expansion scaling factor |
| `delta` | `1` | Width scaling from pairwise distances |
| `theta` | `0` | Merge overlap threshold |
| `epsilon` | `1e-10` | Numerical floor |

### SHEF

```python
from spdal import SHEF
clf = SHEF(M=3, r=1.5, epsilon=1e-10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | `3` | Min samples before adaptive threshold triggers |
| `r` | `1.5` | Ellipsoid radius scaling constant |
| `epsilon` | `1e-10` | Regularization / numerical floor |

### D4

```python
from spdal import D4
clf = D4(width_parameter=1, reduce_dims=0, delta=1, norm=2, r=1.5, threshold=15, epsilon=1e-10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width_parameter` | `1` | Blend: `1` = pure statistical width, `0` = pure expansion-based |
| `reduce_dims` | `0` | Principal axes to drop in disambiguation subspace |
| `delta` | `1` | Width scaling from pairwise distances |
| `norm` | `2` | Lp norm for projected distance |
| `r` | `1.5` | Radius scaling factor |
| `threshold` | `15` | Angle threshold (degrees) for axis pairing |
| `epsilon` | `1e-10` | Numerical floor |

D4 maintains **one neuron per class**. When two nearest neurons belong to different classes, it pairs their principal axes by smallest angle and assigns the class with the smaller projected distance in that subspace.

### TRACED

```python
from spdal import TRACED
clf = TRACED(
    alpha=0.5, beta=0.01, delta=2, width_parameter=1,
    reduce_dims=1, N0=3, r=2.507, norm=2,
    method='overlap-outside', distance_metric='boundary',
    threshold=15, epsilon=1e-10,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | EMA weight for displacement smoothing (`0` = disabled) |
| `beta` | `0.01` | EMA weight for expansion-rate smoothing (`0` = disabled) |
| `delta` | `2` | Dynamic threshold scaling (mean NN distance × delta) |
| `width_parameter` | `1` | Blend: `1` = statistical, `0` = expansion-based |
| `reduce_dims` | `1` | Axes to drop in coincident-region disambiguation |
| `N0` | `3` | Min samples for an active neuron |
| `r` | `sqrt(2π)` | Statistical width scaling |
| `norm` | `2` | Lp norm for distance calculation |
| `method` | `'overlap-outside'` | Corrections to apply: `'overlap'`, `'outside'`, or both |
| `distance_metric` | `'boundary'` | `'boundary'` or `'center'` |
| `threshold` | `15` | Angle threshold (degrees) for axis pairing |
| `epsilon` | `1e-10` | Numerical floor |

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

1. **VEBF** — Jaiyen, S., Lursinsap, C., & Phimoltares, S. (2010). A New Versatile Elliptic Basis Function Neural Network. *IEEE Transactions on Neural Networks*, 21(3), 381–392.
2. **LRHE** — Jindadoungrut, K., Phimoltares, S., & Lursinsap, C. (2020). Neural Learning With Recoil Behavior in Hyperellipsoidal Structure. *IEEE Access*, 8, 114643–114655.
3. **SCIL** — Junsawang, P., Phimoltares, S., & Lursinsap, C. (2019). Streaming chunk incremental learning for class-wise data stream classification with fast learning speed and low structural complexity. *PLOS ONE*, 14(9), e0220624.
4. **SHEF** — Rungcharassang, P., & Lursinsap, C. (2020). Scalable Hyper-Ellipsoidal Function with Projection Ratio for Local Distributed Streaming Data Classification. *IEEE Access*. DOI: 10.1109/ACCESS.2020.2997944.
5. **D4** — Wongsriphisant, P., Plaimas, K., & Lursinsap, C. (2026). Markov-based continuous learning with diversion of data distribution direction for streaming data in limited memory. *Expert Systems With Applications*, 298, 129818.
   - Corrigendum (Theorem 2): [docs/D4_corrigendum.md](docs/D4_corrigendum.md) · [PDF](docs/D4_corrigendum.pdf)
6. **TRACED** — Wongsriphisant, P., Plaimas, K., & Lursinsap, C. TRACED: Trend-Adaptive Classification with Ellipsoidal Disambiguation for Resolving Exterior and Coincident Regions in Data Streams. *Preprint submitted to Elsevier*.

---

## Notes

- The original monolithic implementation is preserved at [`deprecated/spdal.py`](deprecated/spdal.py) for reference.
- Refactoring into the modular `src/spdal/` package structure, docstrings, and parameter naming were performed by Claude (Anthropic) and reviewed by the project owner.
