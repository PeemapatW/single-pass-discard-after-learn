"""
Regression tests using the Phishing dataset from river.

Each scenario trains the package classifier and its deprecated/spdal.py
counterpart on the same data in the same process and asserts they agree, rather
than comparing against prediction lists frozen on one machine -- those turned
out to be tied to the numeric libraries underneath (see the note above
PARITY_CASES in test_classifiers.py).

Dataset: river.datasets.Phishing — 1250 samples, 9 binary features, 2 classes.
Split: train=X[:1000], test=X[1000:] (250 test samples).

Two modes are tested per classifier:
  - Batch (single partial_fit with 1000 training samples)
  - Incremental (10 chunks of 100 samples each via repeated partial_fit)

Requires river package (pip install -e ".[experiments]").
Tests are skipped automatically if river is not installed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("river", reason="river not installed; skip Phishing tests")

from river.datasets import Phishing  # noqa: E402 — after importorskip
from spdal import LRHE, VEBF, SCIL, SHEF, D4, TRACED  # noqa: E402
from test_classifiers import (  # noqa: E402 — same test package
    TRACED_MIN_AGREEMENT,
    _ALL,
    _NEW_CTORS,
    _TRACED_NEW_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


def _phishing():
    data = Phishing()
    X = np.array([np.array(list(s[0].values())) for s in data])
    y = np.array([int(s[1]) for s in data])
    return X, y


@pytest.fixture(scope="module")
def phishing_batch():
    """Single partial_fit on 1000 samples; test on remaining 250."""
    X, y = _phishing()
    return [(X[:1000], y[:1000])], X[1000:], np.unique(y)


@pytest.fixture(scope="module")
def phishing_chunks():
    """10 chunks of 100 samples for incremental partial_fit; test on 250."""
    X, y = _phishing()
    chunks = [(X[i:i+100], y[i:i+100]) for i in range(0, 1000, 100)]
    return chunks, X[1000:], np.unique(y)


# ---------------------------------------------------------------------------
# Parity against deprecated/spdal.py
# ---------------------------------------------------------------------------

PARITY_CASES = (
    [(f'batch-{n}', 'phishing_batch', n) for n in _ALL]
    + [(f'chunks-{n}', 'phishing_chunks', n) for n in _ALL]
)


def _train(clf, chunks, classes):
    for Xc, yc in chunks:
        clf.partial_fit(Xc, yc, classes=classes)
    return clf


def _make_dep(dep_module, name):
    if name == 'TRACED':
        return dep_module.TRACED(**_TRACED_NEW_DEFAULTS)
    return getattr(dep_module, name)()


@pytest.mark.parametrize("case", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
def test_matches_deprecated(request, dep_module, case):
    """Package output must match deprecated/spdal.py on the Phishing stream."""
    _, fixture, name = case
    chunks, X_test, classes = request.getfixturevalue(fixture)

    clf_new = _train(_NEW_CTORS[name](), chunks, classes)
    clf_dep = _train(_make_dep(dep_module, name), chunks, classes)
    preds_new = np.asarray(clf_new.predict(X_test))
    preds_dep = np.asarray(clf_dep.predict(X_test))

    if name == 'TRACED':
        # TRACED deliberately diverges from the paper code; see
        # _TRACED_ORIENTATION_SKIP in test_classifiers.py.
        agreement = float((preds_new == preds_dep).mean())
        assert agreement >= TRACED_MIN_AGREEMENT, (
            f"TRACED agreement with the paper code fell to {agreement:.1%} "
            f"(floor {TRACED_MIN_AGREEMENT:.0%})"
        )
    else:
        assert len(clf_new.neuron_list) == len(clf_dep.neuron_list), (
            f"neuron count: new={len(clf_new.neuron_list)} "
            f"deprecated={len(clf_dep.neuron_list)}"
        )
        np.testing.assert_array_equal(preds_new, preds_dep)
