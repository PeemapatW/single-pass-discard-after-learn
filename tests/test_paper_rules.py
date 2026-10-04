"""The 0.4.0 changes, one rule per test.

The parity tests compare against the paper code through the frozen 0.3.0 classes (tests/legacy_0_3_0), so
they do not exercise these rules. Each test here pins one of them on a small hand-built case, and most
also check that 0.3.0 behaved differently, so a test cannot pass by testing nothing.
"""
import numpy as np
import pytest

from spdal import D4, LRHE, SCIL, SHEF, TRACED, VEBF
import legacy_0_3_0 as old


def _line(n, step=0.1, d=2):
    """n points of one class on a short line, so every one is captured by the first neuron."""
    X = np.zeros((n, d))
    X[:, 0] = np.arange(n) * step
    return X, np.zeros(n, dtype=int)


# --------------------------------------------------------------------------- VEBF


def test_vebf_new_neuron_starts_with_zero_covariance():
    """VEBF Algorithm step 6, 'Else' b: S = 0 for a new node (the base class, and LRHE, use I)."""
    new, legacy = VEBF(), old.VEBF()
    new.init_width = legacy.init_width = {0: np.ones(3)}
    assert np.array_equal(new.create_new_neuron(np.ones(3), 0)['cov'], np.zeros((3, 3)))
    assert np.array_equal(legacy.create_new_neuron(np.ones(3), 0)['cov'], np.eye(3))


@pytest.mark.parametrize("cls, legacy", [(VEBF, old.VEBF), (LRHE, old.LRHE)])
def test_width_grows_only_after_n0_samples(cls, legacy):
    """VEBF step 6e / LRHE p. 4: the widths are updated only when n > N0 = 2 after the update."""
    X, y = _line(3)
    w = {}
    for c in (cls, legacy):
        clf = c()
        clf.neuron_list, clf.init_width = [], {0: np.full(2, 50.0)}   # wide: every point is captured
        clf.partial_fit(X[:1], y[:1])
        w0 = clf.neuron_list[0]['width'].copy()
        clf.partial_fit(X[1:2], y[1:2])
        w1 = clf.neuron_list[0]['width'].copy()
        clf.partial_fit(X[2:3], y[2:3])
        w2 = clf.neuron_list[0]['width'].copy()
        assert len(clf.neuron_list) == 1
        w[c] = (w0, w1, w2)
    w0, w1, w2 = w[cls]
    assert np.array_equal(w0, w1), "the second sample (n = 2) must not grow the widths"
    assert not np.array_equal(w1, w2), "the third sample (n = 3 > N0) grows them"
    lw0, lw1, _ = w[legacy]
    assert not np.array_equal(lw0, lw1), "0.3.0 grew them on every update"


@pytest.mark.parametrize("cls", [VEBF, LRHE])
def test_growth_test_uses_the_tentative_eigenvectors(cls):
    """VEBF steps 4-5 / LRHE Alg. 1 steps 12-15: psi of the incoming point is computed with the
    eigenvectors of the TENTATIVE covariance (0.3.0 used the stored ones)."""
    calls = []

    class Spy(cls):
        def hyperellipsoidal_fn(self, x, c, U, w):
            calls.append((np.array(x), np.array(U)))
            return super().hyperellipsoidal_fn(x, c, U, w)

    rng = np.random.RandomState(0)
    X = rng.randn(6, 3) * [1.0, 0.3, 0.1]
    y = np.zeros(6, dtype=int)
    clf = Spy(delta=50)
    clf.fit(X[:5], y[:5])
    before = {k: np.array(v) for k, v in clf.neuron_list[0].items() if k in ('n', 'center', 'cov')}
    calls.clear()
    clf.partial_fit(X[5:], y[5:])
    growth = [U for x, U in calls if np.array_equal(x, X[5])]
    n = before['n']
    cen_t = (n * before['center'] + X[5]) / (n + 1)
    cov_t = ((n * before['cov'] + np.outer(X[5], X[5]) - np.outer(before['center'], before['center'])) / (n + 1)
             - np.outer(cen_t, cen_t) + np.outer(before['center'], before['center']))
    U_t, _ = clf.compute_sorted_eigencomponent(cov_t)
    assert any(np.allclose(U, U_t) for U in growth)


# --------------------------------------------------------------------------- LRHE


def _three_neurons(cls):
    """Neuron 0 covers the centres of neurons 1 and 2; 1 and 2 do not cover each other."""
    clf = cls(delta=1)
    clf.neuron_list = []
    for c, s2, w, n in [((0.0, 0.0), 1.0, 3.0, 50), ((1.0, 0.0), 0.01, 0.1, 5), ((-1.0, 0.0), 0.01, 0.1, 5)]:
        clf.neuron_list.append({'y': 0, 'center': np.array(c), 'cov': np.eye(2) * s2,
                                'eig_component': np.eye(2), 'width': np.array([w, w]), 'n': n})
    return clf


def test_lrhe_merges_every_qualifying_neuron():
    """LRHE Alg. 1 steps 18-22 / 25-29: every other same-class neuron meeting the merging condition is
    merged into the new / updated one. 0.3.0 merged once and stopped."""
    new = _three_neurons(LRHE)
    new.merge_neuron(0, 0)
    assert len(new.neuron_list) == 1 and new.neuron_list[0]['n'] == 60
    legacy = _three_neurons(old.LRHE)
    legacy.merge_neuron(0, 0)
    assert len(legacy.neuron_list) == 2


def test_lrhe_one_sample_class_does_not_get_zero_width():
    """A class whose first slice holds one sample used to get width 0 for good; it now takes the batch's
    mean pairwise distance until a slice of >= 2 of its samples fixes it."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    y = np.array([0, 0, 0, 1])
    clf = LRHE()
    clf.fit(X, y)
    assert np.all(clf.init_width[1] > 0)
    legacy = old.LRHE()
    legacy.fit(X, y)
    assert np.all(legacy.init_width[1] == 0)
    X2 = np.array([[5.0, 5.0], [5.5, 5.0], [0.5, 0.5]])
    clf.partial_fit(X2, np.array([1, 1, 0]))
    assert 1 not in clf._waiting_width                   # fixed now, from its own two samples


def test_lrhe_predict_with_a_zero_width_is_finite():
    clf = LRHE()
    clf.neuron_list = [{'y': 0, 'center': np.zeros(2), 'cov': np.eye(2), 'eig_component': np.eye(2),
                        'width': np.array([0.0, 1.0]), 'n': 1},
                       {'y': 1, 'center': np.full(2, 3.0), 'cov': np.eye(2), 'eig_component': np.eye(2),
                        'width': np.array([1.0, 1.0]), 'n': 1}]
    clf.set_classes()
    assert list(clf.predict(np.array([[3.0, 3.0]]))) == [1]


# --------------------------------------------------------------------------- SCIL


def test_scil_predict_with_a_zero_variance_is_finite():
    """A zero (or round-off negative) eigenvalue used to divide by zero / give NaN in predict."""
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3)
    X[:, 2] = 0.0                                         # constant feature -> zero eigenvalue
    y = (X[:, 0] > 0).astype(int)
    clf = SCIL(delta=2)
    clf.fit(X, y)
    for n in clf.neuron_list:
        n['variance'] = np.where(np.arange(len(n['variance'])) == 2, -1e-18, n['variance'])
    pred = clf.predict(X)
    assert set(pred) <= {0, 1} and (pred == y).mean() > 0.5


# --------------------------------------------------------------------------- SHEF


def test_shef_one_sample_class_starts_at_sqrt_eps():
    """Eq. 25 / Lemma 1: sqrt(epsilon), not sqrt(e)."""
    clf = SHEF()
    assert clf.find_median_dist_to_neighbor(np.zeros((1, 2))) == pytest.approx(np.sqrt(clf.epsilon))
    assert old.SHEF().find_median_dist_to_neighbor(np.zeros((1, 2))) == pytest.approx(np.sqrt(np.e))


def test_shef_late_class_starts_at_sqrt_eps():
    """p. 6: a class first seen after the first chunk starts at sqrt(epsilon); first-chunk classes start
    at their own median nearest-neighbour distance."""
    rng = np.random.RandomState(1)
    X = rng.randn(40, 2)
    clf = SHEF()
    clf.distance_init(X[:20], np.zeros(20, dtype=int))
    assert clf.dist_ths[0] == pytest.approx(clf.find_median_dist_to_neighbor(X[:20]))
    clf.distance_init(X[20:], np.ones(20, dtype=int))
    assert clf.dist_ths[1] == pytest.approx(np.sqrt(clf.epsilon))
    legacy = old.SHEF()
    legacy.distance_init(X[:20], np.zeros(20, dtype=int))
    legacy.distance_init(X[20:], np.ones(20, dtype=int))
    assert legacy.dist_ths[1] > 1e-3


def test_shef_inverse_is_regularised():
    """Eq. 17: (S + epsilon I)^-1, which also exists for a singular S."""
    clf = SHEF()
    S = np.diag([1.0, 0.0])
    assert np.allclose(clf._reg_inv(S), np.diag([1 / (1 + clf.epsilon), 1 / clf.epsilon]))


def test_shef_merge_gate_includes_m():
    """Alg. 1 step 15: a neuron with n = M is eligible for merging (0.3.0 required n > M)."""
    def two(cls):
        clf = cls(M=3, r=1.5)
        clf.dist_ths = {0: 1.0}
        clf.neuron_list = [{'y': 0, 'center': np.zeros(2), 'cov': np.eye(2) * 0.5, 'n': 3},
                           {'y': 0, 'center': np.array([0.1, 0.0]), 'cov': np.eye(2) * 0.5, 'n': 3}]
        clf.merge_neuron(0, 0)
        return len(clf.neuron_list)
    assert two(SHEF) == 1
    assert two(old.SHEF) == 2


# --------------------------------------------------------------------------- TRACED, D4


def test_traced_one_sample_class_starts_at_sqrt_eps():
    clf = TRACED()
    assert clf.find_mean_dist_to_neighbor(np.zeros((1, 2))) == pytest.approx(np.sqrt(clf.epsilon))


def test_d4_has_no_r():
    """D4's Eq. 7 subtracts 1; the `r` it used to subtract instead never changed a prediction."""
    assert 'r' not in D4().get_params()
    with pytest.raises(TypeError):
        D4(r=1.5)


def test_shef_default_epsilon_is_the_papers():
    """0.4.1: SHEF's default epsilon is the paper's 1e-4 (p. 5 and 13); 0.4.0 used 1e-10."""
    assert SHEF().epsilon == 1e-4
    assert old.SHEF().epsilon == 1e-10
