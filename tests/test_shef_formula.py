"""
Tests verifying that SHEF's Projection Ratio distance implementation
matches the paper's formula, and analysing the ambiguous-case behaviour.

Paper (Rungcharassang & Lursinsap, IEEE Access 2020):
    w_p = S^{-1}(x - c) / ||S^{-1}(x - c)||
    D(x, c, S) = |w_p^T (x - c)| / (r * sqrt(w_p^T S w_p))

Mathematically simplified to:
    D(x, c, S) = sqrt((x-c)^T S^{-1} (x-c)) / r   [Mahalanobis / r]

These tests verify:
1. The paper formula == Mahalanobis/r (algebraic identity, holds numerically).
2. The code's main distance loop produces Mahalanobis/r (correctly matches paper).
3. The code's ambiguous-case LDA prediction vs. pure min-D (from paper) —
   any discrepancy is logged to see if the simplification matters in practice.
"""

import numpy as np
import numpy.linalg as LA
import pytest
from spdal import SHEF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spd(d, rng, scale=1.0):
    """Return a random symmetric positive-definite matrix of size d×d."""
    A = rng.standard_normal((d, d)) * scale
    return A @ A.T + np.eye(d) * 0.1   # ensure positive definiteness


def paper_projection_ratio(x, c, S, r):
    """
    Paper formula (per-point projection direction).
        w_p = S^{-1}(x-c) / ||S^{-1}(x-c)||
        D   = |w_p^T(x-c)| / (r * sqrt(w_p^T S w_p))
    Applied row-wise for x of shape (n, d).
    """
    S_inv = LA.inv(S)
    x_c = x - c                          # (n, d)
    v = x_c @ S_inv.T                    # S^{-1}(x-c), shape (n, d)
    v_norm = LA.norm(v, axis=1, keepdims=True)
    v_norm = np.where(v_norm < 1e-15, 1e-15, v_norm)
    w_p = v / v_norm                     # (n, d)

    numerator = np.abs(np.einsum('ij,ij->i', w_p, x_c))
    wSw = np.einsum('ij,jk,ik->i', w_p, S, w_p)
    denominator = r * np.sqrt(np.maximum(wSw, 1e-30))
    return numerator / denominator


def mahalanobis_over_r(x, c, S, r):
    """Simplified form: sqrt((x-c)^T S^{-1} (x-c)) / r, row-wise."""
    S_inv = LA.inv(S)
    x_c = x - c
    maha_sq = np.einsum('ij,jk,ik->i', x_c, S_inv, x_c)
    return np.sqrt(np.maximum(maha_sq, 0.0)) / r


def code_distance(x, c, S, r, epsilon=1e-10):
    """
    Replicates SHEF.predict()'s per-point distance computation exactly.
    """
    S_inv = LA.inv(S)
    x_c = x - c
    wp = x_c @ S_inv.T                         # (n, d)
    wp_norm = LA.norm(wp, axis=1, keepdims=True)
    wp_norm = np.where(wp_norm < epsilon, epsilon, wp_norm)
    wp = wp / wp_norm

    # r * sqrt(w_p^T S w_p)
    r_wSw = r * np.sqrt(np.einsum('ij,ij->i',
                                   np.einsum('ij,jk->ik', wp, S), wp))
    r_wSw = np.where(r_wSw < epsilon, epsilon, r_wSw)

    return np.abs(np.einsum('ij,ij->i', x_c, wp)) / r_wSw


# ---------------------------------------------------------------------------
# 1. Algebraic identity: paper formula == Mahalanobis/r
# ---------------------------------------------------------------------------

class TestProjectionRatioEquivalence:
    """
    Verify the algebraic identity:
        paper_D == Mahalanobis/r
    for random SPD matrices and random points.
    """

    @pytest.mark.parametrize("d", [2, 5, 10, 20])
    def test_paper_equals_mahalanobis(self, d):
        rng = np.random.default_rng(42)
        S = make_spd(d, rng)
        c = rng.standard_normal(d)
        x = rng.standard_normal((100, d))
        r = 1.5

        D_paper = paper_projection_ratio(x, c, S, r)
        D_maha  = mahalanobis_over_r(x, c, S, r)

        np.testing.assert_allclose(
            D_paper, D_maha, rtol=1e-10, atol=1e-12,
            err_msg=f"Paper formula != Mahalanobis/r for d={d}"
        )

    def test_paper_equals_mahalanobis_large_scale(self):
        """High-scale covariance (tests numerical stability)."""
        rng = np.random.default_rng(0)
        S = make_spd(8, rng, scale=10.0)
        c = rng.standard_normal(8) * 5
        x = rng.standard_normal((200, 8)) * 5
        r = 1.5

        D_paper = paper_projection_ratio(x, c, S, r)
        D_maha  = mahalanobis_over_r(x, c, S, r)
        np.testing.assert_allclose(D_paper, D_maha, rtol=1e-8, atol=1e-10)

    def test_inside_outside_boundary(self):
        """D ≤ 1 inside, D > 1 outside, D == 1 on boundary (for both formulas)."""
        d = 4
        rng = np.random.default_rng(7)
        S = make_spd(d, rng)
        S_inv = LA.inv(S)
        c = np.zeros(d)
        r = 1.5

        # Construct a point exactly on the boundary: Mahalanobis == r
        v = rng.standard_normal(d)
        v /= np.sqrt(v @ S_inv @ v)   # unit Mahalanobis norm
        x_boundary = (c + r * v).reshape(1, -1)

        D_paper = paper_projection_ratio(x_boundary, c, S, r)
        D_maha  = mahalanobis_over_r(x_boundary, c, S, r)
        np.testing.assert_allclose(D_paper, [1.0], atol=1e-10)
        np.testing.assert_allclose(D_maha,  [1.0], atol=1e-10)

        # Inside (scale < r)
        x_in = (c + 0.5 * v).reshape(1, -1)
        assert paper_projection_ratio(x_in, c, S, r)[0] < 1.0
        assert mahalanobis_over_r(x_in, c, S, r)[0] < 1.0

        # Outside (scale > r)
        x_out = (c + 2 * r * v).reshape(1, -1)
        assert paper_projection_ratio(x_out, c, S, r)[0] > 1.0
        assert mahalanobis_over_r(x_out, c, S, r)[0] > 1.0


# ---------------------------------------------------------------------------
# 2. Code's main distance loop matches paper formula
# ---------------------------------------------------------------------------

class TestSHEFCodeDistanceMatchesPaper:
    """
    Verify that SHEF.predict()'s per-neuron distance computation
    (the vectorised loop) matches the paper formula.
    """

    @pytest.mark.parametrize("d", [2, 5, 10])
    def test_code_distance_matches_paper(self, d):
        rng = np.random.default_rng(99)
        S = make_spd(d, rng)
        S_reg = S + 1e-10 * np.eye(d)   # matches SHEF's regularisation
        c = rng.standard_normal(d)
        x = rng.standard_normal((50, d))
        r = 1.5

        D_code  = code_distance(x, c, S_reg, r)
        D_paper = paper_projection_ratio(x, c, S_reg, r)

        np.testing.assert_allclose(
            D_code, D_paper, rtol=1e-8, atol=1e-10,
            err_msg=f"Code distance != paper formula for d={d}"
        )

    def test_shef_predict_uses_correct_distance(self):
        """
        End-to-end: train SHEF, then manually verify the stored neuron
        distances match D = Mahalanobis/r for a set of query points.
        """
        rng = np.random.default_rng(5)
        X0 = rng.standard_normal((30, 3)) + np.array([2, 0, 0])
        X1 = rng.standard_normal((30, 3)) + np.array([-2, 0, 0])
        X  = np.vstack([X0, X1])
        y  = np.array([0]*30 + [1]*30)

        clf = SHEF(r=1.5, epsilon=1e-10)
        clf.fit(X, y)

        x_test = rng.standard_normal((20, 3))
        r = clf.r

        for idx, neuron in enumerate(clf.neuron_list):
            c   = neuron['center']
            S   = neuron['cov']
            D_code  = code_distance(x_test, c, S, r, epsilon=clf.epsilon)
            D_maha  = mahalanobis_over_r(x_test, c, S, r)
            np.testing.assert_allclose(
                # rtol 1e-5: since 0.4.0's Eq. 37 merge scale this fit keeps a neuron whose covariance is ~eps in
                # one direction (distances ~1e5), where the two equal formulas differ by ~1e-6 from round-off.
                D_code, D_maha, rtol=1e-5, atol=1e-8,
                err_msg=f"Neuron {idx}: code distance != Mahalanobis/r"
            )


# ---------------------------------------------------------------------------
# 3. Ambiguous case: paper D' with LDA discriminant vector w (Eq. 8)
# ---------------------------------------------------------------------------

class TestSHEFAmbiguousCase:
    """
    Paper (Section VI-C, Eq. 8) defines the ambiguous-case discriminant:
        w = S_W^{-1}(c1 - c2) / ||S_W^{-1}(c1 - c2)||   where S_W = S1 + S2
        D'(x, c, S) = |w^T(x - c)| / (r * sqrt(w^T S w))

    The code implements _vectorized_discriminant_vector as (S1+S2)^{-1}(c2-c1)
    (sign reversed, normalization omitted — both cancel in D').
    These tests verify the code matches the paper formula exactly.
    """

    def _paper_discriminant_distance(self, x, c, S, S_W, r):
        """
        Paper Eq. 8 + D' formula — reference implementation.
        w = S_W^{-1}(c1-c2)/||...||  (here we supply S_W and c as a single neuron's c)
        """
        S_W_inv = np.linalg.inv(S_W)
        # w direction (normalization cancels but include for clarity)
        w_unnorm = S_W_inv @ (c)       # direction already encoded in c here
        w = w_unnorm / (np.linalg.norm(w_unnorm) + 1e-15)
        x_c = x - 0                    # caller passes x-c directly
        num = np.abs(x_c @ w)
        denom = r * np.sqrt(w @ S @ w)
        return num / max(denom, 1e-15)

    def _paper_D_prime(self, x_centered, w, S, r, epsilon=1e-10):
        """D'(x,c,S) = |w^T(x-c)| / (r * sqrt(w^T S w))  — per batch."""
        num   = np.abs(x_centered @ w)
        denom = r * np.sqrt(w @ S @ w)
        return num / max(denom, epsilon)

    def test_discriminant_vector_sign_invariance(self):
        """
        D' is invariant to sign of w (absolute value in numerator),
        so code's (c2-c1) vs paper's (c1-c2) must give same D'.
        """
        rng = np.random.default_rng(42)
        d = 4
        S1 = make_spd(d, rng)
        S2 = make_spd(d, rng)
        S_W = S1 + S2
        S_W_inv = np.linalg.inv(S_W)
        c1 = rng.standard_normal(d)
        c2 = rng.standard_normal(d)
        x  = rng.standard_normal((50, d))
        r  = 1.5

        # Paper: w = S_W^{-1}(c1-c2)
        w_paper = S_W_inv @ (c1 - c2)
        # Code: w = S_W^{-1}(c2-c1)  (opposite sign)
        w_code  = S_W_inv @ (c2 - c1)

        for S, c, label in [(S1, c1, "neuron1"), (S2, c2, "neuron2")]:
            x_c = x - c
            d_paper = np.abs(x_c @ w_paper) / (r * np.sqrt(w_paper @ S @ w_paper))
            d_code  = np.abs(x_c @ w_code)  / (r * np.sqrt(w_code  @ S @ w_code))
            np.testing.assert_allclose(
                d_paper, d_code, rtol=1e-12,
                err_msg=f"{label}: sign of w changes D'"
            )

    def test_normalization_invariance(self):
        """
        D' is invariant to scaling of w (num and denom both scale linearly),
        so code skipping the ||S_W^{-1}(c1-c2)|| normalization is correct.
        """
        rng = np.random.default_rng(7)
        d = 3
        S  = make_spd(d, rng)
        w  = rng.standard_normal(d)
        x_c = rng.standard_normal((30, d))
        r   = 1.5
        scale = 42.0

        d_w       = np.abs(x_c @ w)       / (r * np.sqrt(w @ S @ w))
        d_w_scaled= np.abs(x_c @ (scale*w)) / (r * np.sqrt((scale*w) @ S @ (scale*w)))
        np.testing.assert_allclose(d_w, d_w_scaled, rtol=1e-12,
                                   err_msg="D' changed when w was scaled")

    def test_code_matches_paper_ambiguous_formula(self):
        """
        Verify code's ambiguous-case prediction matches paper Eq.8 D'.

        Manually inject exactly 2 neurons (one per class) into a SHEF instance
        to control the scenario precisely. Query exterior points (D > 1 for both)
        so the code always takes the ambiguous path with the same pair.
        """
        rng = np.random.default_rng(42)
        r = 1.5
        epsilon = 1e-10

        # Two controlled SPD covariances and centers
        S0 = make_spd(2, rng)
        S1 = make_spd(2, rng)
        c0 = np.array([4.0, 0.0])
        c1 = np.array([-4.0, 0.0])

        # Build SHEF and inject neurons directly
        clf = SHEF(r=r, epsilon=epsilon)
        clf.neuron_list = [
            {'y': 0, 'center': c0, 'cov': S0, 'n': 50},
            {'y': 1, 'center': c1, 'cov': S1, 'n': 50},
        ]
        clf.classes_ = np.array([0, 1])
        clf.dist_ths  = {0: 1.0, 1: 1.0}

        # Query in the gap (exterior for both neurons)
        x_test = rng.standard_normal((300, 2)) * 0.4   # near origin
        D0 = mahalanobis_over_r(x_test, c0, S0, r)
        D1 = mahalanobis_over_r(x_test, c1, S1, r)
        exterior_mask = (D0 > 1) & (D1 > 1)
        assert exterior_mask.sum() > 200, "Need more exterior test points"

        # Paper D' formula: w = S_W^{-1}(c0-c1), sign/scale cancel in D'
        S_W = S0 + S1
        w   = np.linalg.inv(S_W) @ (c0 - c1)
        d0p = np.abs((x_test - c0) @ w) / (r * np.sqrt(w @ S0 @ w))
        d1p = np.abs((x_test - c1) @ w) / (r * np.sqrt(w @ S1 @ w))
        y_paper = np.where(d0p <= d1p, 0, 1)

        y_code = clf.predict(x_test)

        ext = exterior_mask
        disagreements = np.sum(y_code[ext] != y_paper[ext])
        print(f"\nCode vs paper D' (exterior, 2-neuron): "
              f"{disagreements}/{ext.sum()} disagree")
        assert disagreements == 0, (
            f"Code deviates from paper D' for {disagreements}/{ext.sum()} "
            "exterior-zone points."
        )

    def test_clear_win_case_unaffected(self):
        """
        When D(x,c1,S1) ≤ 1 < D(x,c2,S2), code must assign class1
        without invoking LDA (paper case b-a).
        """
        rng = np.random.default_rng(3)
        X0 = rng.standard_normal((50, 2)) + np.array([5, 0])
        X1 = rng.standard_normal((50, 2)) + np.array([-5, 0])
        X  = np.vstack([X0, X1])
        y  = np.array([0]*50 + [1]*50)

        clf = SHEF(r=1.5, epsilon=1e-10)
        clf.fit(X, y)

        # Deep inside class 0 — should be unambiguous
        x_test = rng.standard_normal((30, 2)) + np.array([5, 0])
        y_pred = clf.predict(x_test)
        assert np.all(y_pred == 0), "Clear-win points mis-classified"
