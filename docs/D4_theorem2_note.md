# Technical Note: Theorem 2 of the D4 Paper

**Related paper:** Wongsriphisant, P., Plaimas, K., & Lursinsap, C. (2026). Markov-based continuous
learning with diversion of data distribution direction for streaming data in limited memory.
*Expert Systems With Applications*, 298, 129818.
DOI: [10.1016/j.eswa.2025.129818](https://doi.org/10.1016/j.eswa.2025.129818)

> **Note:** This correction concerns only the theoretical proof in the paper.
> It does not affect the algorithm's implementation, functionality, or experimental results.

---

## 1. Correction to Theorem 2

In the original paper, the proof for Theorem 2 contains an algebraic sign error in the
final steps. When taking the reciprocal and subtracting from a constant, the inequality
sign must be reversed. Consequently, the original conclusion stating that the distance
between the projected hyper-ellipsoids increases
(d(Ω̂\_a, Ω̂\_b) ≥ d(Ω\_a, Ω\_b)) is incorrect.

The mathematically corrected conclusion is the opposite: the projected distance is always
less than or equal to the original distance:

```
d(Ω̂_a, Ω̂_b)  ≤  d(Ω_a, Ω_b)       (Eq. 10, corrected)
```

Therefore, the resolution of classification ambiguity is not achieved by proving the
hyper-ellipsoids become further apart in terms of Definition 2, but by substituting the
original, unreliable classification scores with a new set of projected scores that exhibit
greater separability.

---

## 2. Corrected Mechanism: Differential Penalty

The theoretical justification for D4 relies on the asymmetric reduction of the metric
scores. Let ψ\_a(**x**) be the original metric for a hyper-ellipsoid Ω\_a, and ψ'\_a(**x**)
be the projected metric after selecting a subset of eigenvector indices D\_a.

The original metric can be partitioned into the projected metric and a non-negative
*removed penalty* term Δ\_a(**x**) representing the summation over the discarded axes (D'\_a):

```
Δ_a(x) = Σ_{j ∈ D'_a}  ((x − c_a)ᵀ u_{a,j})² / w²_{a,j}
```

Since w²\_{a,j} > 0, it is guaranteed that Δ\_a(**x**) ≥ 0. The relationship between the
scores simplifies to:

```
ψ_a(x)  =  ψ'_a(x)  +  Δ_a(x)
```

It follows directly that the projected score is always less than or equal to the original
score (ψ'\_a(**x**) ≤ ψ\_a(**x**)).

In the overlap region (where ψ\_a < 0 and ψ\_b < 0), the algorithm substitutes the
ambiguous decision arg min(ψ\_a, ψ\_b) with a more robust decision:

```
arg min(ψ'_a, ψ'_b)  ≡  arg min(ψ_a − Δ_a, ψ_b − Δ_b)
```

In particular, this differential penalty mechanism becomes most decisive in cases where
the original scores are nearly indistinguishable (ψ\_a(**x**) ≈ ψ\_b(**x**)), as the
classification outcome becomes predominantly determined by the difference
Δ\_a(**x**) − Δ\_b(**x**).

To ensure this penalty reliably breaks ties, the D4 algorithm selects axes using a
two-stage priority:

1. **Parallelism Criterion:** Pairs of eigenvectors from both classes that are highly
   aligned (angle ≤ θ) are selected first. This establishes a comparable subspace where
   the geometric structures share similar orientations.
2. **Compactness Criterion:** The remaining dimensions are filled by selecting the
   narrowest axes (smallest eigenvalues). This guarantees that the discarded axes (D'\_a)
   are those with the largest widths (w²\_{a,j}), maximizing the denominators in the
   penalty term.

As a consequence, the discarded axes tend to capture directions where the two classes are
structurally most divergent (non-parallel) and exhibit their largest geometric spreads.
Because these discarded components, selected for their structural divergence and large
geometric spreads, generally differ between classes, the penalty reductions tend to be
asymmetric (Δ\_a(**x**) ≠ Δ\_b(**x**)), effectively resolving the classification ambiguity.

---

## 3. Impact Summary

| Component | Status | Reason |
|-----------|--------|--------|
| Algorithm implementation | Unaffected | Theorem 2 concerns analysis only |
| Experimental results | Unaffected | No algorithmic change |
| Theorem 1 (Parallelism) | Valid | Independent proof |
| Theorem 3 (Closeness) | Valid | Uses triangle inequality only |
| **Theorem 2 (Eq. 10)** | **Corrected** | **See Section 2** |

The corrected theoretical framework provides a more precise characterization of the D4
mechanism. All conclusions drawn from the experimental evaluation in the original paper
remain valid.
