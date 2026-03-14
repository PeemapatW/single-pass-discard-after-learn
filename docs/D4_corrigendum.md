# Corrigendum: Theorem 2 of the D4 Paper

**Paper:** Wongsriphisant, P., Plaimas, K., & Lursinsap, C. (2026). Markov-based continuous
learning with diversion of data distribution direction for streaming data in limited memory.
*Expert Systems With Applications*, 298, 129818.
DOI: [10.1016/j.eswa.2025.129818](https://doi.org/10.1016/j.eswa.2025.129818)

> This notice addresses a mathematical error in Theorem 2 of the originally published paper
> and provides the corrected theoretical mechanism underlying the D4 algorithm.
> **This theoretical correction does not affect the algorithm's implementation,
> functionality, or the experimental results reported in the paper.**

---

## 1. Correction to Theorem 2

In the original paper, the proof for Theorem 2 contains an algebraic sign error in the
final steps. When taking the reciprocal and subtracting from a constant, the inequality
sign must be reversed. Consequently, the original conclusion stating that the distance
between the projected hyper-ellipsoids increases
(d(Ω̂_a, Ω̂_b) ≥ d(Ω_a, Ω_b)) is incorrect.

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
scores. Let ψ_a(**x**) be the original metric for a hyper-ellipsoid Ω_a, and ψ'_a(**x**)
be the projected metric after selecting a subset of eigenvector indices D_a.

The original metric can be partitioned into the projected metric and a non-negative
*removed penalty* term Δ_a(**x**) representing the summation over the discarded axes (D'_a):

```
Δ_a(x) = Σ_{j ∈ D'_a}  ((x − c_a)ᵀ u_{a,j})² / w²_{a,j}
```

Since w²_{a,j} > 0, it is guaranteed that Δ_a(**x**) ≥ 0. The relationship between the
scores simplifies to:

```
ψ_a(x)  =  ψ'_a(x)  +  Δ_a(x)
```

It follows directly that the projected score is always less than or equal to the original
score (ψ'_a(**x**) ≤ ψ_a(**x**)).

The effectiveness of D4 arises because this reduction is **not uniform** for both classes.
In the overlap region (where ψ_a < 0 and ψ_b < 0), the algorithm substitutes the
ambiguous decision arg min(ψ_a, ψ_b) with a more robust decision:

```
arg min(ψ'_a, ψ'_b)  ≡  arg min(ψ_a − Δ_a, ψ_b − Δ_b)
```

A central design goal of D4 is to enhance the score separability,
|ψ'_a(**x**) − ψ'_b(**x**)|. This is achieved when the *differential penalty*
|Δ_a(**x**) − Δ_b(**x**)| amplifies the original separation.

The D4 **Compactness criterion** serves as the primary heuristic for achieving favorable
differential penalties. By prioritizing the selection of eigenvectors corresponding to the
smallest eigenvalues, it guarantees that the discarded axes (D'_a) are those with the
largest widths (w²_{a,j}). Maximizing the denominators in the penalty term systematically
reduces the per-unit-projection contribution of each removed axis. This geometric criterion
exploits the natural spread of the classes to create asymmetric penalties, transforming an
ambiguous classification into a decisive one.

---

## 3. Impact Summary

| Component | Status | Reason |
|-----------|--------|--------|
| Algorithm implementation | Unaffected | Theorem 2 concerns analysis only |
| Experimental results | Unaffected | No algorithmic change |
| Theorem 1 (Parallelism) | Valid | Independent proof |
| Theorem 3 (Closeness) | Valid | Uses triangle inequality only |
| **Theorem 2 (Eq. 10)** | **Replaced** | **See Section 2** |

The corrected theoretical framework provides a more precise characterization of the D4
mechanism. All conclusions drawn from the experimental evaluation in the original paper
remain valid.
