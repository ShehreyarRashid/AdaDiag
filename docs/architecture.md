# Architecture notes

## Where AdaDiag sits

TVDiag's forward pass is unchanged. The weighting module reads the graph-level embeddings
the encoders already produce (detached) plus three alert counts, and emits a 3×2 matrix that
scales the six task-oriented contrastive terms. Nothing it produces reaches the heads, so
the inference path is byte-for-byte TVDiag's.

```
                    ┌────────────── training only ──────────────┐
X (3,N,128) ──► GraphSAGE ×3 ──► max-pool ──► F ∈ ℝ⁹⁶ ──┬──► concat ──► RCL head
A (N,N)     ──►                                          │              FTI head
                                                         │
                                            detach ──────┘
                                              + c ∈ ℝ³
                                                 │
                                         MLP 99→64→6
                                                 │
                                  softmax per task column
                                                 │
                                          w ∈ ℝ³ˣ²
                                                 │
                            6 × SupCon(f_m, y_t) · w[m,t]  ──► l_to
```

## Design decisions, and why

**Dense tensors instead of DGL.** Incident graphs have 10–21 nodes. A batched
`(B, N, N) @ (B, N, F)` matmul is faster than a graph library's gather/scatter at this size,
removes the single worst dependency-pinning problem in reproducing TVDiag, and makes the
GraphSAGE update inspectable in four lines. `dgl.SAGEConv(aggregator_type='mean', bias=False)`
is exactly `W_self·h + W_neigh·mean(h_in)`.

**Per-sample contrastive loss.** TVDiag's `SupConLoss` reduces over the batch, so a
per-incident weight cannot multiply it. `losses.supcon_per_sample` returns the per-sample
term and the non-zero count; summing and dividing reproduces the original scalar exactly.
`tests/test_supcon_equivalence.py` asserts this on four shapes — it is the first thing to
run after any change, because a silent divergence there invalidates every arm including the
baseline.

**Simplex columns, not free weights.** See README §1. The short version: σ already sets the
magnitude, so an unconstrained **w** is unidentifiable and the optimiser can "win" by
shrinking everything.

**…but the simplex fixes the wrong degree of freedom.** This is the most important design
note in the repository and it was found by reviewing the finished code, not by planning it.
The simplex removes the *scale* ambiguity, which is real. It does nothing about the
*direction* of the incentive, because `w` is used in exactly one place — the auxiliary
contrastive term — and is deliberately kept off the prediction path so that inference stays
free. The consequence is that `∂l_rcl/∂θ_w = ∂l_fti/∂θ_w = 0` **exactly**: no diagnosis
gradient ever reaches the weighting module. And since `l_to` is *linear* in `w`, a linear
objective on a simplex is minimised at a **vertex** — so one-hot collapse is the solution
and a graded allocation is not. Both facts are asserted in
`tests/test_weighting_identifiability.py` and measured by `identifiability.py`; README §6.6
has the numbers. Anyone extending this work should read that section first: the obvious
extensions (deeper MLP, better descriptor, more conditioning) all leave both problems
untouched.

**Detached descriptor.** The weighting module should choose an allocation given the
incident, not reshape the encoders so its own job gets easier. Removing the detach makes the
module able to co-adapt with the representation, which conflates the two effects.

**Masking after augmentation.** Dropped instances are excluded from the max-pool *and* from
the RCL softmax. Leaving them in the softmax lets a node that is not in the graph compete as
a candidate root cause — a bug that inflates HR@k.

## What is deliberately *not* fixed

`AutomaticWeightedLoss` is reproduced **inert**, because that is what TVDiag ships (see
README §7.2). Making it learnable is a one-argument change
(`AutomaticWeightedLoss(4, learnable=True)`) but it is no longer a reproduction, so it is
off by default and is not used in any reported arm.
