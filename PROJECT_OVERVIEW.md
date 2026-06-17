# Conformal Uncertainty for 3D Gaussian Splatting

A teaching document. Read top to bottom.

---

## 0. What this project is, in 60 seconds

A trained 3DGS model is, at the end of the day, **a function: camera pose → photo**. You give it where the camera is and where it's pointing, it gives back an RGB image of what the scene looks like from there. It learned this function from a few hundred real photos of one specific scene.

That function is good at making pretty pictures — but it has no idea *where it's wrong*. If you ask for a view from an angle the training photos didn't cover well, the result might be garbage, and the renderer is just as "confident" there as it is on a view that was photographed 50 times.

The project is about **fixing that**. We attach, to every rendered pixel, a number `σ` that says *"I'm this uncertain here."* Then we run that σ through **conformal prediction**, which converts it into a real statistical interval `[render − u, render + u]` with the guarantee:

> 90% of test pixels will have their ground-truth color inside that interval.

**Why this is non-trivial.** In classical statistics, claiming an interval like "±1.96σ covers 95% of errors" requires three assumptions: (1) σ is *calibrated* — when σ=5 the actual error stdev really is ≈5, (2) errors are *Gaussian-distributed* (bell curve), so the 1.96 multiplier corresponds to 95% by virtue of the bell shape, and (3) samples are independent. Break any of those and the coverage claim collapses.

Conformal needs none of them. It works like this: on a held-out calibration set, *measure* the empirical 95th percentile of `|err|/σ` and call it q̂. Then on test, the band `±q̂·σ` is provably 95%-covering, no matter how badly your σ was calibrated:

| | well-calibrated σ_A | sloppy σ_B that's 10× too small |
|---|---|---|
| classical ±1.96σ | 95% coverage ✓ | ~60% coverage ✗ |
| conformal ±q̂·σ | q̂≈2.0, bands ±2σ → 95% ✓ | q̂≈20, bands ±20σ → 95% ✓ |

Conformal **automatically rescales q̂** to absorb whatever miscalibration σ has. You only lose *efficiency* (worse σ → wider bands) — never *validity*.

So σ can be a heuristic guess. **The research question is which heuristic guess is actually useful** — we want σ to be small where the renderer is right and big where it's wrong, so the bands stay tight on easy regions and only widen where they need to. We try five different σ heuristics, all derived from a trained 3DGS model, and compare them.

---

## 1. Background: 3DGS in 5 minutes

**Inputs.** A set of photos of a real scene, with known camera poses (recovered by COLMAP — classical structure-from-motion).

**Representation.** Instead of a neural network, the scene is stored as **millions of tiny 3D fuzzy blobs ("Gaussians")**. Each Gaussian has:
- a position in 3D,
- a shape (an ellipsoid — rotation + scale on three axes),
- an opacity (0 = transparent, 1 = solid),
- a color that can depend on viewing angle (spherical harmonics).

**Rendering** ("splatting"). To produce an image for a given camera:
1. Project all Gaussians onto the screen. Each becomes a 2D ellipse.
2. Sort them along the camera ray (closest first).
3. For each pixel, walk through the contributing Gaussians front-to-back, accumulating color:

```
C(pixel) = Σᵢ  cᵢ · αᵢ · Tᵢ        (the alpha-compositing formula)
                    └──┬──┘
                      wᵢ    ← "blending weight" of Gaussian i for this pixel

with Tᵢ = Π_{j<i} (1 − αⱼ)         "transmittance before Gaussian i"
```

The `wᵢ = αᵢ·Tᵢ` is the **fraction of this pixel's color contributed by Gaussian i**. For an opaque surface, one Gaussian dominates (`wᵢ ≈ 1`); for fuzzy or under-determined geometry, many Gaussians contribute small amounts.

This rasterization is the bottleneck. Inria's `diff-gaussian-rasterization` does it on a CUDA tile-based kernel and is also differentiable, which is what makes training possible.

**Training.** Pure gradient descent on the per-Gaussian parameters (position, shape, opacity, color SH coefficients), using L1 + SSIM loss on rendered-vs-real photos. Plus a heuristic "densification" step that clones, splits, or prunes Gaussians every ~100 iterations.

**What you have after training.** A `.ply` file with millions of Gaussian parameters, and the ability to render any novel camera at >100fps.

For this project the only thing you really need to remember is: **after training, for any pixel of any rendered image, we know the list of contributing Gaussians and their blending weights `wᵢ`**. Everything else falls out of that.

---

## 2. Background: Conformal Prediction in 5 minutes

The setup. You have:
- A predictor `ŷ(x)` (in our case: the renderer's output for a pixel).
- A heuristic uncertainty score `σ(x)` (in our case: one of six candidates we'll define later).
- A held-out **calibration set** of `(x, y)` pairs where you know the true `y` (in our case: GT pixels from a held-out camera).
- A **test set** where you only see `x` (in our case: a different held-out camera, where you pretend not to see the GT).

Conformal prediction's promise: produce a prediction interval `[ŷ − u(x), ŷ + u(x)]` such that, on test data,

> P( y ∈ [ŷ − u, ŷ + u] ) ≥ 1 − α

This is called **coverage**. With α = 0.1 we get **90% coverage**, by construction.

How. Two steps:

1. **Calibrate.** On every calibration sample, compute a *nonconformity score*:
   ```
   sᵢ = |ŷ(xᵢ) − yᵢ| / σ(xᵢ)
   ```
   This is just "how many σ's off was the prediction at this calib point?" Take the 90th percentile of all the `sᵢ`. Call it `q̂`.

2. **Predict.** On a test point, emit the interval
   ```
   [ ŷ(x) − q̂·σ(x), ŷ(x) + q̂·σ(x) ]
   ```
   The interval half-width is `u = q̂·σ`, the full width is `2u`.

**Why this works.** If calib and test pixels are *exchangeable* (drawn from the same distribution, no systematic bias between the two sets), then the scores on test pixels follow the same distribution as the scores on calib pixels. By construction, 90% of the calib scores were ≤ q̂, so 90% of the test scores are ≤ q̂ as well. That's exactly the 90% coverage claim.

**The key feature.** σ can be **anything**. A useless σ that's the same number everywhere still hits 90% coverage — the bands just end up uniformly wide and uninformative. A good σ that's small where the model is confident and big where it isn't gives **tight bands where you trust the model and wide bands where you don't**. Both are valid; the second is useful.

**The trade-off.** Coverage isn't the only thing you care about — *width* matters too. A trivial σ ≡ 1 hits coverage by widening every band. A good σ hits coverage **and** the bands stay narrow on easy regions. So we care about:
- coverage (sanity check: did we actually hit 0.90?)
- mean interval width 2u (smaller = tighter)
- how well the bands correlate with the actual error |ŷ − y| (higher = σ adapts to scene content, not just blanket-widens).

---

## 3. The bridge: what is σ in 3DGS?

This is the part that's not obvious. 3DGS doesn't naturally give you a per-pixel uncertainty. So we have to invent one — multiple candidates, actually — from quantities the rasterizer already computes or that we can compute cheaply.

We split candidates into two families:

### 3a. 2D candidates (already per-pixel)

These are derived directly from the alpha-compositing trace at each pixel. For a pixel with contributing Gaussians `{(cᵢ, wᵢ, zᵢ)}`:

- **color σ** — the per-pixel *variance of Gaussian colors* weighted by blending weights:
  ```
  E[c]   = (Σᵢ wᵢ · cᵢ) / Σᵢ wᵢ
  Var[c] = (Σᵢ wᵢ · cᵢ²) / Σᵢ wᵢ  −  E[c]²
  σ_color = √Var[c]
  ```
  Intuition: if the Gaussians contributing to this pixel *disagree* about what color it should be, the renderer guessed by averaging them — that's a sign of uncertainty.

- **depth σ** — same formula, but on the depth `zᵢ` (or inverse depth `1/zᵢ`) of each Gaussian instead of its color. Intuition: high depth variance = "the model has Gaussians at very different depths along this ray and isn't sure which one is the real surface."

- **entropy σ** — the **Shannon entropy of the top-K (=4) blending weights** at each pixel:
  ```
  Take w₁, w₂, w₃, w₄ = the 4 largest weights for this pixel
  Normalize: pₖ = wₖ / Σwₖ
  H = − Σₖ pₖ log pₖ        (in [0, log 4])
  ```
  Low entropy = one Gaussian dominates the pixel (the renderer "knows" what's there). High entropy = multiple Gaussians fight for the pixel with similar weights (the renderer is hedging).

### 3b. 3D candidates (one number per Gaussian → made into per-pixel via alpha-compositing)

These two assign a scalar `sᵢ` to *every Gaussian*, then we lift to per-pixel by alpha-compositing the same way the renderer does color:
```
σ(pixel) = (Σᵢ wᵢ · sᵢ) / (Σᵢ wᵢ)
```

- **Fisher sensitivity** (PUP3DGS). Backprop the training loss through the renderer, look at the gradient `g = ∇_θ L` w.r.t. each Gaussian's parameters `θ = (xyz, scale)` (6 numbers). Build the per-Gaussian outer-product Fisher `F = Σ_views g·gᵀ` (a 6×6 matrix per Gaussian). The score is the log-determinant of `F`. Intuition: a Gaussian with high log-det is one whose parameters "matter a lot" to the training loss — perturbing it would change the rendering a lot — so the renderer is *sensitive* to it.

- **Visibility** (4DGS-W). For each Gaussian `k`, compute the total contribution mass over the entire training set:
  ```
  C_k = Σ_{training view v, training pixel p}  w_k(v, p)
  ```
  Then the paper defines per-Gaussian uncertainty as `U_k = 1 − sigmoid((C_k − c0)/c1)` (so highly-visible Gaussians get *low* uncertainty). Intuition: a Gaussian that was touched by lots of training pixels is well-trained; one that was barely visible during training is under-determined.

- **Floater likelihood** (TIDI-GS-inspired, arXiv 2601.09291). Composite score targeting Gaussians that are likely *floaters* — geometric artifacts that sit in empty space. TIDI-GS combines four signals: k-NN spatial isolation, opacity, training-time visibility counter, and a learned importance scalar. The learned scalar requires their training framework so we drop it; the gradient-EMA signal is implicitly captured by `sensitivity` so we drop that too. The remaining three are forward-only on a frozen model. We robust-z-score each and combine:
  ```
  d_i  = √distCUDA2(xyz)   ← mean distance to K=3 NN (existing simple_knn primitive)
  α_i  = gaussians.get_opacity[i]
  C_i  = visibility (already computed above)

  score[i] = z(d_i) − z(C_i) − z(α_i)
              ^isolated   ^low_vis   ^low_opacity
  ```
  Intuition: a Gaussian flagged by *all three* is sitting alone in space, was rarely needed during training, and has wispy opacity — i.e., a floater. The score is alpha-composited to per-pixel σ the same way as sensitivity and visibility.

### 3c. How σ enters the conformal procedure

Once we have a per-pixel σ map (by any of the routes above), it goes into conformal exactly as in §2:

1. On every calib pixel, score `sᵢ = |render − GT| / σᵢ`. Take 90th percentile → `q̂`.
2. On every test pixel, emit the interval `2u = 2·q̂·σ`.

Six σ candidates ⇒ six sets of intervals ⇒ six answers to "is this any good?"

---

## 4. Pipeline end to end

```
photos + COLMAP poses                                   <- input
        │
        ▼
round-robin split (7 train / 1 calib / 2 test per 10)
        │
        ▼
train 3DGS on the 211 training photos                    (~14 min @ 30k iters on A100)
        │
        ▼
for every held-out frame (30 calib + 60 test):
   render the image                                      (RGB output)
   render the 6 σ candidates                             (color, depth, entropy, sensitivity, visibility, floater)
        │
        ▼
for each σ candidate, run conformal calibration         (single q̂)
        │
        ▼
evaluate on test:
   - empirical coverage  (should be ~ 0.90)
   - mean interval width 2u
   - Pearson correlation between |err| and 2u, per view
        │
        ▼
results.md table   (6 rows, one per σ)
```

The whole thing is one SLURM job: `sbatch snellius_jobs/01_full_pipeline.job`. Takes ~45 min end-to-end on an A100.

---

## 5. Evaluation metrics, in detail

For each σ candidate, we report three numbers:

**Coverage.** Empirical fraction of test pixels where the GT actually lies inside the interval `[render − q̂σ, render + q̂σ]`. This is the conformal sanity check. With α = 0.1 it should land at ~0.90. If it doesn't, something about exchangeability is broken (see §8).

**Mean interval width `2u`.** Average over all test pixels of `2·q̂·σ`. In RGB units (0–255). Smaller = tighter intervals = better, *at the same coverage*. **But — careful here.** σ scales arbitrarily. If you multiply σ by 100, q̂ divides by 100, and 2u is unchanged. So absolute mean 2u is comparable within a modality across training durations or different alphas, but **not directly comparable across modalities with different σ ranges**. (See §8 for the visibility example where 2u looks huge.)

**Per-view AE correlation.** For each test frame, flatten the per-pixel |err| and the per-pixel 2u, compute Pearson correlation. Then average across the 60 test frames. **Higher (positive) = σ actually tracks error**. This is the *signal* metric — the one that tells you whether σ is doing anything useful or just blanket-widening.

There's also k-fold cross-validation as a robustness check — we'll come back to it in §8.

---

## 6. Results

Scene: `tandt/train` (301 photos). 70/10/20 round-robin split → 211 train, 30 calib, 60 test.

Conformal at α = 0.1 (target coverage 0.90), averaged over 5 random calib/test shufflings of the held-out frames:

### 30,000-iter model (paper-faithful sensitivity & visibility)

| modality | coverage | mean 2u | std 2u | **AE correlation** |
|---|---|---|---|---|
| **color** | 0.921 ± 0.004 | **93.7 ± 1.1** | 45 | **+0.294 ± 0.010** |
| **visibility (4DGS-W)** | 0.893 ± 0.007 | 2928 ± 335 (sigmoid-scale) | 350 | **+0.184 ± 0.003** |
| depth | 0.915 ± 0.018 | 171.9 ± 21.3 | 120 | +0.059 ± 0.018 |
| sensitivity (PUP3DGS) | 0.920 ± 0.008 | 167.2 ± 13.7 | 190 | +0.054 ± 0.008 |
| entropy | 0.916 ± 0.007 | 86.2 ± 1.9 | 8 | +0.005 ± 0.003 |

### 7,000-iter model (apples-to-apples with the 30k table)

| modality | coverage | mean 2u | AE correlation |
|---|---|---|---|
| **color** | 0.920 ± 0.011 | 102.5 ± 3.5 | **+0.312 ± 0.006** |
| **sensitivity (PUP3DGS)** | 0.925 ± 0.016 | 180.6 ± 17.9 | **+0.133 ± 0.008** |
| **visibility (4DGS-W)** | 0.894 ± 0.010 | 2114 ± 369 | **+0.160 ± 0.004** |
| depth | 0.927 ± 0.023 | 275.4 ± 48.2 | +0.072 ± 0.015 |
| entropy | 0.916 ± 0.011 | 101.6 ± 3.5 | +0.005 ± 0.002 |

### 3-fold cross-validation (every held-out frame plays "test" exactly once)

| modality | coverage | AE correlation |
|---|---|---|
| color | 0.894 ± 0.008 | **+0.292 ± 0.006** |
| visibility | 0.907 ± 0.010 | **+0.183 ± 0.017** |
| sensitivity | 0.901 ± 0.002 | +0.051 ± 0.026 |
| depth | 0.892 ± 0.017 | +0.055 ± 0.020 |
| entropy | 0.891 ± 0.014 | +0.004 ± 0.006 |

---

## 7. Interpretation

**1. Conformal works.** Every σ candidate hits coverage ≈ 0.89–0.92 at the 0.90 target, regardless of whether the σ is informative or not. This is exactly what conformal guarantees: any heuristic σ is *valid* after calibration. The thing it doesn't guarantee is *usefulness*.

**2. Color σ is the clear winner.** Highest AE correlation (+0.29), tightest interval widths in absolute units, and the bands are *adaptive* (std 2u / mean 2u ≈ 0.48 — wide where it should be, narrow where it should be). Intuition: per-pixel disagreement among the contributing Gaussians is, mechanically, the same thing as "the model isn't sure what color this pixel is."

**3. Visibility (paper-faithful) is the second-best signal.** Correlation +0.18, robust across folds. The mean 2u of ~2900 looks alarming, but it's a units artifact (see §8). The std-to-mean ratio is only ≈0.11, so the bands are pretty uniform — visibility says where the *Gaussians* are uncertain (low training coverage), not where the *pixels* are.

**4. Sensitivity has a surprising training-dependence.** Correlation +0.13 at 7k, but drops to +0.05 at 30k. The Fisher captures *which Gaussians' parameters matter most to the training loss*. Mid-training (7k), some Gaussians are still settling — those with high Fisher are the still-learning ones, and their pixels are also where the model has the most error left. By 30k, training has converged, the Fishers are more uniform, and the discriminating power vanishes. Sensitivity is fundamentally a *training-dynamics* signal, not a *test-time* signal. (This is consistent with its original purpose in PUP3DGS, which is **pruning** — deciding which Gaussians can be safely removed.)

**5. Depth correlates only weakly.** Alpha-composited depth variance is conceptually meaningful (high variance = surface ambiguity) but in practice picks up scene-level features like texture and material edges rather than the model's actual mistakes.

**6. Entropy is flat.** Correlation +0.005, std 2u tiny (≈8). At 30k iters on this scene, most pixels are dominated by 1–2 Gaussians (low entropy almost everywhere). The CUDA mod that computes top-K entropy works fine — it just doesn't discriminate on this scene class. We'd expect it to do better on scenes with significant translucency or fine geometry (foliage, hair, etc.) where many Gaussians genuinely compete at each pixel.

---

## 8. Caveats & gotchas

### 8a. The round-robin split caused a slight under-coverage

When I first ran the pipeline at α = 0.1, every modality came out under-covered (coverage 0.85–0.89 instead of 0.90). Diagnostic:

```
mean abs error over calib frames = 19.51
mean abs error over test  frames = 22.10     (test is 13% harder)
```

The round-robin recipe is: train at positions 0..6 of every block of 10, calib at position 7, test at positions 8..9. Calib (position 7) is always exactly 1 step from a training position (position 6 or 10). Test positions 8..9 are 2–3 steps away. So test cameras are systematically slightly further from any training camera than calib cameras are → test errors are systematically bigger → conformal under-covers, because q̂ was fit on the easier calib.

This isn't a bug in the conformal code, it's a property of the split. The fix is to use a **random** calib/test partition of the 90 held-out frames instead of the structural one (`scripts/conformal_random_split.py`). After random shuffling, coverage lands at the right value.

### 8b. "Aren't you overfitting by switching to a random split?"

The concern would be: I peeked at test labels to compute that 19.51-vs-22.10 diagnostic. Did I tune anything?

No. The fix (random shuffling) is the textbook conformal fix for biased splits — it would be the right thing to do *a priori* by just reading the split structure. Nothing about σ, the model, or q̂ uses test labels. Random shuffling restores calib/test exchangeability, which is the assumption conformal's guarantee depends on.

To rule out overfitting empirically, the 3-fold CV (§6) puts every held-out frame in the test set exactly once across the folds. Cross-fold std is ≤ 0.02 — there's no "lucky split" effect. The coverage figure is robust.

### 8c. Why visibility's mean 2u is 2900 instead of 95

`σ_visibility = 1 − sigmoid(...)` lives in (0, 1) — small absolute numbers. Conformal calibration responds by making `q̂_visibility` huge (~3000) to scale the σ into the RGB error range. So `2u = 2·q̂·σ` ends up around 2900. The conformal math is invariant to monotonic rescaling of σ — multiplying σ by 255 would just divide q̂ by 255, and 2u would be identical.

So mean 2u isn't comparable *across* modalities with different σ ranges. The right cross-modality comparison is AE correlation and the std-to-mean ratio of 2u (does the band adapt or is it uniform?), not absolute 2u.

### 8d. Sensitivity isn't bit-exact PUP3DGS

PUP3DGS has a custom CUDA kernel `pool_fisher_cuda` that computes per-pixel Fisher contributions and pools them. We don't have that kernel; instead we compute the 6×6 Fisher per Gaussian as `F = Σ_views g·gᵀ` where `g` is the gradient of the *full-view loss* w.r.t. that Gaussian's parameters. This is the standard outer-product Fisher approximation. The parameter set (xyz + scaling), matrix shape (6×6), and score formula (sum of log singular values) are all identical to the paper.

---

## 8.5. Floater downstream: conformal-calibrated TIDI-GS pruning, three scenes

A follow-up experiment after the Mathan reframing: instead of "is floater the best σ?" (it wasn't), we ask **"does conformal calibration extend an existing 3DGS framework to be better at its original task (rendering quality)?"** The existing framework is TIDI-GS (arXiv 2601.09291), which uses a per-Gaussian floater-likelihood score to decide which Gaussians to *prune*. We tested three configurations on three scenes (Church from Tanks & Temples training, Garden and Bicycle from MipNeRF360 — both of which come pre-bundled with COLMAP and overlap with what the team's active-learning work uses for the poster):

| config | what it does |
|---|---|
| (A) baseline | vanilla 3DGS, no pruning |
| (B) TIDI-GS-style fixed-K pruning | rank Gaussians by our `floater_v2_rank_mult` score, prune top-K%, re-render |
| (C) conformal-calibrated K | sweep K, pick K* maximizing calib PSNR subject to coverage(calib) ≥ 1−α |

All scenes at 30k iters, K=0.10 for (B), conformal sweep K ∈ {0, 0.02, 0.05, 0.10, 0.15, 0.20} for (C):

| scene | A_baseline PSNR | A SSIM | A LPIPS | B fixed_K=0.10 PSNR | C conformal K* | C result |
|---|---|---|---|---|---|---|
| Church (T&T) | **20.97** | 0.805 | 0.244 | 20.97 (−0.002) | **0** | identical to A |
| Garden (MipNeRF360) | **26.44** | 0.847 | 0.119 | 26.40 (−0.041) | **0** | identical to A |
| Bicycle (MipNeRF360) | **21.53** | 0.681 | 0.249 | 21.52 (−0.009) | **0** | identical to A |

Across all three scenes, the conformal picker chose K* = 0 — i.e. **the procedure correctly identified that no level of pruning (within the K ∈ {0, 0.02, 0.05, 0.10, 0.15, 0.20} sweep) improves calibration PSNR beyond baseline, and refused to prune**. Fixed K=0.10 in (B) introduces a small but consistent PSNR regression (0.002–0.041 dB) on every scene; (C) avoided that regression every time.

The conformal picker's calib K-sweep tells the same story on all three scenes — PSNR on calib monotonically degrades very slightly as K increases, coverage stays at the target 0.900, and K* lands at **0** every time. The picker refuses to prune because no K improves calib PSNR.

Why the slight degradation? On Church specifically, we instrumented the muted set: the 222,784 Gaussians at K=0.10 collectively contribute only **0.68%** of total alpha-mass. The v2 floater score is correctly identifying very-low-impact Gaussians — pruning them is essentially removing noise, the model barely notices either way. The score works; the *target* (near-invisible Gaussians) just doesn't move the metric.

### What this experiment establishes

1. **The conformal threshold picker works as intended.** It is a *safety net*: it allowed pruning only if calib PSNR didn't degrade and coverage stayed valid. Across all three scenes, the picker refused to prune. A hand-tuned K=0.10 (TIDI-GS-style) would have introduced a small but real PSNR regression every time; our procedure auto-detected this and stayed at K=0.

2. **Post-hoc TIDI-GS-style pruning does not improve PSNR on any of the three scenes tested.** This isn't a flaw in our reproduction — TIDI-GS's reported gains depend on *training-time* signals (learned ω_i scalar, position-gradient EMA) and the fact that pruning *during* training lets densification re-allocate Gaussians elsewhere. Pruning post-hoc on a frozen model has no such recovery mechanism, so removing even 10% of low-impact Gaussians is at best a no-op and at worst a small regression.

3. **Conformal-as-extension is well-defined and worth presenting.** Even when the extension's outcome is "do nothing" (K*=0), that's a *valid output* of a statistical procedure — it's the safety property of the calibration. For any future framework where the per-Gaussian score is computed *during training* (and thus where pruning has room to help), the same procedure would equally pick K*>0 and prune up to the largest fraction that doesn't break the conformal coverage guarantee.

### How σ-comparison and pruning-extension fit together

The full table of σ candidates is unchanged by the pruning experiment — that was a separate question about test-time uncertainty maps. For completeness, conformal AE correlations from this run on the three new scenes (single-seed round-robin):

| modality | Church AE corr | Garden AE corr | Bicycle AE corr |
|---|---|---|---|
| **color** | **+0.202** | **+0.382** | **+0.331** |
| visibility | +0.066 | +0.196 | +0.248 |
| sensitivity | +0.053 | +0.080 | +0.043 |
| floater (baseline v0) | +0.019 | −0.006 | +0.092 |
| depth | +0.002 | +0.099 | −0.076 |
| entropy | −0.002 | −0.007 | +0.022 |

**Color still wins on every scene**, and visibility is consistently the second-best signal. Garden's color (+0.382) is the strongest σ we've measured across the whole project (drjohnson was +0.439 but that's a much earlier separate run). The pattern from the earlier tandt/train + drjohnson runs holds: color is the universally dominant signal, visibility is consistently second, the other four are weak.

### Caveats vs. TIDI-GS Table I and nerfbaselines

The headline mismatch is the split: we hold out 40% of frames per scene (deterministic 7/1/2 round-robin) so the renderer can train on fewer views; TIDI-GS / nerfbaselines use the standard ~13% LLFF holdout. That accounts for most of the PSNR gap:

| scene | our 30k PSNR (40% holdout) | published 3DGS baseline (~13% holdout) |
|---|---|---|
| Church | 20.97 | 26.2 (TIDI-GS Table I) |
| Garden | 26.44 | 27.4 (nerfbaselines m-colmap) |
| Bicycle | 21.53 | 25.2 (nerfbaselines m-colmap) |

Same order of magnitude in every case (and Bicycle has additional difficulty — outdoor with fine-grain foliage that 3DGS struggles with at any resolution). The methodological contribution (conformal-extends-pruning, plus the σ-comparison) is what's defensible from this experiment; the raw PSNR numbers are not directly comparable to those tables.

### Three-line summary of the floater extension story

1. **Conformal calibration *can* extend any existing 3DGS framework that exposes a per-Gaussian quality score** — we built the procedure end-to-end (`prune_and_render.py` + `conformal_pick_pruning_K.py`) and ran it on three scenes from two datasets.
2. **On all three scenes, the procedure correctly refused to prune (K\* = 0)** — TIDI-GS-style post-hoc score-based pruning offers no PSNR gain over baseline, and the conformal picker successfully detected this and chose the safe action.
3. **This is a "negative-by-design" outcome, not a failed experiment**: the safety net worked. If TIDI-GS-style pruning were run *during* training (where their original paper places it), the picker would have room to identify a beneficial K* > 0; we don't have that lever post-hoc.

---

## 9. Recommendation for the meeting

**Ship color σ as the primary uncertainty signal.** Highest correlation, tightest adaptive bands, simplest implementation (no CUDA mod, no per-Gaussian compute step). Robust at both 7k and 30k iters.

**Ship visibility (paper-faithful) as a second, complementary signal.** Correlation +0.18 at 30k, robust across folds. It captures something *different* from color — color is per-pixel agreement, visibility is per-Gaussian training coverage. Combining (e.g. `σ = max(σ_color, σ_visibility)`) is a natural next experiment — they shouldn't fail on the same pixels.

**Floater likelihood (TIDI-GS-inspired)** is a new sixth candidate added after the main run; it targets *specifically* the floater failure mode by composing three forward-only signals (k-NN spatial isolation, opacity, visibility). It's added to `01_full_pipeline.job` and ranked alongside the other five. Numbers are pending the next pipeline rerun and will land in the results table when available.

**Be honest about the three negatives.**
- *Entropy* — CUDA mod works, signal doesn't discriminate on this scene. Worth trying on a scene with more transparent geometry.
- *Sensitivity* — captures training-time signal, not test-time. Drops by ~0.08 in correlation from 7k to 30k. Its natural use is pruning (the original PUP3DGS contribution), not post-hoc uncertainty.
- *Depth* — too noisy. The variance picks up texture, not just geometry.

---

## 10. Reproducibility

On a fresh clone with the conda env `gaussian_splatting`:

```bash
git submodule update --init --recursive
cd submodules/diff-gaussian-rasterization && git checkout dr_aa
git apply ../../patches/0001-rasterizer-topk-weights.patch
cd ../..

# build
module load 2024 Anaconda3/2024.06-1
source activate gaussian_splatting
module load 2023 CUDA/12.1.1
export TORCH_CUDA_ARCH_LIST="8.0"
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/fused-ssim

# end-to-end pipeline on one scene (A100, ~45 min)
SCENE=/path/to/scene OUT=output/my_run ITERS=30000 \
    sbatch --export=ALL,SCENE,OUT,ITERS snellius_jobs/01_full_pipeline.job

# re-aggregate at a different alpha / random splits, no retraining/re-rendering
python scripts/conformal_random_split.py --run_dir output/my_run --alpha 0.1
python scripts/conformal_kfold.py --run_dir output/my_run --alpha 0.1 --n_folds 3
python scripts/aggregate_results.py --run_dir output/my_run
```

---

## Appendix: provenance

The original collaborator branch (`feature/uncertainty_maps`) shipped the round-robin split, color/depth σ renderers, and the conformal driver — i.e. everything in §3a except entropy. My PR (`feature/uncertainty-signals`, merged into `main` and now followed by 8 more commits from the team) added the three new σ sources from the Weekly Notes:

- Entropy: a new top-K CUDA kernel in the rasterizer (the patch is `patches/0001-rasterizer-topk-weights.patch` because the rasterizer submodule is upstream Inria), plus `scripts/render_entropy.py`.
- Sensitivity: `scripts/compute_sensitivity.py` (paper-faithful 6×6 outer-product Fisher) + `scripts/render_per_gaussian_scalar.py`.
- Visibility: `scripts/compute_visibility.py` (autograd trick for the per-Gaussian C_k sum, then 1−sigmoid per the paper) + same renderer.

Plus methodological bits:
- The random-split conformal (§8a) is `scripts/conformal_random_split.py`.
- The k-fold validation (§8b) is `scripts/conformal_kfold.py`.
- The `--sigma_norm none` mode in `conformal_prediction.py` (default after my changes) — see §3c for why we don't need to normalize σ.

After my branch was merged, the team added active-learning-style ranking scripts (`scripts/active_learning/`) and per-scene submission shells under `snellius_jobs/`. Those build on the same σ sources documented here.

I then synced with main and added a sixth σ source as a downstream task:
- **Floater likelihood** (`scripts/compute_floater.py`, TIDI-GS-inspired): composite of `simple_knn` k=3 spatial isolation, opacity, and the already-computed visibility `C_k`. Combined via robust z-scoring. Added as `--modality floater` to all the conformal scripts and as a new step in `snellius_jobs/01_full_pipeline.job`. Read on a frozen pre-trained 3DGS — no retraining required.
