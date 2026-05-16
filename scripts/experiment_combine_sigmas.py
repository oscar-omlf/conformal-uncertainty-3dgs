"""
Combine σ_color (per-pixel RGB variance) with σ_v2_floater (alpha-composited
v2_rank_mult floater score) and re-evaluate via conformal.

Hypothesis: color catches *per-pixel* uncertainty (Gaussians disagree on color
at this pixel), floater catches *per-Gaussian* uncertainty (this Gaussian is a
floater). They should fire on different pixels — combining them via max should
beat color alone.

The σ maps live on very different scales, so we median-normalize each before
combining. Conformal picks q̂ for the combined σ from scratch.

We try a few combination rules and report all of them in one table:
  max     = max(σ_color_n, σ_v2_n)
  mean    = (σ_color_n + σ_v2_n) / 2
  sumsq   = sqrt(σ_color_n² + σ_v2_n²)
  gmean   = sqrt(σ_color_n · σ_v2_n + eps)
And as baselines for comparison:
  color_alone (re-evaluated through this same script for sanity)
  v2_alone
  max with visibility (instead of v2)

Writes combined σ maps to raw_sigma/<combo_name>/<frame>.npz with key
'sigma_mean' (matching the renderer convention) so we can re-use run_one().
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conformal_random_split import run_one


def load_sigma_frame(split_dir, modality, frame_stem, key):
    npz = np.load(split_dir / "raw_sigma" / modality / f"{frame_stem}.npz")
    sigma = np.asarray(npz[key], dtype=np.float32)
    if sigma.ndim == 3:
        sigma = sigma.mean(axis=0) if sigma.shape[0] in (1, 3) else sigma.mean(axis=-1)
    mask = npz["mask"].astype(bool) if "mask" in npz.files else np.ones_like(sigma, dtype=bool)
    return sigma, mask


def collect_global_median(run_dir, iteration, frame_stems, modality, key, eps=1e-6):
    """Median of σ over all calib + test pixels (across all frames). Used as a
    scale normalizer so different modalities are comparable before combining."""
    vals = []
    for split in ("calib", "test"):
        for stem in frame_stems[split]:
            sd = run_dir / split / f"ours_{iteration}"
            s, m = load_sigma_frame(sd, modality, stem, key)
            v = s[m & np.isfinite(s)]
            if v.size:
                vals.append(v)
    if not vals:
        raise RuntimeError(f"No σ values for {modality}/{key}")
    return max(float(np.median(np.concatenate(vals))), eps)


def list_frame_stems(run_dir, iteration):
    out = {}
    for split in ("calib", "test"):
        d = run_dir / split / f"ours_{iteration}" / "render"
        stems = sorted(p.stem for p in d.glob("*.png"))
        out[split] = stems
    return out


def combine_and_save(run_dir, iteration, combo_name, rule_fn, components, frame_stems, normalizers):
    """For every calib+test frame, compute σ_combined = rule_fn(*components) and
    save to raw_sigma/<combo_name>/<frame_stem>.npz with key 'sigma_mean'."""
    for split in ("calib", "test"):
        out_dir = run_dir / split / f"ours_{iteration}" / "raw_sigma" / combo_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in frame_stems[split]:
            sd = run_dir / split / f"ours_{iteration}"
            sigmas, masks = [], None
            for (modality, key) in components:
                s, m = load_sigma_frame(sd, modality, stem, key)
                s = s / normalizers[(modality, key)]
                sigmas.append(s)
                masks = m if masks is None else (masks & m)
            sigma_combined = rule_fn(sigmas)
            sigma_combined = np.nan_to_num(sigma_combined, nan=0.0, posinf=0.0, neginf=0.0)
            np.savez_compressed(
                out_dir / f"{stem}.npz",
                sigma_mean=sigma_combined.astype(np.float32),
                mask=masks.astype(np.bool_),
            )


def main():
    parser = ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    frame_stems = list_frame_stems(run_dir, args.iteration)
    print(f"Found calib frames: {len(frame_stems['calib'])}, test frames: {len(frame_stems['test'])}")

    # Components to consider. Each is (modality_folder, sigma_key_in_npz).
    color_comp = ("color", "color_std")
    v2_comp = ("floater_v2_rank_mult", "sigma_mean")
    vis_comp = ("visibility", "sigma_mean")

    # Global medians (calib+test pooled) as normalizers.
    print("Computing per-modality medians (calib+test pooled) for scale normalization ...")
    normalizers = {}
    for comp in (color_comp, v2_comp, vis_comp):
        m = collect_global_median(run_dir, args.iteration, frame_stems, *comp, eps=args.eps)
        normalizers[comp] = m
        print(f"  {comp[0]:30s}/{comp[1]:14s}: median = {m:.4g}")

    # Define experiments: name -> (combination_rule, list_of_components)
    rules = {
        "color_alone":             (lambda S: S[0],                                                                          [color_comp]),
        "v2_alone":                (lambda S: S[0],                                                                          [v2_comp]),
        "visibility_alone":        (lambda S: S[0],                                                                          [vis_comp]),

        "color_x_v2_max":          (lambda S: np.maximum(S[0], S[1]),                                                        [color_comp, v2_comp]),
        "color_x_v2_mean":         (lambda S: 0.5 * (S[0] + S[1]),                                                            [color_comp, v2_comp]),
        "color_x_v2_sumsq":        (lambda S: np.sqrt(S[0]**2 + S[1]**2),                                                    [color_comp, v2_comp]),
        "color_x_v2_gmean":        (lambda S: np.sqrt(np.maximum(S[0] * S[1], 0.0) + args.eps),                              [color_comp, v2_comp]),

        "color_x_vis_max":         (lambda S: np.maximum(S[0], S[1]),                                                        [color_comp, vis_comp]),
        "color_x_vis_mean":        (lambda S: 0.5 * (S[0] + S[1]),                                                            [color_comp, vis_comp]),
        "color_x_vis_x_v2_max":    (lambda S: np.maximum(np.maximum(S[0], S[1]), S[2]),                                      [color_comp, vis_comp, v2_comp]),
        "color_x_vis_x_v2_mean":   (lambda S: (S[0] + S[1] + S[2]) / 3.0,                                                    [color_comp, vis_comp, v2_comp]),
    }

    results = {}
    for combo_name, (rule_fn, comps) in rules.items():
        print(f"\n--- {combo_name} ({len(comps)} component(s)) ---")
        combine_and_save(run_dir, args.iteration, combo_name, rule_fn, comps, frame_stems, normalizers)
        r = run_one(
            modality=combo_name,
            sigma_key="sigma_mean",
            run_dir=run_dir,
            iteration=args.iteration,
            alpha=args.alpha,
            calib_n=30,
            calib_sample_ratio=0.1,
            seed=args.seed,
            eps=args.eps,
        )
        cov = r["test_pixel_coverage"]
        w = r["test_mean_full_width"]
        corr = r["mean_per_view_ae_uncertainty_corr"]
        ause = r.get("mean_per_view_ause", 0.0)
        print(f"  -> cov={cov:.4f}  mean_2u={w:.4f}  AE_corr={corr:+.4f}  AUSE={ause:.4f}")
        results[combo_name] = {
            "coverage": cov,
            "mean_2u": w,
            "AE_corr": corr,
            "AUSE": ause,
        }

    print(f"\n\n{'='*82}")
    print(f"SIGMA-COMBINATION ABLATION (alpha={args.alpha}, target cov={1-args.alpha:.2f}, seed={args.seed})")
    print(f"{'='*82}")
    print(f"{'combo':28} {'cov':>8} {'mean_2u':>12} {'AE_corr':>12} {'AUSE':>10}")
    print("-" * 82)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["AE_corr"]):
        print(f"{name:28} {r['coverage']:8.4f} {r['mean_2u']:12.4f} {r['AE_corr']:+12.4f} {r['AUSE']:10.4f}")

    out_path = run_dir / "uncertainty" / "combine_sigmas_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"normalizers": {f"{m}/{k}": v for (m, k), v in normalizers.items()}, "results": results}, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
