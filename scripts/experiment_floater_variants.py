"""
Floater-score ablation: try several per-Gaussian aggregation formulas, render
each to per-pixel σ, run conformal on each, print a summary table.

Variants:
  V0 (baseline): z(d) - z(C) - z(α)           (the original additive z-score)
  V1: rank(d) + (1 - rank(C)) + (1 - rank(α))  (additive percentile ranks)
  V2: rank(d) * (1 - rank(C)) * (1 - rank(α))  (multiplicative AND)
  V3: z(log d) + z(-C) + z(-α)                 (log-transform isolation first)
  V4: (1 - sigmoid((C-c0)/c1)) * rank(d)       (visibility-uncertainty boosted by isolation rank)
  V5: (1 - sigmoid((C-c0)/c1)) * rank(d) * (1 - rank(α))  (V4 with opacity)

The motivating diagnostic: spatial isolation has a max/median ratio of ~4700,
so z-score lets it dominate the additive sum. Percentile-rank kills that tail.
V4/V5 explicitly build on the visibility signal that already gets +0.18 AE corr.

Renders each variant to raw_sigma/floater_v<N>/ under the run dir, then runs
in-memory conformal via conformal_random_split.run_one().
"""

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conformal_random_split import run_one


def percentile_rank(values):
    """Map values to [0, 1] using their empirical CDF rank (handles ties as midranks)."""
    n = values.size
    order = np.argsort(values)
    rank = np.empty(n, dtype=np.float32)
    # midrank for ties
    rank[order] = np.arange(1, n + 1, dtype=np.float32) / float(n)
    return rank


def robust_z(values, mad_eps=1e-6):
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = max(1.4826 * mad, mad_eps)
    return ((values - med) / scale).astype(np.float32)


def fit_sigmoid_params(C):
    c0 = float(np.median(C))
    c1 = max(float(np.subtract(*np.percentile(C, [75, 25])) / 2.0), 1e-6)
    return c0, c1


def compute_variants(d, C, alpha, mad_eps=1e-6):
    """Return dict {name: per_gaussian_score}. Higher score = more floater-likely."""
    z_d = robust_z(d, mad_eps)
    z_C = robust_z(C, mad_eps)
    z_a = robust_z(alpha, mad_eps)

    rank_d = percentile_rank(d)
    rank_C = percentile_rank(C)
    rank_a = percentile_rank(alpha)

    c0, c1 = fit_sigmoid_params(C)
    U_vis = (1.0 - 1.0 / (1.0 + np.exp(-(C - c0) / c1))).astype(np.float32)

    z_log_d = robust_z(np.log(np.clip(d, 1e-12, None)), mad_eps)

    return {
        "v0_baseline": z_d - z_C - z_a,
        "v1_rank_add": rank_d + (1.0 - rank_C) + (1.0 - rank_a),
        "v2_rank_mult": rank_d * (1.0 - rank_C) * (1.0 - rank_a),
        "v3_logd_z": z_log_d + (-z_C) + (-z_a),
        "v4_vis_iso": U_vis * rank_d,
        "v5_vis_iso_op": U_vis * rank_d * (1.0 - rank_a),
    }


def shift_positive(arr, eps=1e-6):
    arr = arr.astype(np.float32)
    return (arr - float(arr.min()) + eps).astype(np.float32)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="e.g. /scratch-shared/.../output/train")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Conformal run dir (defaults to --model_path). Must contain calib/ and test/ split dirs.")
    parser.add_argument("--split_dir", default=None, type=str,
                        help="Defaults to <model_path>/splits")
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--variants", default="", type=str,
                        help="Comma-separated list of variants to run; empty = all")
    parser.add_argument("--skip_render", action="store_true",
                        help="Re-use existing per-frame .npz under raw_sigma/<modality>/ "
                             "if already rendered (for re-running conformal only)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    run_dir = Path(args.run_dir) if args.run_dir else model_path
    split_dir = Path(args.split_dir) if args.split_dir else (model_path / "splits")

    # Load the per-Gaussian components from the prior floater.npz (saves recomputing distCUDA2).
    floater_path = model_path / "uncertainty" / "floater.npz"
    if not floater_path.exists():
        raise SystemExit(f"Missing {floater_path}. Run compute_floater.py first.")
    d_floater = np.load(floater_path)
    d_iso = np.asarray(d_floater["spatial_isolation"], dtype=np.float32)
    C_vis = np.asarray(d_floater["visibility"], dtype=np.float32)
    a_op = np.asarray(d_floater["opacity"], dtype=np.float32)
    print(f"Loaded P={d_iso.size} per-Gaussian components.")

    variants = compute_variants(d_iso, C_vis, a_op)
    if args.variants.strip():
        wanted = set(s.strip() for s in args.variants.split(","))
        variants = {k: v for k, v in variants.items() if k in wanted}
    print(f"Variants to evaluate: {list(variants.keys())}")

    out_dir = model_path / "uncertainty" / "floater_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for vname, vscore in variants.items():
        print(f"\n{'='*72}\n[{vname}] computing & rendering ...\n{'='*72}")
        # 1. Save per-Gaussian score to its own .npz
        vscore = np.asarray(vscore, dtype=np.float32).reshape(-1)
        score_shifted = shift_positive(vscore)
        npz_path = out_dir / f"{vname}.npz"
        np.savez_compressed(
            str(npz_path),
            score=vscore,
            score_shifted=score_shifted,
            spatial_isolation=d_iso,
            visibility=C_vis,
            opacity=a_op,
        )
        print(f"  scores -> {npz_path}")
        print(f"  range  raw     : [{float(vscore.min()):+.4f}, {float(vscore.max()):+.4f}]")
        print(f"  range  shifted : [{float(score_shifted.min()):+.4f}, {float(score_shifted.max()):+.4f}]")

        modality_name = f"floater_{vname}"
        # 2. Render to per-pixel σ for calib + test (unless --skip_render).
        target_calib = run_dir / "calib" / f"ours_{args.iteration}" / "raw_sigma" / modality_name
        if args.skip_render and target_calib.exists():
            print(f"  [skip_render] reusing {target_calib}")
        else:
            cmd = [
                "python", str(REPO_ROOT / "scripts" / "render_per_gaussian_scalar.py"),
                "-m", str(model_path),
                "--split_dir", str(split_dir),
                "--output_root", str(run_dir.parent),
                "--skip_train",
                "--modality", modality_name,
                "--scores_path", str(npz_path),
                "--score_key", "score_shifted",
                "--iteration", str(args.iteration),
            ]
            print("  $ " + " ".join(cmd))
            subprocess.run(cmd, check=True)

        # 3. Conformal evaluation via run_one() — overrides modality + sigma_key.
        print(f"\n  conformal eval ({modality_name}) ...")
        r = run_one(
            modality_name,
            "sigma_mean",
            run_dir.resolve(),
            args.iteration,
            args.alpha,
            calib_n=30,
            calib_sample_ratio=0.1,
            seed=args.seed,
            eps=1e-6,
        )
        cov = r["test_pixel_coverage"]
        w = r["test_mean_full_width"]
        corr = r["mean_per_view_ae_uncertainty_corr"]
        ause = r.get("mean_per_view_ause", 0.0)
        print(f"  -> cov={cov:.4f}  mean_2u={w:.2f}  AE_corr={corr:+.4f}  AUSE={ause:.4f}")
        results[vname] = {
            "coverage": cov,
            "mean_2u": w,
            "std_2u": r.get("test_full_width_std", 0.0),
            "AE_corr": corr,
            "AUSE": ause,
        }

    # Final summary
    print(f"\n\n{'='*78}\nFLOATER-SCORE VARIANT ABLATION (alpha={args.alpha}, target cov={1-args.alpha:.2f}, seed={args.seed})\n{'='*78}")
    print(f"{'variant':18} {'cov':>8} {'mean_2u':>12} {'std_2u':>12} {'AE_corr':>12}")
    print("-" * 78)
    # Sort by AE correlation descending
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["AE_corr"]):
        print(f"{name:18} {r['coverage']:8.4f} {r['mean_2u']:12.4f} {r['std_2u']:12.4f} {r['AE_corr']:+12.4f}")

    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {out_json}")


if __name__ == "__main__":
    main()
