"""
Per-view quality prediction from conformal uncertainty.

Research question: Can the mean conformal uncertainty of a view predict
how bad the render will be -- before having ground truth?

For each test view, computes mean conformal uncertainty width (2*q_hat*sigma),
PSNR, SSIM, and MAE, then checks whether uncertainty correlates with each
quality metric across views.

Different from per-pixel Pearson r in conformal_prediction.py:
  - Per-pixel r: within one view, do uncertain pixels have higher error?
  - Per-view r (this script): across views, do more uncertain views render worse?

Usage:
  python scripts/predict_quality.py --run_dir output/my_run [--iteration -1]

Outputs written to <run_dir>/quality_prediction/:
  quality_prediction.json   per-view stats and per-modality correlations
  quality_prediction.png    scatter plots (uncertainty vs PSNR, SSIM, MAE)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    from skimage.metrics import structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def load_rgb_uint8(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def compute_psnr(render, gt):
    mse = np.mean((render.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(render, gt):
    if not SKIMAGE_AVAILABLE:
        return None
    return float(structural_similarity(render, gt, channel_axis=2, data_range=255))


def compute_mae(render, gt):
    return float(np.mean(np.abs(render.astype(np.float32) - gt.astype(np.float32))))


def find_iteration(run_dir, iteration):
    if iteration != -1:
        return iteration
    candidates = []
    for split_name in ("calib", "test"):
        split_dir = run_dir / split_name
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if child.is_dir() and child.name.startswith("ours_"):
                try:
                    candidates.append(int(child.name.split("_", 1)[1]))
                except ValueError:
                    continue
    if not candidates:
        raise FileNotFoundError(f"No iteration directories found under {run_dir}")
    return max(candidates)


def compute_image_metrics(render_dir, gt_dir):
    metrics = {}
    for render_path in sorted(render_dir.glob("*.png")):
        gt_path = gt_dir / render_path.name
        if not gt_path.exists():
            continue
        render = load_rgb_uint8(render_path)
        gt = load_rgb_uint8(gt_path)
        metrics[render_path.stem] = {
            "psnr": compute_psnr(render, gt),
            "ssim": compute_ssim(render, gt),
            "mae":  compute_mae(render, gt),
        }
    return metrics


def load_uncertainty_per_view(arrays_dir):
    uncertainty = {}
    for npz_path in sorted(arrays_dir.glob("*.npz")):
        d = np.load(npz_path)
        unc = d["uncertainty_full_width"].astype(np.float32)
        mask = d["mask"].astype(bool) if "mask" in d else np.ones_like(unc, dtype=bool)
        valid = mask & np.isfinite(unc)
        if not valid.any():
            continue
        coverage = (
            float(d["within_interval"].astype(float)[valid].mean())
            if "within_interval" in d else None
        )
        uncertainty[npz_path.stem] = {
            "mean_uncertainty": float(unc[valid].mean()),
            "coverage": coverage,
            "n_pixels": int(valid.sum()),
        }
    return uncertainty


def merge_records(uncertainty_map, image_metrics):
    records = []
    for stem, unc in uncertainty_map.items():
        if stem not in image_metrics:
            continue
        records.append({
            "frame":            stem,
            "mean_uncertainty": unc["mean_uncertainty"],
            "coverage":         unc["coverage"],
            "psnr":             image_metrics[stem]["psnr"],
            "ssim":             image_metrics[stem]["ssim"],
            "mae":              image_metrics[stem]["mae"],
        })
    return sorted(records, key=lambda r: r["frame"])


def compute_correlations(records):
    if len(records) < 3:
        return {}
    corrs = {}
    for metric in ("psnr", "ssim", "mae"):
        vals = [r[metric] for r in records if r[metric] is not None]
        unc_ = [r["mean_uncertainty"] for r in records if r[metric] is not None]
        if len(vals) < 3:
            corrs[metric] = (float("nan"), float("nan"))
            continue
        v, u = np.array(vals), np.array(unc_)
        if v.std() < 1e-9 or u.std() < 1e-9:
            corrs[metric] = (float("nan"), float("nan"))
            continue
        r_val, p_val = pearsonr(u, v)
        corrs[metric] = (float(r_val), float(p_val))
    return corrs


METRIC_META = {
    "psnr": ("PSNR (dB)",   "negative"),
    "ssim": ("SSIM",        "negative"),
    "mae":  ("MAE [0-255]", "positive"),
}


def plot_row(axes_row, analysis_name, records, corrs):
    active = [m for m in ("psnr", "ssim", "mae")
              if any(r[m] is not None for r in records)]
    for col, metric in enumerate(active):
        ax = axes_row[col]
        valid = [r for r in records if r[metric] is not None]
        unc_vals = np.array([r["mean_uncertainty"] for r in valid])
        met_vals = np.array([r[metric] for r in valid])

        ax.scatter(unc_vals, met_vals, s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=3)

        r_val, p_val = corrs.get(metric, (float("nan"), float("nan")))
        if not math.isnan(r_val) and len(valid) >= 3:
            coeffs = np.polyfit(unc_vals, met_vals, 1)
            x_line = np.linspace(unc_vals.min(), unc_vals.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line),
                    "r--", linewidth=1.5, alpha=0.7)

        for rec in valid:
            ax.annotate(rec["frame"],
                        xy=(rec["mean_uncertainty"], rec[metric]),
                        fontsize=6, alpha=0.55,
                        xytext=(3, 3), textcoords="offset points")

        label, direction = METRIC_META[metric]
        r_str = f"{r_val:+.3f}" if not math.isnan(r_val) else "n/a"
        p_str = f"{p_val:.3f}" if not math.isnan(p_val) else "n/a"
        expected = "negative r expected" if direction == "negative" else "positive r expected"
        ax.set_title(
            f"{analysis_name}  -  vs {label}\n"
            f"r={r_str}  p={p_str}  ({expected})  n={len(valid)} views",
            fontsize=9
        )
        ax.set_xlabel("Mean uncertainty (2*q_hat*sigma)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    for col in range(len(active), len(axes_row)):
        axes_row[col].set_visible(False)


def main():
    parser = argparse.ArgumentParser(
        description="Per-view quality prediction from conformal uncertainty"
    )
    parser.add_argument("--run_dir",   required=True, type=str)
    parser.add_argument("--iteration", default=-1,    type=int)
    args = parser.parse_args()

    run_dir       = Path(args.run_dir).resolve()
    iteration     = find_iteration(run_dir, args.iteration)
    test_iter_dir = run_dir / "test" / f"ours_{iteration}"
    out_dir       = run_dir / "quality_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir   : {run_dir}")
    print(f"Iteration : {iteration}")
    if not SKIMAGE_AVAILABLE:
        print("  [NOTE] scikit-image not found - SSIM skipped. pip install scikit-image")

    render_dir = test_iter_dir / "render"
    gt_dir     = test_iter_dir / "gt"
    if not render_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(
            f"Missing render/gt dirs under {test_iter_dir}. "
            "Run render_color.py or render_depth.py first."
        )

    print("Computing PSNR, SSIM, MAE for test frames...")
    image_metrics = compute_image_metrics(render_dir, gt_dir)
    print(f"  {len(image_metrics)} test frames\n")

    conformal_base = test_iter_dir / "conformal"
    if not conformal_base.exists():
        raise FileNotFoundError(
            f"No conformal outputs at {conformal_base}. "
            "Run conformal_prediction.py first."
        )

    analysis_dirs = sorted(d for d in conformal_base.iterdir() if d.is_dir())
    if not analysis_dirs:
        raise FileNotFoundError(f"No analysis subdirectories under {conformal_base}")

    print(f"Analyses: {[d.name for d in analysis_dirs]}\n")

    all_results = {}
    valid_items = []

    for analysis_dir in analysis_dirs:
        arrays_dir = analysis_dir / "arrays"
        if not arrays_dir.exists():
            print(f"  [SKIP] {analysis_dir.name} - no arrays/ subdir")
            continue
        unc_map = load_uncertainty_per_view(arrays_dir)
        records = merge_records(unc_map, image_metrics)
        if len(records) < 2:
            print(f"  [SKIP] {analysis_dir.name} - fewer than 2 matched frames")
            continue
        corrs = compute_correlations(records)

        print(f"  {analysis_dir.name}")
        for metric, (r_val, p_val) in corrs.items():
            label = METRIC_META[metric][0]
            r_str = f"{r_val:+.3f}" if not math.isnan(r_val) else "  n/a"
            p_str = f"{p_val:.3f}" if not math.isnan(p_val) else "  n/a"
            print(f"    vs {label:<12}  r={r_str}  p={p_str}")
        print()

        all_results[analysis_dir.name] = {
            "n_views":      len(records),
            "correlations": {m: {"r": r, "p": p} for m, (r, p) in corrs.items()},
            "per_view":     records,
        }
        valid_items.append((analysis_dir.name, records, corrs))

    if not valid_items:
        print("No valid results to plot.")
        return

    json_path = out_dir / "quality_prediction.json"
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"JSON  -> {json_path}")

    n_rows = len(valid_items)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)
    fig.suptitle(
        "Per-view quality prediction: does mean uncertainty predict render quality?\n"
        "Each dot = one test view",
        fontsize=13
    )
    for row, (name, records, corrs) in enumerate(valid_items):
        plot_row(axes[row], name, records, corrs)

    plt.tight_layout()
    fig_path = out_dir / "quality_prediction.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot  -> {fig_path}")

    print("\n" + "=" * 70)
    print(f"{'Analysis':<35} {'vs PSNR':>10} {'vs SSIM':>10} {'vs MAE':>10}")
    print("-" * 70)
    for name, _, corrs in valid_items:
        row_vals = []
        for m in ("psnr", "ssim", "mae"):
            r_val = corrs.get(m, (float("nan"),))[0]
            row_vals.append(f"{r_val:+.3f}" if not math.isnan(r_val) else "  n/a")
        print(f"  {name:<33} {row_vals[0]:>10} {row_vals[1]:>10} {row_vals[2]:>10}")
    print("=" * 70)
    print("\nExpected: uncertainty vs PSNR -> negative r (uncertain = lower quality)")
    print("          uncertainty vs SSIM -> negative r")
    print("          uncertainty vs MAE  -> positive r (uncertain = higher error)")


if __name__ == "__main__":
    main()
