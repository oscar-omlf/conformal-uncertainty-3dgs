"""
Pick the pruning fraction K* that maximizes calibration-set rendering quality
(PSNR), subject to *not* breaking the pre-existing conformal coverage guarantee.

This is the "conformal-calibrated pruning" extension to TIDI-GS. Instead of
hand-tuning K like TIDI-GS does per scene, we sweep K over a small grid, run
the per-K pruned model on calib views, and pick the largest K whose calib PSNR
is monotonically better than baseline *and* whose conformal coverage on calib
is at least 1−α (so the bands stay valid).

Inputs
------
--model_path   : path to trained run dir (must contain calib/ split rendered already
                 with the chosen sigma modality)
--scores_npz, --score_key : per-Gaussian score (e.g. floater.npz)
--sigma_modality, --sigma_key : which σ to use for coverage check (default: color)
--K_grid       : comma list of fractions to sweep, default "0.00,0.05,0.10,0.15,0.20,0.25"
--alpha        : conformal target miscoverage (default 0.1)

Outputs
-------
- For each K: writes pruned renders to <model_path>/_prune_sweep/K_<frac>/
- Computes per-K calib PSNR (via in-memory PSNR), per-K calib coverage
- Picks K* = argmax_K calib PSNR(K) subject to coverage(K) >= 1−α
- Writes <model_path>/uncertainty/pruning_pick.json with the table
"""

import json
import math
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def psnr_per_view(render_dir: Path, gt_dir: Path):
    """Per-view PSNR over the calib split. Returns (mean_psnr, list_per_view)."""
    psnrs = []
    for f in sorted(render_dir.glob("*.png")):
        r = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
        g = np.asarray(Image.open(gt_dir / f.name).convert("RGB"), dtype=np.float32)
        mse = float(np.mean((r - g) ** 2))
        # PSNR in 0-255 RGB scale.
        psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(max(mse, 1e-12))
        psnrs.append(psnr)
    return float(np.mean(psnrs)) if psnrs else 0.0, psnrs


def coverage_per_view(render_dir: Path, gt_dir: Path, sigma_dir: Path, sigma_key: str, q_hat: float):
    """Per-view conformal coverage at a given q_hat. Returns mean coverage."""
    covs = []
    for f in sorted(render_dir.glob("*.png")):
        r = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
        g = np.asarray(Image.open(gt_dir / f.name).convert("RGB"), dtype=np.float32)
        err = np.abs(r - g).mean(axis=2)
        npz_path = sigma_dir / (f.stem + ".npz")
        if not npz_path.exists():
            continue
        npz = np.load(npz_path)
        sigma = np.asarray(npz[sigma_key], dtype=np.float32)
        if sigma.ndim == 3:
            sigma = sigma.mean(axis=0) if sigma.shape[0] in (1, 3) else sigma.mean(axis=-1)
        mask = npz["mask"].astype(bool) if "mask" in npz.files else np.ones_like(sigma, dtype=bool)
        valid = mask & np.isfinite(err) & np.isfinite(sigma)
        if not np.any(valid):
            continue
        u = q_hat * sigma[valid]
        covs.append(float((err[valid] <= u).mean()))
    return float(np.mean(covs)) if covs else 0.0


def calib_q_hat(sigma_dir: Path, render_dir: Path, gt_dir: Path, sigma_key: str, alpha: float, eps: float = 1e-6, sample_ratio: float = 0.1, seed: int = 42):
    """Compute conformal q_hat on this calib set with the given sigma."""
    rng = np.random.default_rng(seed)
    scores = []
    for f in sorted(render_dir.glob("*.png")):
        r = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
        g = np.asarray(Image.open(gt_dir / f.name).convert("RGB"), dtype=np.float32)
        err = np.abs(r - g).mean(axis=2).reshape(-1)
        npz_path = sigma_dir / (f.stem + ".npz")
        if not npz_path.exists():
            continue
        npz = np.load(npz_path)
        sigma = np.asarray(npz[sigma_key], dtype=np.float32)
        if sigma.ndim == 3:
            sigma = sigma.mean(axis=0) if sigma.shape[0] in (1, 3) else sigma.mean(axis=-1)
        sigma = sigma.reshape(-1)
        sigma_safe = np.maximum(sigma, eps)
        n = err.size
        ns = max(1, int(math.ceil(sample_ratio * n)))
        idx = rng.choice(n, size=ns, replace=False)
        scores.extend((err[idx] / sigma_safe[idx]).tolist())
    if not scores:
        raise RuntimeError("No calib scores collected.")
    n = len(scores)
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    q_hat = float(np.quantile(np.asarray(scores, dtype=np.float32), level, method="higher"))
    return q_hat, level


def main():
    p = ArgumentParser()
    p.add_argument("--model_path", required=True, type=str)
    p.add_argument("--split_dir", required=True, type=str)
    p.add_argument("--iteration", default=30000, type=int)
    p.add_argument("--scores_npz", required=True, type=str)
    p.add_argument("--score_key", default="score_shifted", type=str)
    p.add_argument("--sigma_modality", default="color", type=str)
    p.add_argument("--sigma_key", default="color_std", type=str)
    p.add_argument("--K_grid", default="0.00,0.02,0.05,0.10,0.15,0.20,0.25", type=str)
    p.add_argument("--alpha", default=0.1, type=float)
    p.add_argument("--sweep_out_root", default=None, type=str,
                   help="Root dir under which K-pruned renders go. Defaults to <model_path>/_prune_sweep")
    args = p.parse_args()

    model_path = Path(args.model_path).resolve()
    iter_str = f"ours_{args.iteration}"
    K_grid = [float(x) for x in args.K_grid.split(",")]

    sweep_root = Path(args.sweep_out_root) if args.sweep_out_root else (model_path / "_prune_sweep")
    sweep_root.mkdir(parents=True, exist_ok=True)

    # Baseline calib/sigma path (from the pipeline output).
    baseline_calib_dir = model_path / "calib" / iter_str
    baseline_render_dir = baseline_calib_dir / "render"
    baseline_gt_dir = baseline_calib_dir / "gt"
    baseline_sigma_dir = baseline_calib_dir / "raw_sigma" / args.sigma_modality

    # Compute baseline q_hat and baseline calib PSNR / coverage (K=0).
    print(f"Baseline calib q_hat using {args.sigma_modality}/{args.sigma_key} ...")
    q_hat_baseline, level = calib_q_hat(
        baseline_sigma_dir, baseline_render_dir, baseline_gt_dir, args.sigma_key, args.alpha
    )
    psnr_baseline, _ = psnr_per_view(baseline_render_dir, baseline_gt_dir)
    cov_baseline = coverage_per_view(baseline_render_dir, baseline_gt_dir, baseline_sigma_dir, args.sigma_key, q_hat_baseline)
    print(f"  baseline calib PSNR     : {psnr_baseline:.4f}")
    print(f"  baseline calib coverage : {cov_baseline:.4f} (target {1-args.alpha:.2f})")
    print(f"  baseline q_hat (level {level:.4f}): {q_hat_baseline:.4g}")

    # Sweep K.
    rows = [{"K": 0.0, "psnr_calib": psnr_baseline, "coverage_calib": cov_baseline, "q_hat": q_hat_baseline, "render_dir": str(baseline_render_dir)}]
    for K in K_grid:
        if K <= 0.0:
            continue
        sub_out = sweep_root / f"K_{K:.4f}"
        sub_out.mkdir(parents=True, exist_ok=True)
        print(f"\n--- K={K:.4f}: pruning and re-rendering calib ---")
        cmd = [
            "python", str(REPO_ROOT / "scripts" / "prune_and_render.py"),
            "-m", str(model_path),
            "--split_dir", args.split_dir,
            "--iteration", str(args.iteration),
            "--scores_npz", str(args.scores_npz),
            "--score_key", args.score_key,
            "--prune_fraction", str(K),
            "--output_dir", str(sub_out),
            "--skip_train",
            "--skip_test",  # only need calib here for the K* selection
        ]
        print("  $ " + " ".join(cmd))
        subprocess.run(cmd, check=True)

        pruned_calib_render = sub_out / "calib" / iter_str / "render"
        pruned_calib_gt = sub_out / "calib" / iter_str / "gt"
        # Re-use baseline σ maps (the σ maps don't change with pruning since
        # they were rendered from the baseline trained model). Slightly
        # imperfect — strictly we'd re-render σ after pruning — but for the
        # coverage gate this is a conservative proxy and avoids a 3x cost.
        psnr_K, _ = psnr_per_view(pruned_calib_render, pruned_calib_gt)
        cov_K = coverage_per_view(pruned_calib_render, pruned_calib_gt, baseline_sigma_dir, args.sigma_key, q_hat_baseline)
        print(f"  K={K:.4f}: calib PSNR {psnr_K:.4f}, calib coverage {cov_K:.4f}")
        rows.append({
            "K": K,
            "psnr_calib": psnr_K,
            "coverage_calib": cov_K,
            "q_hat": q_hat_baseline,
            "render_dir": str(pruned_calib_render),
        })

    # Pick K*: largest K whose calib coverage >= 1-alpha AND psnr_calib >= baseline.
    target_cov = 1.0 - args.alpha
    print(f"\n--- Picking K* with cov >= {target_cov:.3f} and PSNR >= baseline {psnr_baseline:.4f} ---")
    feasible = [r for r in rows if r["coverage_calib"] >= target_cov and r["psnr_calib"] >= psnr_baseline]
    if not feasible:
        print("  no K satisfies the constraint — falling back to K=0 (baseline).")
        K_star = 0.0
    else:
        best = max(feasible, key=lambda r: (r["psnr_calib"], r["K"]))
        K_star = best["K"]
        print(f"  K* = {K_star:.4f} (calib PSNR {best['psnr_calib']:.4f}, coverage {best['coverage_calib']:.4f})")

    summary = {
        "alpha": args.alpha,
        "target_coverage": target_cov,
        "sigma_modality": args.sigma_modality,
        "sigma_key": args.sigma_key,
        "score_npz": args.scores_npz,
        "score_key": args.score_key,
        "K_grid": K_grid,
        "rows": rows,
        "K_star": K_star,
        "psnr_baseline": psnr_baseline,
        "coverage_baseline": cov_baseline,
        "q_hat_baseline": q_hat_baseline,
    }
    out_path = model_path / "uncertainty" / "pruning_pick.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nK* = {K_star}")


if __name__ == "__main__":
    main()
