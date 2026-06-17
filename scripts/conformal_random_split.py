"""
Re-run conformal prediction on the 90 held-out frames, but with a *random*
30/60 calib/test split instead of the original round-robin (positions 7 vs 8-9
in each block of 10). This restores calib/test exchangeability: with the
deterministic round-robin split, test frames are systematically slightly
further from any train frame than calib frames are, so test errors are larger
and conformal under-covers.

Reads existing rendered outputs (render/, gt/, raw_sigma/<modality>/), pools
the 90 frames, shuffles, splits, runs conformal in-memory.

Output: writes one summary metrics.json per modality under
    <run_dir>/conformal/<modality>_<sigma_key>_random/ours_<iter>/metrics.json
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SIGMA_KEYS = {
    "color": "color_std",
    "depth": "invdepth_std",
    "entropy": "entropy",
    "sensitivity": "sigma_mean",
    "visibility": "sigma_mean",
    "floater": "sigma_mean",
}


def load_frame(split_dir, modality, frame_name, sigma_key):
    rgb_render = np.asarray(Image.open(split_dir / "render" / frame_name).convert("RGB"), dtype=np.float32)
    rgb_gt = np.asarray(Image.open(split_dir / "gt" / frame_name).convert("RGB"), dtype=np.float32)
    npz = np.load(split_dir / "raw_sigma" / modality / frame_name.replace(".png", ".npz"))
    sigma = np.asarray(npz[sigma_key], dtype=np.float32)
    if sigma.ndim == 3:
        sigma = sigma.mean(axis=0) if sigma.shape[0] in (1, 3) else sigma.mean(axis=-1)
    mask = npz["mask"].astype(bool) if "mask" in npz.files else np.ones_like(sigma, dtype=bool)
    return rgb_render, rgb_gt, sigma, mask


def conformal_quantile(scores, alpha):
    n = len(scores)
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(scores, level, method="higher")), level


def fit_minmax(pool, modality, sigma_key, eps):
    values = []
    for sd, name in pool:
        _, _, sigma, mask = load_frame(sd, modality, name, sigma_key)
        valid = np.isfinite(sigma) & mask
        if np.any(valid):
            values.append(sigma[valid])
    if not values:
        raise ValueError(f"No finite sigma values for {modality}/{sigma_key}")
    pooled = np.concatenate(values).astype(np.float32)
    sigma_min = float(pooled.min())
    sigma_max = float(pooled.max())
    return {
        "mode": "minmax_calib",
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sigma_scale": max(sigma_max - sigma_min, eps),
    }


def normalize_sigma(sigma, mask, normalization, eps):
    sigma = np.nan_to_num(np.asarray(sigma, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sigma_norm = (sigma - normalization["sigma_min"]) / normalization["sigma_scale"]
    sigma_norm = np.clip(sigma_norm, 0.0, 1.0).astype(np.float32)
    sigma_norm[~mask] = 0.0
    return sigma_norm, np.maximum(sigma_norm, eps)


def pearson(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    if a.std() < 1e-12 or b.std() < 1e-12 or a.size < 2:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ause(abs_error, uncertainty, mask=None):
    errors = np.asarray(abs_error, dtype=np.float32).reshape(-1)
    scores = np.asarray(uncertainty, dtype=np.float32).reshape(-1)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        errors = errors[keep]
        scores = scores[keep]
    keep = np.isfinite(errors) & np.isfinite(scores)
    errors = errors[keep]
    scores = scores[keep]
    if errors.size <= 1:
        return 0.0

    def curve(order):
        ordered = errors[order]
        total = float(ordered.sum())
        if total <= 1e-12:
            return np.zeros(ordered.size + 1, dtype=np.float32)
        removed = np.concatenate([[0.0], np.cumsum(ordered, dtype=np.float64)])
        remaining = np.maximum(total - removed, 0.0)
        counts = np.arange(ordered.size, -1, -1, dtype=np.float64)
        values = np.zeros(ordered.size + 1, dtype=np.float64)
        valid = counts > 0
        values[valid] = remaining[valid] / counts[valid]
        values /= values[0] if values[0] > 1e-12 else 1.0
        return values.astype(np.float32)

    sparse = curve(np.argsort(-scores, kind="mergesort"))
    oracle = curve(np.argsort(-errors, kind="mergesort"))
    fractions = np.linspace(0.0, 1.0, sparse.size, dtype=np.float32)
    return float(np.trapz(sparse - oracle, fractions))


def run_one(modality, sigma_key, run_dir, iteration, alpha, calib_n, calib_sample_ratio, seed, eps):
    calib_dir = run_dir / "calib" / f"ours_{iteration}"
    test_dir = run_dir / "test" / f"ours_{iteration}"
    pool = []
    for sd in (calib_dir, test_dir):
        for path in sorted((sd / "render").glob("*.png")):
            pool.append((sd, path.name))
    if len(pool) != 90:
        print(f"  [warn] expected 90 pooled frames, got {len(pool)}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    calib_idx = perm[:calib_n].tolist()
    test_idx = perm[calib_n:].tolist()
    calib_pool = [pool[i] for i in calib_idx]
    test_pool = [pool[i] for i in test_idx]

    normalization = fit_minmax(calib_pool, modality, sigma_key, eps)

    # Calibration: sample pixels and compute scores = |err| / u_norm.
    calib_scores = []
    for sd, name in calib_pool:
        render, gt, sigma, mask = load_frame(sd, modality, name, sigma_key)
        err = np.abs(render - gt).mean(axis=2)  # mean over RGB channels
        sigma_norm, sigma_safe = normalize_sigma(sigma, mask, normalization, eps)
        n_pix = err.size
        n_samp = max(1, int(math.ceil(calib_sample_ratio * n_pix)))
        idx = rng.choice(n_pix, size=n_samp, replace=False)
        flat_err = err.reshape(-1)[idx]
        flat_sig = sigma_safe.reshape(-1)[idx]
        flat_msk = mask.reshape(-1)[idx]
        keep = flat_msk & np.isfinite(flat_err) & np.isfinite(flat_sig)
        calib_scores.extend((flat_err[keep] / flat_sig[keep]).tolist())

    q_hat, level = conformal_quantile(calib_scores, alpha)

    # Test: every pixel.
    covs, widths, width_stds, corrs, auses, view_metrics = [], [], [], [], [], {}
    for sd, name in test_pool:
        render, gt, sigma, mask = load_frame(sd, modality, name, sigma_key)
        err = np.abs(render - gt).mean(axis=2)
        sigma_norm, _ = normalize_sigma(sigma, mask, normalization, eps)
        u = q_hat * sigma_norm
        within = err <= u
        full_w = 2.0 * u
        valid = np.isfinite(err) & np.isfinite(full_w) & mask
        coverage = float(within[valid].mean()) if np.any(valid) else 0.0
        mean_width = float(full_w[valid].mean()) if np.any(valid) else 0.0
        width_std = float(full_w[valid].std()) if np.any(valid) else 0.0
        corr = pearson(err[valid], full_w[valid])
        view_ause = ause(err, full_w, mask=valid)
        covs.append(coverage)
        widths.append(mean_width)
        width_stds.append(width_std)
        corrs.append(corr)
        auses.append(view_ause)
        view_metrics[name] = {
            "num_pixels": int(err.size),
            "num_valid_pixels": int(valid.sum()),
            "coverage": coverage,
            "ae_uncertainty_corr": corr,
            "ause": view_ause,
            "mean_full_width": mean_width,
            "full_width_std": width_std,
        }

    return {
        "modality": modality,
        "sigma_key": sigma_key,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "test_pixel_coverage": float(np.mean(covs)) if covs else 0.0,
        "test_pixel_coverage_std": float(np.std(covs)) if covs else 0.0,
        "test_mean_full_width": float(np.mean(widths)) if widths else 0.0,
        "test_mean_full_width_std_per_view": float(np.std(widths)) if widths else 0.0,
        "test_full_width_std": float(np.mean(width_stds)) if width_stds else 0.0,
        "test_full_width_std_std_per_view": float(np.std(width_stds)) if width_stds else 0.0,
        "mean_per_view_ae_uncertainty_corr": float(np.mean(corrs)) if corrs else 0.0,
        "std_per_view_ae_uncertainty_corr": float(np.std(corrs)) if corrs else 0.0,
        "mean_per_view_ause": float(np.mean(auses)) if auses else 0.0,
        "std_per_view_ause": float(np.std(auses)) if auses else 0.0,
        "q_hat": q_hat,
        "quantile_level": level,
        "num_calib_frames": len(calib_pool),
        "num_test_frames": len(test_pool),
        "num_calib_scores": len(calib_scores),
        "split_seed": seed,
        "sigma_normalization": normalization,
        "test_frame_metrics": view_metrics,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--calib_n", type=int, default=30)
    p.add_argument("--calib_sample_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1e-6)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if args.iteration == -1:
        candidates = []
        for split in ("calib", "test"):
            for d in (run_dir / split).iterdir():
                if d.name.startswith("ours_"):
                    candidates.append(int(d.name.split("_", 1)[1]))
        iteration = max(candidates)
    else:
        iteration = args.iteration

    print(f"Run dir   : {run_dir}")
    print(f"Iteration : {iteration}")
    print(f"alpha     : {args.alpha}, target cov: {1.0 - args.alpha:.3f}")
    print(f"split     : random seed={args.seed}, calib_n={args.calib_n}")
    print()
    print(f"{'modality':12} {'cov':>7} {'mean_full_width':>16} {'std_full_width':>16} {'AE_corr':>10} {'AUSE':>10}")
    print("-" * 68)

    for mod, key in DEFAULT_SIGMA_KEYS.items():
        try:
            r = run_one(mod, key, run_dir, iteration, args.alpha, args.calib_n, args.calib_sample_ratio, args.seed, args.eps)
        except FileNotFoundError as e:
            print(f"{mod:12} skipped: {e}")
            continue
        out_dir = run_dir / "conformal" / f"{mod}_{key}_random" / f"ours_{iteration}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics.json", "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"{mod:12} {r['test_pixel_coverage']:7.4f} {r['test_mean_full_width']:16.2f} {r['test_full_width_std']:16.2f} {r['mean_per_view_ae_uncertainty_corr']:10.4f} {r['mean_per_view_ause']:10.4f}")


if __name__ == "__main__":
    main()
