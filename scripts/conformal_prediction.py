import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import cm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SIGMA_KEYS = {
    "color": "color_std",
    "depth": "invdepth_std",
    "entropy": "entropy",
    "sensitivity": "sigma_mean",
    "visibility": "sigma_mean",
    "floater": "sigma_mean",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)


def normalize_preview(array, mask=None):
    preview = np.array(array, dtype=np.float32, copy=True)
    finite_mask = np.isfinite(preview)
    if mask is not None:
        finite_mask &= mask.astype(bool)
    if not np.any(finite_mask):
        return np.zeros_like(preview, dtype=np.uint8)
    values = preview[finite_mask]
    lo = float(np.min(values))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    preview = np.clip((preview - lo) / (hi - lo), 0.0, 1.0)
    preview[~finite_mask] = 0.0
    return np.rint(preview * 255.0).astype(np.uint8)


def parse_args():
    parser = ArgumentParser(description="Conformal prediction from saved sigma maps")
    parser.add_argument("--run_dir", required=True, type=str, help="Run directory, e.g. output/my_run")
    parser.add_argument(
        "--modality",
        required=True,
        choices=["color", "depth", "entropy", "sensitivity", "visibility", "floater"],
    )
    parser.add_argument("--sigma_key", default=None, type=str, help="Key to read from raw sigma .npz files")
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to use, default: latest")
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--calib_sample_ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--rgb_error_mode", default="mean", choices=["mean", "max"], help="How to reduce RGB absolute error across channels")
    parser.add_argument(
        "--sigma_norm",
        default="minmax_calib",
        choices=["none", "minmax_calib", "minmax_pooled"],
        help="How to normalize uncertainty u before calibration. "
             "'minmax_calib' fits min/max on calibration pixels only and is the default. "
             "'none' uses raw u. 'minmax_pooled' fits on calib+test for diagnostics only.",
    )
    return parser.parse_args()


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


def list_frame_names(split_dir, modality):
    render_dir = split_dir / "render"
    gt_dir = split_dir / "gt"
    sigma_dir = split_dir / "raw_sigma" / modality
    if not render_dir.exists() or not gt_dir.exists() or not sigma_dir.exists():
        raise FileNotFoundError(f"Missing render/gt/raw_sigma directories under {split_dir}")
    return sorted(path.name for path in render_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def load_sigma_npz(npz_path, sigma_key):
    sigma_data = np.load(npz_path)
    if sigma_key not in sigma_data:
        raise KeyError(f"{sigma_key} not found in {npz_path}. Available keys: {list(sigma_data.files)}")

    sigma = np.asarray(sigma_data[sigma_key], dtype=np.float32)
    if sigma.ndim == 3:
        if sigma.shape[0] in (1, 3):
            sigma = sigma.mean(axis=0)
        elif sigma.shape[-1] in (1, 3):
            sigma = sigma.mean(axis=-1)
        else:
            raise ValueError(f"Unsupported sigma shape for {npz_path}: {sigma.shape}")
    if sigma.ndim != 2:
        raise ValueError(f"Expected 2D sigma map in {npz_path}, got {sigma.shape}")

    mask = sigma_data["mask"].astype(bool) if "mask" in sigma_data else np.ones_like(sigma, dtype=bool)
    return sigma, mask


def compute_error(render_rgb, gt_rgb, modality, rgb_error_mode):
    abs_diff = np.abs(render_rgb - gt_rgb)
    if rgb_error_mode == "max":
        return abs_diff.max(axis=2)
    return abs_diff.mean(axis=2)


def compute_normalization(raw_sigma_values, eps, mode="minmax_calib"):
    """
    mode = 'none'         : identity normalization (sigma_norm = raw sigma).
    mode = 'minmax_calib' : (sigma - min) / (max - min) using calib values only.
    mode = 'minmax_pooled': caller passes calib+test sigma values together.
    """
    if mode == "none":
        return {"mode": "none"}
    values = np.asarray(raw_sigma_values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite sigma values for normalization.")
    sigma_min = float(values.min())
    sigma_max = float(values.max())
    sigma_scale = max(sigma_max - sigma_min, eps)
    return {
        "mode": mode,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sigma_scale": sigma_scale,
    }


def normalize_sigma(raw_sigma, mask, normalization, eps):
    sigma = np.asarray(raw_sigma, dtype=np.float32)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    if normalization.get("mode") == "none":
        sigma_norm = sigma.copy()
    else:
        sigma_norm = (sigma - normalization["sigma_min"]) / normalization["sigma_scale"]
        sigma_norm = np.clip(sigma_norm, 0.0, 1.0).astype(np.float32)
    sigma_norm[~mask] = 0.0
    sigma_safe = np.maximum(sigma_norm, eps)
    return sigma_norm, sigma_safe


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=np.float32)
    n = len(scores)
    if n == 0:
        raise ValueError("No calibration scores collected.")
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    q_hat = float(np.quantile(scores, level, method="higher"))
    return q_hat, level


def pearson_corr(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size < 2 or b.size < 2:
        return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sparsification_curve(abs_error, uncertainty, mask=None):
    errors = np.asarray(abs_error, dtype=np.float32).reshape(-1)
    scores = np.asarray(uncertainty, dtype=np.float32).reshape(-1)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        errors = errors[keep]
        scores = scores[keep]
    keep = np.isfinite(errors) & np.isfinite(scores)
    errors = errors[keep]
    scores = scores[keep]
    if errors.size == 0:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    desc = np.argsort(-scores, kind="mergesort")
    oracle = np.argsort(-errors, kind="mergesort")

    def curve(order):
        ordered_errors = errors[order]
        total = float(ordered_errors.sum())
        n = ordered_errors.size
        if n == 0:
            return np.array([0.0], dtype=np.float32)
        if total <= 1e-12:
            return np.zeros(n + 1, dtype=np.float32)
        removed = np.concatenate([[0.0], np.cumsum(ordered_errors, dtype=np.float64)])
        remaining = np.maximum(total - removed, 0.0)
        remaining_count = np.arange(n, -1, -1, dtype=np.float64)
        values = np.zeros(n + 1, dtype=np.float64)
        valid = remaining_count > 0
        values[valid] = remaining[valid] / remaining_count[valid]
        values /= values[0] if values[0] > 1e-12 else 1.0
        return values.astype(np.float32)

    return curve(desc), curve(oracle)


def ause(abs_error, uncertainty, mask=None):
    sparse, oracle = sparsification_curve(abs_error, uncertainty, mask)
    if sparse.size <= 1:
        return 0.0
    fractions = np.linspace(0.0, 1.0, sparse.size, dtype=np.float32)
    return float(np.trapz(sparse - oracle, fractions))


def finite_stats(array, mask=None):
    values = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    values = values[valid]
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def metric_mean(per_view_metrics, key):
    values = [metrics[key] for metrics in per_view_metrics.values() if key in metrics]
    return float(np.mean(values)) if values else 0.0


def metric_std(per_view_metrics, key):
    values = [metrics[key] for metrics in per_view_metrics.values() if key in metrics]
    return float(np.std(values)) if values else 0.0


def save_grayscale(array, path, mask=None):
    Image.fromarray(normalize_preview(array, mask=mask), mode="L").save(path)


def apply_colormap(array, cmap_name, mask=None, assume_unit_range=False):
    values = np.asarray(array, dtype=np.float32)
    if assume_unit_range:
        norm = np.clip(values, 0.0, 1.0)
    else:
        norm = normalize_preview(values, mask=mask).astype(np.float32) / 255.0
    rgba = cm.get_cmap(cmap_name)(norm)
    rgb = np.rint(rgba[..., :3] * 255.0).astype(np.uint8)
    if mask is not None:
        rgb = rgb.copy()
        rgb[~mask] = 0
    return rgb


def save_colormap(array, path, cmap_name, mask=None, assume_unit_range=False):
    rgb = apply_colormap(array, cmap_name, mask=mask, assume_unit_range=assume_unit_range)
    Image.fromarray(rgb, mode="RGB").save(path)


def save_sigma_normalized(sigma_norm, path):
    save_colormap(sigma_norm, path, cmap_name="turbo", assume_unit_range=True)


def save_panel(path, gt_rgb, render_rgb, raw_sigma, abs_error, sigma_norm, uncertainty_map, mask):
    panels = [
        Image.fromarray(np.rint(np.clip(gt_rgb, 0.0, 255.0)).astype(np.uint8), mode="RGB"),
        Image.fromarray(np.rint(np.clip(render_rgb, 0.0, 255.0)).astype(np.uint8), mode="RGB"),
        Image.fromarray(normalize_preview(raw_sigma, mask=mask), mode="L").convert("RGB"),
        Image.fromarray(apply_colormap(abs_error, cmap_name="turbo"), mode="RGB"),
        Image.fromarray(apply_colormap(sigma_norm, cmap_name="turbo", assume_unit_range=True), mode="RGB"),
        Image.fromarray(apply_colormap(uncertainty_map, cmap_name="turbo"), mode="RGB"),
    ]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for idx, panel in enumerate(panels):
        canvas.paste(panel.resize((width, height)), (idx * width, 0))
    canvas.save(path)


def prepare_output_dirs(run_dir, split_name, iteration, analysis_name):
    base_dir = run_dir / split_name / f"ours_{iteration}" / "conformal" / analysis_name
    paths = {
        "base": base_dir,
        "abs_error": base_dir / "abs_error",
        "raw_sigma": base_dir / "raw_sigma",
        "sigma_normalized": base_dir / "sigma_normalized",
        "uncertainty": base_dir / "uncertainty",
        "panels": base_dir / "panels",
        "arrays": base_dir / "arrays",
    }
    for path in paths.values():
        ensure_dir(str(path))
    return paths


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    sigma_key = args.sigma_key or DEFAULT_SIGMA_KEYS[args.modality]
    norm_tag = "" if args.sigma_norm == "minmax_calib" else f"_{args.sigma_norm}"
    analysis_name = f"{args.modality}_{sigma_key}{norm_tag}"
    iteration = find_iteration(run_dir, args.iteration)

    print("=" * 72)
    print("Conformal Prediction")
    print("=" * 72)
    print(f"Run directory      : {run_dir}")
    print(f"Modality           : {args.modality}")
    print(f"Sigma key          : {sigma_key}")
    print(f"RGB error mode     : {args.rgb_error_mode}")
    print(f"Alpha              : {args.alpha}")
    print(f"Target coverage    : {1.0 - args.alpha:.3f}")
    print(f"Calib sample ratio : {args.calib_sample_ratio}")
    print(f"Epsilon            : {args.eps}")
    print(f"Iteration          : {iteration}")

    calib_dir = run_dir / "calib" / f"ours_{iteration}"
    test_dir = run_dir / "test" / f"ours_{iteration}"
    if not calib_dir.exists():
        raise FileNotFoundError(f"Missing calibration directory: {calib_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    calib_frames = list_frame_names(calib_dir, args.modality)
    test_frames = list_frame_names(test_dir, args.modality)
    print(f"Calibration frames : {len(calib_frames)}")
    print(f"Test frames        : {len(test_frames)}")
    print(f"Calibration dir    : {calib_dir}")
    print(f"Test dir           : {test_dir}")
    print("Scanning calibration sigma values...")

    calib_sigma_values = []
    for frame_name in calib_frames:
        raw_sigma, mask = load_sigma_npz(calib_dir / "raw_sigma" / args.modality / frame_name.replace(".png", ".npz"), sigma_key)
        valid = np.isfinite(raw_sigma) & mask
        if np.any(valid):
            calib_sigma_values.append(raw_sigma[valid])
    if not calib_sigma_values:
        raise ValueError("No valid calibration sigma values found.")

    if args.sigma_norm == "minmax_pooled":
        test_sigma_values = []
        for frame_name in test_frames:
            raw_sigma, mask = load_sigma_npz(test_dir / "raw_sigma" / args.modality / frame_name.replace(".png", ".npz"), sigma_key)
            valid = np.isfinite(raw_sigma) & mask
            if np.any(valid):
                test_sigma_values.append(raw_sigma[valid])
        pooled = np.concatenate(calib_sigma_values + test_sigma_values)
        normalization = compute_normalization(pooled, args.eps, mode="minmax_pooled")
    else:
        normalization = compute_normalization(np.concatenate(calib_sigma_values), args.eps, mode=args.sigma_norm)
    if normalization.get("mode") == "none":
        print("Sigma normalization: none (raw sigma)")
    else:
        print(
            f"Sigma normalization ({normalization['mode']}): "
            f"min={normalization['sigma_min']:.6f}, "
            f"max={normalization['sigma_max']:.6f}, "
            f"scale={normalization['sigma_scale']:.6f}"
        )

    rng = np.random.default_rng(args.seed)
    calib_scores = []
    calib_frame_stats = {}
    calib_out_dirs = prepare_output_dirs(run_dir, "calib", iteration, analysis_name)
    print(f"Calibration outputs: {calib_out_dirs['base']}")
    print("Processing calibration frames...")

    total_calib_sampled_pixels = 0

    for frame_idx, frame_name in enumerate(calib_frames, start=1):
        render_rgb = load_rgb(calib_dir / "render" / frame_name)
        gt_rgb = load_rgb(calib_dir / "gt" / frame_name)
        raw_sigma, mask = load_sigma_npz(calib_dir / "raw_sigma" / args.modality / frame_name.replace(".png", ".npz"), sigma_key)
        abs_error = compute_error(render_rgb, gt_rgb, args.modality, args.rgb_error_mode)
        sigma_norm, sigma_safe = normalize_sigma(raw_sigma, mask, normalization, args.eps)

        num_pixels = abs_error.size
        sample_count = max(1, min(int(math.ceil(args.calib_sample_ratio * num_pixels)), num_pixels))
        valid_flat = (np.isfinite(abs_error) & np.isfinite(sigma_safe) & mask).reshape(-1)
        valid_indices = np.flatnonzero(valid_flat)
        if valid_indices.size == 0:
            raise ValueError(f"No valid calibration pixels for {frame_name}")
        sample_count = min(sample_count, valid_indices.size)
        sampled_indices = rng.choice(valid_indices, size=sample_count, replace=False)
        flat_error = abs_error.reshape(-1)
        flat_sigma = sigma_safe.reshape(-1)
        sampled_scores = flat_error[sampled_indices] / flat_sigma[sampled_indices]
        calib_scores.extend(sampled_scores.tolist())
        total_calib_sampled_pixels += sample_count

        frame_stem = frame_name.replace(".png", "")
        zero_uncertainty = np.zeros_like(sigma_norm, dtype=np.float32)
        save_colormap(abs_error, calib_out_dirs["abs_error"] / f"{frame_stem}.png", cmap_name="turbo")
        save_grayscale(raw_sigma, calib_out_dirs["raw_sigma"] / f"{frame_stem}.png", mask=mask)
        save_sigma_normalized(sigma_norm, calib_out_dirs["sigma_normalized"] / f"{frame_stem}.png")
        save_colormap(zero_uncertainty, calib_out_dirs["uncertainty"] / f"{frame_stem}.png", cmap_name="turbo")
        save_panel(
            calib_out_dirs["panels"] / f"{frame_stem}.png",
            gt_rgb,
            render_rgb,
            raw_sigma,
            abs_error,
            sigma_norm,
            zero_uncertainty,
            mask,
        )
        save_npz(
            str(calib_out_dirs["arrays"] / f"{frame_stem}.npz"),
            abs_error=abs_error.astype(np.float32),
            raw_sigma=raw_sigma.astype(np.float32),
            sigma_normalized=sigma_norm.astype(np.float32),
            mask=mask.astype(np.bool_),
        )
        calib_frame_stats[frame_name] = {
            "num_pixels": int(num_pixels),
            "sampled_pixels": int(sample_count),
            "score_mean": float(np.mean(sampled_scores)),
            "score_std": float(np.std(sampled_scores)),
            "raw_sigma": finite_stats(raw_sigma, mask),
            "sigma_normalized": finite_stats(sigma_norm, mask),
            "abs_error": finite_stats(abs_error, mask),
        }
        print(
            f"  [calib {frame_idx:>3}/{len(calib_frames)}] "
            f"{frame_name} | pixels={num_pixels} sampled={sample_count} "
            f"score_mean={calib_frame_stats[frame_name]['score_mean']:.6f}"
        )

    q_hat, quantile_level = conformal_quantile(calib_scores, args.alpha)
    print(f"Calibration scores : {len(calib_scores)}")
    print(f"Quantile level     : {quantile_level:.6f}")
    print(f"q_hat              : {q_hat:.6f}")

    test_out_dirs = prepare_output_dirs(run_dir, "test", iteration, analysis_name)
    per_view_metrics = {}
    print(f"Test outputs       : {test_out_dirs['base']}")
    print("Processing test frames...")

    for frame_idx, frame_name in enumerate(test_frames, start=1):
        render_rgb = load_rgb(test_dir / "render" / frame_name)
        gt_rgb = load_rgb(test_dir / "gt" / frame_name)
        raw_sigma, mask = load_sigma_npz(test_dir / "raw_sigma" / args.modality / frame_name.replace(".png", ".npz"), sigma_key)
        abs_error = compute_error(render_rgb, gt_rgb, args.modality, args.rgb_error_mode)
        sigma_norm, _ = normalize_sigma(raw_sigma, mask, normalization, args.eps)
        interval_half_width = q_hat * sigma_norm
        uncertainty_full_width = 2.0 * interval_half_width
        within_interval = abs_error <= interval_half_width
        valid = np.isfinite(abs_error) & np.isfinite(uncertainty_full_width) & mask

        corr = pearson_corr(abs_error[valid], uncertainty_full_width[valid])
        view_ause = ause(abs_error, uncertainty_full_width, mask=valid)
        per_view_metrics[frame_name] = {
            "num_pixels": int(abs_error.size),
            "num_valid_pixels": int(valid.sum()),
            "coverage": float(np.mean(within_interval[valid])) if np.any(valid) else 0.0,
            "mean_full_width": float(np.mean(uncertainty_full_width[valid])) if np.any(valid) else 0.0,
            "full_width_std": float(np.std(uncertainty_full_width[valid])) if np.any(valid) else 0.0,
            "ae_uncertainty_corr": corr,
            "ause": view_ause,
            "raw_sigma": finite_stats(raw_sigma, mask),
            "sigma_normalized": finite_stats(sigma_norm, mask),
            "abs_error": finite_stats(abs_error, mask),
        }
        print(
            f"  [test  {frame_idx:>3}/{len(test_frames)}] "
            f"{frame_name} | coverage={per_view_metrics[frame_name]['coverage']:.6f} "
            f"mean_full_width_2q_u={per_view_metrics[frame_name]['mean_full_width']:.6f} "
            f"corr={corr:.6f} ause={view_ause:.6f}"
        )

        frame_stem = frame_name.replace(".png", "")
        save_colormap(abs_error, test_out_dirs["abs_error"] / f"{frame_stem}.png", cmap_name="turbo")
        save_grayscale(raw_sigma, test_out_dirs["raw_sigma"] / f"{frame_stem}.png", mask=mask)
        save_sigma_normalized(sigma_norm, test_out_dirs["sigma_normalized"] / f"{frame_stem}.png")
        save_colormap(uncertainty_full_width, test_out_dirs["uncertainty"] / f"{frame_stem}.png", cmap_name="turbo")
        save_panel(
            test_out_dirs["panels"] / f"{frame_stem}.png",
            gt_rgb,
            render_rgb,
            raw_sigma,
            abs_error,
            sigma_norm,
            uncertainty_full_width,
            mask,
        )
        save_npz(
            str(test_out_dirs["arrays"] / f"{frame_stem}.npz"),
            abs_error=abs_error.astype(np.float32),
            raw_sigma=raw_sigma.astype(np.float32),
            sigma_normalized=sigma_norm.astype(np.float32),
            interval_half_width=interval_half_width.astype(np.float32),
            uncertainty_full_width=uncertainty_full_width.astype(np.float32),
            within_interval=within_interval.astype(np.bool_),
            mask=mask.astype(np.bool_),
        )

    summary_dir = run_dir / "conformal" / analysis_name / f"ours_{iteration}"
    ensure_dir(str(summary_dir))
    per_view_summary = {
        "coverage_mean": metric_mean(per_view_metrics, "coverage"),
        "coverage_std": metric_std(per_view_metrics, "coverage"),
        "mean_full_width_mean": metric_mean(per_view_metrics, "mean_full_width"),
        "mean_full_width_std": metric_std(per_view_metrics, "mean_full_width"),
        "full_width_std_mean": metric_mean(per_view_metrics, "full_width_std"),
        "full_width_std_std": metric_std(per_view_metrics, "full_width_std"),
        "ae_uncertainty_corr_mean": metric_mean(per_view_metrics, "ae_uncertainty_corr"),
        "ae_uncertainty_corr_std": metric_std(per_view_metrics, "ae_uncertainty_corr"),
        "ause_mean": metric_mean(per_view_metrics, "ause"),
        "ause_std": metric_std(per_view_metrics, "ause"),
    }
    summary = {
        "run_dir": str(run_dir),
        "iteration": iteration,
        "modality": args.modality,
        "sigma_key": sigma_key,
        "rgb_error_mode": args.rgb_error_mode,
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "calib_sample_ratio": args.calib_sample_ratio,
        "seed": args.seed,
        "eps": args.eps,
        "num_calib_frames": len(calib_frames),
        "num_test_frames": len(test_frames),
        "num_calibration_scores": len(calib_scores),
        "quantile_level": quantile_level,
        "q_hat": q_hat,
        "sigma_normalization": normalization,
        "test_pixel_coverage": per_view_summary["coverage_mean"],
        "test_pixel_coverage_std": per_view_summary["coverage_std"],
        "test_mean_full_width": per_view_summary["mean_full_width_mean"],
        "test_mean_full_width_std_per_view": per_view_summary["mean_full_width_std"],
        "test_full_width_std": per_view_summary["full_width_std_mean"],
        "test_full_width_std_std_per_view": per_view_summary["full_width_std_std"],
        "mean_per_view_ae_uncertainty_corr": per_view_summary["ae_uncertainty_corr_mean"],
        "std_per_view_ae_uncertainty_corr": per_view_summary["ae_uncertainty_corr_std"],
        "mean_per_view_ause": per_view_summary["ause_mean"],
        "std_per_view_ause": per_view_summary["ause_std"],
        "per_view_summary": per_view_summary,
        "calib_frame_stats": calib_frame_stats,
        "test_frame_metrics": per_view_metrics,
    }
    with open(summary_dir / "metrics.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("-" * 72)
    print(f"Total sampled calib pixels : {total_calib_sampled_pixels}")
    print(f"Mean per-view coverage     : {summary['test_pixel_coverage']:.6f} ± {summary['test_pixel_coverage_std']:.6f}")
    print(f"Mean per-view full width   : {summary['test_mean_full_width']:.6f} ± {summary['test_mean_full_width_std_per_view']:.6f}")
    print(f"Mean per-view full width std: {summary['test_full_width_std']:.6f} ± {summary['test_full_width_std_std_per_view']:.6f}")
    print(f"Mean per-view AE corr      : {summary['mean_per_view_ae_uncertainty_corr']:.6f} ± {summary['std_per_view_ae_uncertainty_corr']:.6f}")
    print(f"Mean per-view AUSE         : {summary['mean_per_view_ause']:.6f} ± {summary['std_per_view_ause']:.6f}")
    print(f"Summary metrics saved to   : {summary_dir / 'metrics.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
