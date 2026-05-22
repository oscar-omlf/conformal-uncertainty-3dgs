"""
Generic per-Gaussian scalar renderer.

Given a (P,) scalar per Gaussian (e.g. Fisher log-det, visibility mass), this
script renders, for each calib+test view, a 2D sigma map by alpha-compositing
the scalar through the existing rasterizer using the override_color trick:

    sigma(p) = sum_i w_i(p) * s_i / sum_i w_i(p)        (mean across contributors)
    raw(p)   = sum_i w_i(p) * s_i                       (raw weighted sum)

Used by both the sensitivity (PUP3DGS) and visibility (4DGS-W) flows. Pick the
modality folder name with --modality, the scalar source with --scores_path /
--score_key.
"""

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scripts.render_common import (
    SPARSE_ADAM_AVAILABLE,
    apply_train_test_crop,
    build_output_dirs,
    load_scene,
    load_split_manifest,
    run_name_from_model_path,
    safe_use_trained_exp,
    save_npz,
    save_preview_png,
    save_rgb_png_if_missing,
    select_views,
    write_metadata,
)


def parse_args():
    parser = ArgumentParser(description="Render per-Gaussian scalars to per-pixel sigma maps")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--weight_threshold", type=float, default=1e-3)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_calib", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_candidate", action="store_true")
    parser.add_argument("--modality", required=True, type=str, help="Output folder name (e.g. sensitivity, visibility)")
    parser.add_argument("--scores_path", required=True, type=str, help="Path to .npz with per-Gaussian scalar")
    parser.add_argument("--score_key", required=True, type=str, help="Key inside the .npz holding the (P,) scalar")
    parser.add_argument("--score_transform", type=str, default="none",
                        choices=["none", "log", "shift_positive"],
                        help="Optional transform applied to scores before compositing")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    return args, model.extract(args), pipeline.extract(args)


def load_per_gaussian_scores(scores_path, score_key, num_gaussians):
    data = np.load(scores_path, allow_pickle=True)
    if score_key not in data.files:
        raise KeyError(f"{score_key} not in {scores_path}. Available: {data.files}")
    scores = np.asarray(data[score_key], dtype=np.float32).reshape(-1)
    if scores.shape[0] != num_gaussians:
        raise ValueError(f"Score count {scores.shape[0]} does not match P={num_gaussians} (from {scores_path})")
    return scores


def transform_scores(scores, mode):
    if mode == "none":
        return scores
    if mode == "log":
        return np.log(np.clip(scores, 1e-12, None))
    if mode == "shift_positive":
        return scores - scores.min() + 1e-6
    raise ValueError(mode)


def render_scalar_passes(view, gaussians, pipeline, background, score_tensor, weight_threshold):
    black_background = torch.zeros_like(background)

    def feature_render(feature_n1):
        feature_rgb = feature_n1.expand(-1, 3).contiguous()
        return render(
            view,
            gaussians,
            pipeline,
            black_background,
            override_color=feature_rgb,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            use_trained_exp=False,
            clamp_output=False,
        )["render"][0]

    weight_map = feature_render(torch.ones((score_tensor.shape[0], 1), device=score_tensor.device))
    raw_sum = feature_render(score_tensor.view(-1, 1))

    mask = weight_map > weight_threshold
    sigma_mean = torch.zeros_like(weight_map)
    sigma_mean[mask] = raw_sum[mask] / weight_map[mask]

    return {
        "raw_sum": raw_sum,
        "sigma_mean": sigma_mean,
        "W": weight_map,
        "mask": mask,
    }


def main():
    args, dataset, pipeline = parse_args()
    split_map = load_split_manifest(args.split_dir or dataset.split_dir)

    with torch.no_grad():
        gaussians, scene, background = load_scene(dataset, args.iteration)

        scores_np = load_per_gaussian_scores(args.scores_path, args.score_key, gaussians.get_xyz.shape[0])
        scores_np = transform_scores(scores_np, args.score_transform)
        score_tensor = torch.from_numpy(scores_np).to(device="cuda", dtype=torch.float32)

        selected_views = select_views(scene, split_map, args)
        run_name = run_name_from_model_path(dataset.model_path)

        for split_name, views in selected_views.items():
            out_dirs = build_output_dirs(args.output_root, run_name, split_name, scene.loaded_iter, args.modality)
            metadata = {
                "run_name": run_name,
                "split": split_name,
                "iteration": scene.loaded_iter,
                "model_path": dataset.model_path,
                "source_path": dataset.source_path,
                "split_dir": args.split_dir or dataset.split_dir,
                "weight_threshold": args.weight_threshold,
                "modality": args.modality,
                "scores_path": str(args.scores_path),
                "score_key": args.score_key,
                "score_transform": args.score_transform,
                "view_names": [view.image_name for view in views],
            }

            for idx, view in enumerate(tqdm(views, desc=f"{args.modality} rendering [{split_name}]")):
                use_trained_exp = safe_use_trained_exp(dataset, gaussians, view.image_name)
                render_pkg = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                    use_trained_exp=use_trained_exp,
                )
                gt = view.original_image[0:3, :, :]
                outputs = render_scalar_passes(view, gaussians, pipeline, background, score_tensor, args.weight_threshold)

                rendering, gt = apply_train_test_crop(view, render_pkg["render"], gt)
                cropped = apply_train_test_crop(
                    view,
                    outputs["raw_sum"],
                    outputs["sigma_mean"],
                    outputs["W"],
                    outputs["mask"].float(),
                )
                raw_sum, sigma_mean, weight_map, mask = cropped
                mask_bool = mask > 0.5

                stem = f"{idx:05d}"
                save_rgb_png_if_missing(gt, os.path.join(out_dirs["gt"], f"{stem}.png"))
                save_rgb_png_if_missing(rendering, os.path.join(out_dirs["render"], f"{stem}.png"))
                save_npz(
                    os.path.join(out_dirs["raw_sigma"], f"{stem}.npz"),
                    raw_sum=raw_sum.detach().cpu().numpy().astype(np.float32),
                    sigma_mean=sigma_mean.detach().cpu().numpy().astype(np.float32),
                    W=weight_map.detach().cpu().numpy().astype(np.float32),
                    mask=mask_bool.detach().cpu().numpy().astype(np.bool_),
                )
                save_preview_png(
                    sigma_mean.detach().cpu().numpy(),
                    os.path.join(out_dirs["preview"], f"{stem}.png"),
                    mask=mask_bool.detach().cpu().numpy(),
                )

            metadata["num_views"] = len(views)
            metadata["raw_sigma_keys"] = ["raw_sum", "sigma_mean", "W", "mask"]
            metadata["preview_key"] = "sigma_mean"
            write_metadata(os.path.join(out_dirs["base"], "metadata.json"), metadata)


if __name__ == "__main__":
    main()
