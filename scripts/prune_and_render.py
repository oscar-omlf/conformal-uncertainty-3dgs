"""
Load a trained 3DGS model, drop the top-K% of Gaussians ranked by a per-Gaussian
floater-likelihood score, and re-render all calib + test views.

"Pruning" = setting opacity to ~0 (a very negative pre-sigmoid value), which
makes those Gaussians contribute zero blending weight at every pixel without
needing to surgically resize the parameter tensors / densification state.

Args
----
--model_path     : trained run dir, e.g. /scratch-shared/.../Church
--scores_npz     : npz file with the per-Gaussian score (e.g. floater.npz)
--score_key      : key inside the npz that gives the (P,) score vector
--prune_fraction : float in [0, 1) — fraction of Gaussians to prune (top-K by score)
--output_dir     : where to write the re-rendered RGBs. Pipeline expects layout
                   <output_dir>/{calib,test}/ours_<iter>/{render,gt}/<frame>.png
--iteration      : trained iteration to load (default: latest)
--split_dir      : split manifest dir (same as pipeline)

The "pruned" run dir then has the same layout as the original, so metrics.py
and the conformal scripts can run on it unchanged.
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scripts.render_common import (
    SPARSE_ADAM_AVAILABLE,
    apply_train_test_crop,
    load_scene,
    load_split_manifest,
    safe_use_trained_exp,
    select_views,
)


def parse_args():
    parser = ArgumentParser(description="Prune top-K Gaussians by score and re-render held-out views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scores_npz", required=True, type=str)
    parser.add_argument("--score_key", default="score", type=str)
    parser.add_argument("--prune_fraction", required=True, type=float,
                        help="Fraction of Gaussians to mute (set opacity ≈ 0). 0 = no prune.")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Where to write rendered RGB images. Will be created if absent.")
    parser.add_argument("--skip_train", action="store_true", default=True)
    parser.add_argument("--skip_calib", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    return args, model.extract(args), pipeline.extract(args)


def main():
    args, dataset, pipeline = parse_args()

    if not 0.0 <= args.prune_fraction < 1.0:
        raise ValueError(f"--prune_fraction must be in [0, 1), got {args.prune_fraction}")

    split_map = load_split_manifest(args.split_dir or dataset.split_dir)

    with torch.no_grad():
        gaussians, scene, background = load_scene(dataset, args.iteration)
        P = gaussians.get_xyz.shape[0]
        print(f"Loaded {P} Gaussians at iteration {scene.loaded_iter}.")

        # Load the per-Gaussian score and pick the top-K mask.
        scores_npz = np.load(args.scores_npz)
        if args.score_key not in scores_npz.files:
            raise KeyError(f"{args.score_key} not in {args.scores_npz}; available: {list(scores_npz.files)}")
        scores = np.asarray(scores_npz[args.score_key], dtype=np.float32).reshape(-1)
        if scores.shape[0] != P:
            raise ValueError(f"score length {scores.shape[0]} != number of Gaussians {P}")

        k = int(round(args.prune_fraction * P))
        if k == 0:
            print("[prune] prune_fraction effectively 0 — no Gaussians muted.")
        else:
            # top-K by score: largest scores are pruned.
            threshold = float(np.partition(scores, -k)[-k])
            prune_mask = scores >= threshold
            # If ties push us over k, just take strict top-k via argpartition.
            if prune_mask.sum() > k:
                top_idx = np.argpartition(-scores, k - 1)[:k]
                prune_mask = np.zeros(P, dtype=bool)
                prune_mask[top_idx] = True
            n_pruned = int(prune_mask.sum())
            print(f"[prune] prune_fraction={args.prune_fraction:.4f} -> muting {n_pruned}/{P} Gaussians (threshold={threshold:.4g}).")

            # In-place: replace the underlying _opacity for muted Gaussians with a very
            # negative number so sigmoid maps to ~0. This is cheaper than slicing all
            # the parameter tensors and avoids messing with densification state.
            very_neg = torch.full(
                (n_pruned, 1),
                fill_value=-30.0,
                device=gaussians._opacity.device,
                dtype=gaussians._opacity.dtype,
            )
            mask_t = torch.from_numpy(prune_mask).to(gaussians._opacity.device)
            gaussians._opacity[mask_t] = very_neg

        # Now render every held-out view to the requested output_dir.
        selected_views = select_views(scene, split_map, args)
        for split_name, views in selected_views.items():
            out_split_dir = Path(args.output_dir) / split_name / f"ours_{scene.loaded_iter}"
            (out_split_dir / "render").mkdir(parents=True, exist_ok=True)
            (out_split_dir / "gt").mkdir(parents=True, exist_ok=True)

            for idx, view in enumerate(tqdm(views, desc=f"Render [{split_name}]")):
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
                rendering, gt = apply_train_test_crop(view, render_pkg["render"], gt)

                stem = f"{idx:05d}"
                # render
                img = rendering.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                Image.fromarray(np.rint(img * 255.0).astype(np.uint8)).save(out_split_dir / "render" / f"{stem}.png")
                # gt
                gt_img = gt.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                Image.fromarray(np.rint(gt_img * 255.0).astype(np.uint8)).save(out_split_dir / "gt" / f"{stem}.png")

        print(f"[done] pruned renders saved to {args.output_dir}")


if __name__ == "__main__":
    main()
