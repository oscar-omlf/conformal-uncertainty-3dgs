"""
Per-Gaussian floater likelihood, inspired by

    "TIDI-GS: Floater Suppression in 3D Gaussian Splatting for Enhanced
     Indoor Scene Fidelity" (arXiv 2601.09291)

We compute a continuous per-Gaussian floater-likelihood score from three
forward-only signals available on a frozen pre-trained 3DGS model:

  1. spatial isolation  d_i  = mean distance to K nearest neighbors in 3D
                              (TIDI-GS uses K=16; we use the existing simple_knn
                               primitive's K=3 mean of squared distances)
  2. opacity            a_i  = gaussians.get_opacity[i]
  3. training visibility C_i = sum_{view, pixel} alpha_i(v, p) * T_i(v, p)
                              (from compute_visibility.py; analogous to TIDI-GS's
                               visibility counter v_i)

A Gaussian is *floater-likely* if it is simultaneously isolated, low-opacity,
and rarely seen during training. We combine via robust z-scoring (median, MAD)
to handle the heavy-tailed distributions of each input:

    z(x) = (x - median(x)) / (1.4826 * MAD(x) + eps)

    score[i] = z(d_i) - z(C_i) - z(a_i)
                  ^isolated      ^low_vis     ^low_opacity

High score => floater-likely. Low (often negative) score => well-supported.

TIDI-GS skipped on a frozen model:
  - learned importance scalar omega_i (requires their training framework)
  - position-gradient EMA (we get this implicitly from compute_sensitivity.py)

Output: <model_path>/uncertainty/floater.npz
  - score                : (P,) composite floater-likelihood
  - score_shifted        : (P,) score - min(score) + eps, for alpha-compositing
  - spatial_isolation    : (P,) sqrt(distCUDA2(xyz)) - mean dist to K=3 NN
  - visibility           : (P,) C_i, copied from visibility.npz if available
  - opacity              : (P,) per-Gaussian opacity in [0, 1]
  - z_isolation/z_vis/z_opacity : (P,) robust z-scored components
  - mad_eps              : float, the MAD floor used
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel

try:
    from simple_knn._C import distCUDA2
except ImportError as exc:
    raise SystemExit(
        "simple_knn is not installed. Build it on a GPU node with "
        "`pip install submodules/simple-knn` before running compute_floater.py."
    ) from exc


def parse_args():
    parser = ArgumentParser(description="Compute per-Gaussian floater likelihood (TIDI-GS-inspired)")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output_name", type=str, default="floater.npz")
    parser.add_argument(
        "--visibility_npz",
        type=str,
        default="",
        help="Path to visibility.npz produced by compute_visibility.py. "
             "If empty, defaults to <model_path>/uncertainty/visibility.npz.",
    )
    parser.add_argument("--mad_eps", type=float, default=1e-6, help="Floor on MAD scale in z-scoring")
    args = get_combined_args(parser)
    return args, model.extract(args)


def robust_zscore(values: np.ndarray, mad_eps: float) -> np.ndarray:
    """Median / MAD z-score. MAD is the median absolute deviation, scaled by
    1.4826 to be a consistent estimator of stdev under a Gaussian."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = max(1.4826 * mad, mad_eps)
    return ((values - med) / scale).astype(np.float32)


def load_visibility(default_path: Path, override_path: str, expected_P: int) -> np.ndarray:
    path = Path(override_path) if override_path else default_path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing visibility.npz at {path}. Run compute_visibility.py first, "
            f"or pass --visibility_npz <path>."
        )
    npz = np.load(path)
    if "visibility" not in npz.files:
        raise KeyError(f"visibility.npz at {path} has no 'visibility' key; saw {npz.files}")
    arr = np.asarray(npz["visibility"], dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_P:
        raise ValueError(
            f"visibility count {arr.shape[0]} does not match P={expected_P}. "
            f"Did you regenerate the trained model since computing visibility?"
        )
    return arr


def main():
    args, dataset = parse_args()

    gaussians = GaussianModel(dataset.sh_degree)
    Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    P = gaussians.get_xyz.shape[0]
    print(f"Computing floater likelihood for P = {P} Gaussians")

    # 1. Spatial isolation via simple_knn K=3 mean squared distance, sqrt'd to
    #    return a distance in scene units.
    with torch.no_grad():
        xyz = gaussians.get_xyz.detach().contiguous()
        mean_sq_dist_k3 = distCUDA2(xyz)
        spatial_isolation = torch.sqrt(torch.clamp_min(mean_sq_dist_k3, 0.0)).cpu().numpy().astype(np.float32)
        opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)

    print(f"  spatial isolation  : min/median/max = "
          f"{spatial_isolation.min():.4g} / {np.median(spatial_isolation):.4g} / {spatial_isolation.max():.4g}")
    print(f"  opacity            : min/median/max = "
          f"{opacity.min():.4g} / {np.median(opacity):.4g} / {opacity.max():.4g}")

    # 2. Training visibility from compute_visibility.py output.
    default_vis_path = Path(dataset.model_path) / "uncertainty" / "visibility.npz"
    visibility = load_visibility(default_vis_path, args.visibility_npz, P)
    print(f"  visibility (C_k)   : min/median/max = "
          f"{visibility.min():.4g} / {np.median(visibility):.4g} / {visibility.max():.4g}")

    # 3. Robust z-score each component.
    z_isolation = robust_zscore(spatial_isolation, args.mad_eps)
    z_vis = robust_zscore(visibility, args.mad_eps)
    z_opacity = robust_zscore(opacity, args.mad_eps)

    # 4. Composite: high score = isolated AND low visibility AND low opacity.
    score = z_isolation - z_vis - z_opacity
    score_shifted = (score - float(score.min()) + args.mad_eps).astype(np.float32)

    print(f"  composite score    : min/median/max = "
          f"{score.min():.4g} / {np.median(score):.4g} / {score.max():.4g}")
    print(f"  top-1% threshold   : "
          f"{np.percentile(score, 99):.4g} ({int((score >= np.percentile(score, 99)).sum())} Gaussians flagged)")

    out_dir = Path(dataset.model_path) / "uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    np.savez_compressed(
        str(out_path),
        score=score.astype(np.float32),
        score_shifted=score_shifted,
        spatial_isolation=spatial_isolation,
        visibility=visibility,
        opacity=opacity,
        z_isolation=z_isolation,
        z_vis=z_vis,
        z_opacity=z_opacity,
        mad_eps=np.float32(args.mad_eps),
        knn_K=np.int32(3),
        signal_combination=np.array(
            "score = z(isolation) - z(visibility) - z(opacity)  [higher = floater-likely]"
        ),
    )
    print(f"Saved floater scores to: {out_path}")


if __name__ == "__main__":
    main()
