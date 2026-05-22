import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def build_parser(description):
    parser = ArgumentParser(description=description)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--weight_threshold", type=float, default=1e-3)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_calib", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_candidate", action="store_true")
    parser.add_argument("--top_k", type=int, default=4, help="K for top-K weight extraction (entropy renderer only)")
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


def parse_args(description):
    parser, model, pipeline = build_parser(description)
    args = get_combined_args(parser)
    return args, model.extract(args), pipeline.extract(args)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_use_trained_exp(dataset, gaussians, image_name):
    if not dataset.train_test_exp:
        return False
    if getattr(gaussians, "pretrained_exposures", None) is not None:
        return image_name in gaussians.pretrained_exposures
    return image_name in getattr(gaussians, "exposure_mapping", {})


def load_scene(dataset, iteration):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return gaussians, scene, background


def load_split_manifest(split_dir):
    if not split_dir:
        return {}
    split_dir = os.path.abspath(split_dir)
    split_map = {}
    for split_name in ("train", "calib", "test", "candidate"):
        split_path = os.path.join(split_dir, f"{split_name}.txt")
        if split_name == "candidate" and not os.path.exists(split_path):
            split_map[split_name] = []
            continue
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing split file: {split_path}")
        with open(split_path, "r") as handle:
            split_map[split_name] = [line.strip() for line in handle if line.strip()]
    return split_map


def apply_train_test_crop(view, *tensors):
    if not tensors:
        return []
    if not hasattr(view, "alpha_mask") or view.alpha_mask is None:
        return list(tensors)
    width = view.alpha_mask.shape[-1]
    if view.alpha_mask[..., : width // 2].sum() == 0:
        return [tensor[..., width // 2:] for tensor in tensors]
    if view.alpha_mask[..., width // 2 :].sum() == 0:
        return [tensor[..., : width // 2] for tensor in tensors]
    return list(tensors)


def select_views(scene, split_map, args):
    if not split_map:
        selected = {}
        if not args.skip_train:
            selected["train"] = scene.getTrainCameras()
        test_views = scene.getTestCameras()
        if test_views and not args.skip_test:
            selected["test"] = test_views
        return selected

    test_views_by_name = {view.image_name: view for view in scene.getTestCameras()}
    train_views_by_name = {view.image_name: view for view in scene.getTrainCameras()}
    selected = {}
    if not args.skip_train:
        selected["train"] = [train_views_by_name[name] for name in split_map["train"] if name in train_views_by_name]
    if not args.skip_calib:
        selected["calib"] = [test_views_by_name[name] for name in split_map["calib"] if name in test_views_by_name]
    if not args.skip_test:
        selected["test"] = [test_views_by_name[name] for name in split_map["test"] if name in test_views_by_name]
    candidate_names = split_map.get("candidate", [])
    if candidate_names and not getattr(args, "skip_candidate", False):
        selected["candidate"] = [test_views_by_name[name] for name in candidate_names if name in test_views_by_name]
    return {split_name: views for split_name, views in selected.items() if views}


def build_output_dirs(output_root, run_name, split_name, iteration, modality):
    base_dir = os.path.join(output_root, run_name, split_name, f"ours_{iteration}")
    paths = {
        "base": base_dir,
        "gt": os.path.join(base_dir, "gt"),
        "render": os.path.join(base_dir, "render"),
        "raw_sigma": os.path.join(base_dir, "raw_sigma", modality),
        "preview": os.path.join(base_dir, "preview", modality),
    }
    for path in paths.values():
        ensure_dir(path)
    return paths


def tensor_rgb_to_uint8(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = tensor.permute(1, 2, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def save_rgb_png(tensor, path):
    Image.fromarray(tensor_rgb_to_uint8(tensor), mode="RGB").save(path)


def save_rgb_png_if_missing(tensor, path):
    if os.path.exists(path):
        return
    save_rgb_png(tensor, path)


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


def save_preview_png(array, path, mask=None):
    Image.fromarray(normalize_preview(array, mask=mask), mode="L").save(path)


def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)


def write_metadata(path, metadata):
    merged = {}
    if os.path.exists(path):
        with open(path, "r") as handle:
            merged = json.load(handle)
    modality = metadata.get("modality")
    modality_payload = dict(metadata)
    if modality:
        modality_payload.pop("modality", None)
        merged.setdefault("modalities", {})
        merged["modalities"][modality] = modality_payload
    for key, value in metadata.items():
        if key == "modality":
            continue
        if key not in {"raw_sigma_keys", "preview_key"}:
            merged[key] = value
    with open(path, "w") as handle:
        json.dump(merged, handle, indent=2)


def run_name_from_model_path(model_path):
    return Path(os.path.normpath(model_path)).name
