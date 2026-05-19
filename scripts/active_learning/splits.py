"""
Create and update active-learning split manifests.

The split layout is:
  train.txt      views currently used for 3DGS training
  calib.txt      fixed calibration views for conformal evaluation
  test.txt       fixed held-out test views, never selected
  candidate.txt  selectable pool for the next AL round

Selection methods:
  random      random candidate views
  uniform     evenly spaced candidate views in image order
  conformal_color       color conformal mean interval width
  conformal_visibility  visibility conformal mean interval width
  conformal_sensitivity sensitivity conformal mean interval width
  raw_sensitivity       PUP-style raw Fisher/sensitivity baseline
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_extrinsics_text


ACQUISITION_KEYS = {
    "fisher": "raw_sensitivity",
    "pup": "raw_sensitivity",
    "raw_sensitivity": "raw_sensitivity",
    "raw_color": "raw_color",
    "raw_visibility": "raw_visibility",
    "raw_combined": "raw_combined",
    "sensitivity": "conformal_sensitivity",
    "color": "conformal_color",
    "visibility": "conformal_visibility",
    "combined": "conformal_combined",
    "conformal_sensitivity": "conformal_sensitivity",
    "conformal_color": "conformal_color",
    "conformal_visibility": "conformal_visibility",
    "conformal_combined": "conformal_combined",
}


def read_names(path):
    with open(path, "r") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_names(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(names))
        if names:
            handle.write("\n")


def list_images(source_path, images):
    image_dir = Path(source_path).resolve() / images
    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    return sorted(path.name for path in image_dir.iterdir() if path.is_file())


def count_from_arg(value, total, name):
    if isinstance(value, str) and value.endswith("%"):
        frac = float(value[:-1]) / 100.0
        return max(1, int(round(frac * total)))
    parsed = float(value)
    if 0.0 < parsed < 1.0:
        return max(1, int(round(parsed * total)))
    count = int(parsed)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def load_camera_centers(source_path):
    sparse_dir = Path(source_path).resolve() / "sparse" / "0"
    try:
        extrinsics = read_extrinsics_binary(sparse_dir / "images.bin")
    except Exception:
        extrinsics = read_extrinsics_text(sparse_dir / "images.txt")
    centers = {}
    for extr in extrinsics.values():
        rotation = qvec2rotmat(extr.qvec)
        centers[extr.name] = (-rotation.T @ extr.tvec).astype(np.float32)
    return centers


def farthest_point_train_indices(images, pool_idx, count, source_path, seed, first_mode):
    if count <= 0:
        return []
    if count > len(pool_idx):
        raise ValueError(f"Cannot choose {count} train views from pool of {len(pool_idx)}")
    rng = np.random.default_rng(seed)
    centers_by_name = load_camera_centers(source_path)
    missing = [images[idx] for idx in pool_idx if images[idx] not in centers_by_name]
    if missing:
        raise ValueError(f"Missing camera poses for FPS init images: {missing[:10]}")

    selected = [pool_idx[int(rng.integers(len(pool_idx)))] if first_mode == "random" else pool_idx[0]]
    remaining = [idx for idx in pool_idx if idx not in selected]
    while len(selected) < count:
        selected_centers = np.stack([centers_by_name[images[idx]] for idx in selected], axis=0)
        remaining_centers = np.stack([centers_by_name[images[idx]] for idx in remaining], axis=0)
        distances = np.linalg.norm(remaining_centers[:, None, :] - selected_centers[None, :, :], axis=2)
        next_idx = int(np.argmax(distances.min(axis=1)))
        selected.append(remaining.pop(next_idx))
    return sorted(selected)


def choose_initial_train_indices(args, images, train_pool, init_train_n):
    if args.init_method == "random":
        rng = np.random.default_rng(args.seed)
        return sorted(rng.permutation(train_pool).tolist()[:init_train_n])
    if args.init_method == "random_fps":
        return farthest_point_train_indices(images, train_pool, init_train_n, args.source_path, args.seed, "random")
    if args.init_method == "first_fps":
        return farthest_point_train_indices(images, train_pool, init_train_n, args.source_path, args.seed, "first")
    raise ValueError(f"Unknown init_method={args.init_method}")


def write_summary(output_dir, payload):
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(output_dir / "summary.txt", "w") as handle:
        for key in ("train", "calib", "test", "candidate"):
            names = payload.get(key, [])
            handle.write(f"{key}: {len(names)}\n")


def init_splits(args):
    images = list_images(args.source_path, args.images)
    total = len(images)
    if total == 0:
        raise ValueError("No images found.")

    init_train_n = count_from_arg(args.init_train, total, "init_train")
    calib_n = count_from_arg(args.calib, total, "calib")
    test_n = count_from_arg(args.test, total, "test")
    if args.test_mode == "llffhold":
        test_idx = [idx for idx in range(total) if idx % args.llffhold == 0]
        train_pool = [idx for idx in range(total) if idx not in set(test_idx)]
        if init_train_n + calib_n >= len(train_pool):
            raise ValueError(
                f"init_train+calib must leave candidates: "
                f"{init_train_n}+{calib_n} >= {len(train_pool)}"
            )
        train_idx = choose_initial_train_indices(args, images, train_pool, init_train_n)
        rng = np.random.default_rng(args.seed)
        calib_pool = [idx for idx in train_pool if idx not in set(train_idx)]
        calib_idx = sorted(rng.permutation(calib_pool).tolist()[:calib_n])
        used = set(train_idx) | set(calib_idx) | set(test_idx)
        candidate_idx = [idx for idx in range(total) if idx not in used]
    elif init_train_n + calib_n + test_n >= total:
        raise ValueError(
            f"init_train+calib+test must leave candidates: "
            f"{init_train_n}+{calib_n}+{test_n} >= {total}"
        )
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(total).tolist()
        test_idx = sorted(perm[:test_n])
        train_pool = [idx for idx in range(total) if idx not in set(test_idx)]
        train_idx = choose_initial_train_indices(args, images, train_pool, init_train_n)
        calib_pool = [idx for idx in train_pool if idx not in set(train_idx)]
        calib_idx = sorted(rng.permutation(calib_pool).tolist()[:calib_n])
        used = set(train_idx) | set(calib_idx) | set(test_idx)
        candidate_idx = [idx for idx in range(total) if idx not in used]

    split_map = {
        "train": [images[idx] for idx in train_idx],
        "calib": [images[idx] for idx in calib_idx],
        "test": [images[idx] for idx in test_idx],
        "candidate": [images[idx] for idx in candidate_idx],
    }
    output_dir = Path(args.output_dir).resolve()
    for split_name, names in split_map.items():
        write_names(output_dir / f"{split_name}.txt", names)

    payload = {
        "mode": "init",
        "source_path": str(Path(args.source_path).resolve()),
        "images": args.images,
        "seed": args.seed,
        "init_train": args.init_train,
        "calib": args.calib,
        "test": args.test,
        "test_mode": args.test_mode,
        "llffhold": args.llffhold,
        "init_method": args.init_method,
        **split_map,
    }
    write_summary(output_dir, payload)
    print(f"Wrote initial AL splits to {output_dir}")


def load_scores(rankings_path, method):
    key = ACQUISITION_KEYS[method]
    with open(rankings_path, "r") as handle:
        payload = json.load(handle)
    rankings = payload.get("rankings", {})
    if key not in rankings:
        raise KeyError(f"{key} not found in {rankings_path}. Available: {sorted(rankings)}")
    return [(entry["frame"], float(entry.get("score", 0.0))) for entry in rankings[key]]


def image_index_lookup(names):
    return {name: idx for idx, name in enumerate(sorted(names))}


def min_index_distance(name, reference_names, index_by_name):
    if name not in index_by_name or not reference_names:
        return None
    idx = index_by_name[name]
    distances = [abs(idx - index_by_name[ref]) for ref in reference_names if ref in index_by_name]
    return min(distances) if distances else None


def nearest_camera_distance(name, reference_names, centers_by_name):
    if name not in centers_by_name or not reference_names:
        return None
    ref_centers = [centers_by_name[ref] for ref in reference_names if ref in centers_by_name]
    if not ref_centers:
        return None
    distances = np.linalg.norm(np.stack(ref_centers, axis=0) - centers_by_name[name][None, :], axis=1)
    return float(distances.min())


def auto_camera_distance_scale(names, centers_by_name):
    centers = [centers_by_name[name] for name in names if name in centers_by_name]
    if len(centers) < 2:
        return 1.0
    xyz = np.stack(centers, axis=0)
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    distances[distances <= 0.0] = np.inf
    nearest = distances.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 1.0
    return max(float(np.median(finite)), 1e-6)


def camera_distance_adjusted_score(score, distance, penalty, scale):
    if penalty <= 0.0 or distance is None:
        return score
    penalty_factor = penalty * float(np.exp(-distance / max(scale, 1e-6)))
    return score * max(0.0, 1.0 - penalty_factor)


def select_diverse_ranked(
    ranked,
    candidate,
    train,
    k,
    min_gap,
    camera_penalty,
    centers_by_name=None,
    camera_scale=1.0,
):
    candidate_set = set(candidate)
    all_names = list(dict.fromkeys(train + candidate + [name for name, _ in ranked]))
    index_by_name = image_index_lookup(all_names)
    selected = []
    selected_set = set()
    reference = list(train)
    ranked_candidates = [(name, score) for name, score in ranked if name in candidate_set]
    if ranked and not ranked_candidates:
        ranked_preview = [name for name, _ in ranked[:5]]
        candidate_preview = candidate[:5]
        raise ValueError(
            "No ranked frames matched candidate.txt names. "
            f"ranked preview={ranked_preview}, candidate preview={candidate_preview}. "
            "Check render metadata/view_names and ranking frame names."
        )

    while len(selected) < k and ranked_candidates:
        viable = []
        relaxed = []
        for name, score in ranked_candidates:
            index_distance = min_index_distance(name, reference, index_by_name)
            camera_distance = nearest_camera_distance(name, reference, centers_by_name) if centers_by_name else None
            adjusted = score
            if camera_penalty > 0.0:
                adjusted = camera_distance_adjusted_score(score, camera_distance, camera_penalty, camera_scale)
            diversity_distance = camera_distance if camera_distance is not None else index_distance
            item = (adjusted, score, diversity_distance if diversity_distance is not None else 10**9, name)
            relaxed.append(item)
            if min_gap <= 0 or index_distance is None or index_distance > min_gap:
                viable.append(item)
        pool = viable if viable else relaxed
        _, _, _, chosen = max(pool, key=lambda item: (item[0], item[1], item[2]))
        selected.append(chosen)
        selected_set.add(chosen)
        reference.append(chosen)
        ranked_candidates = [(name, score) for name, score in ranked_candidates if name not in selected_set]
    return selected


def choose_candidates(args, candidate):
    k = min(args.add_k, len(candidate))
    if k <= 0:
        return []
    method = args.method
    if method == "random":
        rng = np.random.default_rng(args.seed + args.round)
        return [candidate[idx] for idx in rng.choice(len(candidate), size=k, replace=False).tolist()]
    if method == "uniform":
        if k == 1:
            return [candidate[len(candidate) // 2]]
        indices = np.linspace(0, len(candidate) - 1, num=k)
        return [candidate[int(round(idx))] for idx in indices]

    ranked_frames = load_scores(Path(args.rankings_path), method)
    centers_by_name = None
    camera_scale = 1.0
    if args.camera_distance_penalty > 0.0:
        centers_by_name = load_camera_centers(args.source_path)
        needed = list(dict.fromkeys(args.train + candidate))
        missing = [name for name in needed if name not in centers_by_name]
        if missing:
            raise ValueError(f"Missing camera poses for camera-distance penalty: {missing[:10]}")
        camera_scale = args.camera_distance_scale
        if camera_scale <= 0.0:
            camera_scale = auto_camera_distance_scale(needed, centers_by_name)
    selected = select_diverse_ranked(
        ranked_frames,
        candidate,
        args.train,
        k,
        args.min_index_gap,
        args.camera_distance_penalty,
        centers_by_name,
        camera_scale,
    )
    if len(selected) < k:
        selected_set = set(selected)
        reference = args.train + selected
        index_by_name = image_index_lookup(args.train + candidate)
        for name in candidate:
            if name in selected_set:
                continue
            distance = min_index_distance(name, reference, index_by_name)
            if args.min_index_gap <= 0 or distance is None or distance > args.min_index_gap:
                selected.append(name)
                selected_set.add(name)
            if len(selected) == k:
                break
        if len(selected) < k:
            selected.extend([name for name in candidate if name not in selected_set][: k - len(selected)])
    args.camera_distance_scale_used = camera_scale if args.camera_distance_penalty > 0.0 else 0.0
    return selected


def update_splits(args):
    if args.min_index_gap > 0 and args.camera_distance_penalty > 0.0:
        raise ValueError("Use either --min_index_gap or --camera_distance_penalty, not both.")
    if not 0.0 <= args.camera_distance_penalty <= 1.0:
        raise ValueError("--camera_distance_penalty must be in [0, 1].")
    if args.camera_distance_penalty > 0.0 and not args.source_path:
        raise ValueError("--source_path is required when --camera_distance_penalty > 0")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    train = read_names(input_dir / "train.txt")
    calib = read_names(input_dir / "calib.txt")
    test = read_names(input_dir / "test.txt")
    candidate = read_names(input_dir / "candidate.txt")

    args.train = train
    selected = choose_candidates(args, candidate)
    selected_set = set(selected)
    next_train = train + selected
    next_candidate = [name for name in candidate if name not in selected_set]

    split_map = {
        "train": next_train,
        "calib": calib,
        "test": test,
        "candidate": next_candidate,
    }
    for split_name, names in split_map.items():
        write_names(output_dir / f"{split_name}.txt", names)

    payload = {
        "mode": "update",
        "input_dir": str(input_dir),
        "method": args.method,
        "round": args.round,
        "seed": args.seed,
        "add_k": args.add_k,
        "min_index_gap": args.min_index_gap,
        "camera_distance_penalty": args.camera_distance_penalty,
        "camera_distance_scale": args.camera_distance_scale,
        "camera_distance_scale_used": getattr(args, "camera_distance_scale_used", 0.0),
        "source_path": str(Path(args.source_path).resolve()) if args.source_path else "",
        "rankings_path": str(Path(args.rankings_path).resolve()) if args.rankings_path else "",
        "selected": selected,
        **split_map,
    }
    write_summary(output_dir, payload)
    print(f"Selected {len(selected)} views with method={args.method}:")
    for name in selected:
        print(f"  {name}")
    print(f"Wrote next AL splits to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare/update active-learning split manifests")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--source_path", required=True)
    init.add_argument("--images", default="images")
    init.add_argument("--output_dir", required=True)
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--init_train", default="10%")
    init.add_argument("--calib", default="10%")
    init.add_argument("--test", default="20%")
    init.add_argument("--test_mode", default="random", choices=["random", "llffhold"])
    init.add_argument("--llffhold", type=int, default=8)
    init.add_argument("--init_method", default="random", choices=["random", "random_fps", "first_fps"])
    init.set_defaults(func=init_splits)

    update = sub.add_parser("update")
    update.add_argument("--input_dir", required=True)
    update.add_argument("--output_dir", required=True)
    update.add_argument(
        "--method",
        required=True,
        choices=[
            "random",
            "uniform",
            "fisher",
            "pup",
            "sensitivity",
            "color",
            "visibility",
            "combined",
            "raw_sensitivity",
            "raw_color",
            "raw_visibility",
            "raw_combined",
            "conformal_sensitivity",
            "conformal_color",
            "conformal_visibility",
            "conformal_combined",
        ],
    )
    update.add_argument("--rankings_path", default="")
    update.add_argument("--add_k", type=int, default=5)
    update.add_argument("--min_index_gap", type=int, default=0)
    update.add_argument(
        "--source_path",
        default="",
        help="Scene path; required for camera-distance diversity.",
    )
    update.add_argument(
        "--camera_distance_penalty",
        type=float,
        default=0.0,
        help="Soft camera-center proximity penalty in [0, 1]. Mutually exclusive with --min_index_gap.",
    )
    update.add_argument(
        "--camera_distance_scale",
        type=float,
        default=0.0,
        help="Distance scale for camera penalty. If <=0, uses median nearest-neighbor camera distance.",
    )
    update.add_argument("--round", type=int, required=True)
    update.add_argument("--seed", type=int, default=0)
    update.set_defaults(func=update_splits)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
