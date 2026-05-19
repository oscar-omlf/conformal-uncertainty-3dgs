"""
Diagnose camera-distance diversity penalties on completed active-learning runs.

This is read-only: it does not retrain, rerender, or edit old split files. For
each completed acquisition round, it loads the saved candidate ranking, computes
each candidate's nearest COLMAP camera-center distance to the current training
set, and simulates which view would be selected under several penalty values.

Example:
  python scripts/active_learning/diagnose_camera_penalty.py \
    --source_path mipnerf/garden \
    --al_root /scratch-shared/$USER/output/active_learning_mipnerf/garden_popgs20 \
    --method conformal_visibility \
    --seed 0
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.active_learning.splits import (  # noqa: E402
    ACQUISITION_KEYS,
    auto_camera_distance_scale,
    camera_distance_adjusted_score,
    load_camera_centers,
    nearest_camera_distance,
    read_names,
)


def find_rankings(round_dir):
    base = round_dir / "active_learning"
    if not base.exists():
        return None
    candidates = sorted(base.glob("ours_*/view_rankings.json"))
    return candidates[-1] if candidates else None


def load_ranked(path, method):
    key = ACQUISITION_KEYS[method]
    with open(path, "r") as handle:
        payload = json.load(handle)
    rankings = payload.get("rankings", {})
    if key not in rankings:
        raise KeyError(f"{key} not found in {path}. Available: {sorted(rankings)}")
    return [(entry["frame"], float(entry.get("score", 0.0))) for entry in rankings[key]]


def load_selected(next_round_dir):
    summary_path = next_round_dir / "splits" / "summary.json"
    if not summary_path.exists():
        return []
    with open(summary_path, "r") as handle:
        return json.load(handle).get("selected", [])


def original_rank(name, ranked_candidates):
    for idx, (candidate_name, _) in enumerate(ranked_candidates, start=1):
        if candidate_name == name:
            return idx
    return None


def choose_with_penalty(ranked_candidates, train, centers_by_name, penalty, scale):
    best = None
    for name, score in ranked_candidates:
        distance = nearest_camera_distance(name, train, centers_by_name)
        adjusted = camera_distance_adjusted_score(score, distance, penalty, scale)
        item = (adjusted, score, distance if distance is not None else -1.0, name)
        if best is None or item > best:
            best = item
    if best is None:
        return None
    adjusted, score, distance, name = best
    return {
        "name": name,
        "score": score,
        "adjusted_score": adjusted,
        "nearest_camera_distance": distance,
        "original_rank": original_rank(name, ranked_candidates),
    }


def write_outputs(out_dir, rows, penalties):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "camera_penalty_diagnostic.csv"
    fields = [
        "round",
        "train_views",
        "actual_selected",
        "actual_original_rank",
        "actual_nearest_camera_distance",
        "top_unpenalized",
        "top_unpenalized_nearest_camera_distance",
    ]
    for penalty in penalties:
        tag = f"penalty_{penalty:g}"
        fields.extend([f"{tag}_selected", f"{tag}_original_rank", f"{tag}_nearest_camera_distance"])

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    md_path = out_dir / "camera_penalty_diagnostic.md"
    with open(md_path, "w") as handle:
        handle.write("# Camera Penalty Diagnostic\n\n")
        handle.write("| round | train views | actual selected | actual rank | actual cam dist | top unpenalized |")
        for penalty in penalties:
            handle.write(f" penalty {penalty:g} |")
        handle.write("\n")
        handle.write("|---:|---:|---|---:|---:|---|")
        for _ in penalties:
            handle.write("---|")
        handle.write("\n")
        for row in rows:
            handle.write(
                f"| {row['round']} | {row['train_views']} | {row.get('actual_selected', '')} | "
                f"{row.get('actual_original_rank', '')} | {row.get('actual_nearest_camera_distance', ''):.4g} | "
                f"{row.get('top_unpenalized', '')} |"
            )
            for penalty in penalties:
                tag = f"penalty_{penalty:g}"
                handle.write(
                    f" {row.get(f'{tag}_selected', '')} "
                    f"(r={row.get(f'{tag}_original_rank', '')}, "
                    f"d={row.get(f'{tag}_nearest_camera_distance', 0.0):.4g}) |"
                )
            handle.write("\n")

    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Diagnose camera-distance penalties on completed AL runs")
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--al_root", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=sorted(ACQUISITION_KEYS))
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--penalties", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--out_dir", default=None, type=Path)
    args = parser.parse_args()

    run_root = args.al_root / args.method / f"seed_{args.seed}"
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    centers_by_name = load_camera_centers(args.source_path)
    rows = []
    round_dirs = sorted(path for path in run_root.glob("round_*") if path.is_dir())
    for round_dir in round_dirs:
        round_idx = int(round_dir.name.replace("round_", ""))
        next_round_dir = run_root / f"round_{round_idx + 1:02d}"
        rankings_path = find_rankings(round_dir)
        if not rankings_path or not next_round_dir.exists():
            continue

        train = read_names(round_dir / "splits" / "train.txt")
        candidate = read_names(round_dir / "splits" / "candidate.txt")
        candidate_set = set(candidate)
        ranked = load_ranked(rankings_path, args.method)
        ranked_candidates = [(name, score) for name, score in ranked if name in candidate_set]
        if not ranked_candidates:
            continue

        needed = list(dict.fromkeys(train + candidate))
        missing = [name for name in needed if name not in centers_by_name]
        if missing:
            raise ValueError(f"Missing camera poses: {missing[:10]}")
        scale = auto_camera_distance_scale(needed, centers_by_name)
        top_unpenalized = ranked_candidates[0]
        actual_selected = load_selected(next_round_dir)
        actual_name = actual_selected[0] if actual_selected else ""
        actual_distance = nearest_camera_distance(actual_name, train, centers_by_name) if actual_name else None

        row = {
            "round": round_idx,
            "train_views": len(train),
            "camera_distance_scale_used": scale,
            "actual_selected": actual_name,
            "actual_original_rank": original_rank(actual_name, ranked_candidates) if actual_name else "",
            "actual_nearest_camera_distance": actual_distance if actual_distance is not None else 0.0,
            "top_unpenalized": top_unpenalized[0],
            "top_unpenalized_nearest_camera_distance": nearest_camera_distance(top_unpenalized[0], train, centers_by_name),
        }
        for penalty in args.penalties:
            pick = choose_with_penalty(ranked_candidates, train, centers_by_name, penalty, scale)
            tag = f"penalty_{penalty:g}"
            row[f"{tag}_selected"] = pick["name"]
            row[f"{tag}_original_rank"] = pick["original_rank"]
            row[f"{tag}_nearest_camera_distance"] = pick["nearest_camera_distance"]
        rows.append(row)

    if not rows:
        raise SystemExit(f"No completed acquisition rounds found under {run_root}")

    out_dir = args.out_dir or (run_root / "diagnostics")
    csv_path, md_path = write_outputs(out_dir, rows, args.penalties)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print("\nQuick summary:")
    for penalty in args.penalties:
        tag = f"penalty_{penalty:g}"
        changed = sum(1 for row in rows if row.get(f"{tag}_selected") != row.get("top_unpenalized"))
        dists = [row.get(f"{tag}_nearest_camera_distance", 0.0) for row in rows]
        print(f"  penalty={penalty:g}: changed {changed}/{len(rows)} picks, mean nearest-camera distance={np.mean(dists):.4g}")


if __name__ == "__main__":
    main()
