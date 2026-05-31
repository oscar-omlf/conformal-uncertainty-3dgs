"""
Standalone PSNR-vs-train-views plotting for active-learning runs.

Reads existing round folders:
  <al_root>/<method>/seed_<seed>/round_<rr>/results.json
  <al_root>/<method>/seed_<seed>/round_<rr>/splits/train.txt

Writes:
  <out_dir>/psnr_vs_train_views_clean.png
  <out_dir>/psnr_vs_train_views_shaded.png
  <out_dir>/psnr_vs_train_views.csv

Example:
  python scripts/active_learning/plot_psnr_curves.py \
    --al_root /scratch-shared/$USER/output/active_learning_mipnerf/garden_popgs20_fixed \
    --title "Garden, MipNeRF360 20-view"
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "conformal_color": "Conformal color",
    "conformal_sensitivity": "Conformal sensitivity",
    "conformal_visibility": "Conformal visibility",
    "raw_sensitivity": "Raw sensitivity",
    "uniform": "Uniform",
    "random": "Random",
}

COLORS = {
    "conformal_color": "#1f77b4",
    "conformal_sensitivity": "#ff7f0e",
    "conformal_visibility": "#2ca02c",
    "raw_sensitivity": "#d62728",
    "uniform": "#7e57c2",
    "random": "#7f7f7f",
}

BASELINE_POINTS = [
    {
        "label": "FisherRF avg. (20 views)",
        "x": 20.0,
        "y": 20.89,
        "color": "#111111",
        "marker": "X",
    },
    {
        "label": "POP-GS avg. (20 views)",
        "x": 20.0,
        "y": 20.568,
        "color": "#e91e63",
        "marker": "P",
    },
]


def read_train_count(round_dir):
    path = round_dir / "splits" / "train.txt"
    if not path.exists():
        return None
    with open(path, "r") as handle:
        return sum(1 for line in handle if line.strip())


def read_psnr(round_dir):
    path = round_dir / "results.json"
    if not path.exists():
        return None
    with open(path, "r") as handle:
        data = json.load(handle)
    for payload in data.values():
        if isinstance(payload, dict) and payload.get("PSNR") is not None:
            return float(payload["PSNR"])
    return None


def collect_rows(al_root, methods=None):
    rows = []
    method_dirs = [p for p in sorted(al_root.iterdir()) if p.is_dir()]
    if methods:
        keep = set(methods)
        method_dirs = [p for p in method_dirs if p.name in keep]

    for method_dir in method_dirs:
        method = method_dir.name
        for seed_dir in sorted(p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            seed = seed_dir.name.replace("seed_", "")
            for round_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("round_")):
                psnr = read_psnr(round_dir)
                train_views = read_train_count(round_dir)
                if psnr is None or train_views is None:
                    continue
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "round": int(round_dir.name.replace("round_", "")),
                        "train_views": train_views,
                        "psnr": psnr,
                    }
                )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["train_views"])].append(row["psnr"])

    summary = []
    for (method, train_views), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        summary.append(
            {
                "method": method,
                "train_views": train_views,
                "mean_psnr": mean,
                "std_psnr": var ** 0.5,
                "n": len(values),
            }
        )
    return summary


def write_csv(path, rows, summary):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "method", "seed", "round", "train_views", "psnr", "mean_psnr", "std_psnr", "n"])
        for row in rows:
            writer.writerow(["point", row["method"], row["seed"], row["round"], row["train_views"], row["psnr"], "", "", ""])
        for row in summary:
            writer.writerow(["summary", row["method"], "", "", row["train_views"], "", row["mean_psnr"], row["std_psnr"], row["n"]])


def series_from_summary(summary):
    by_method = defaultdict(list)
    for row in summary:
        by_method[row["method"]].append(row)
    for method in by_method:
        by_method[method] = sorted(by_method[method], key=lambda row: row["train_views"])
    return by_method


def style_axes(ax, title):
    ax.set_title(title, fontsize=18, weight="bold", pad=12)
    ax.set_xlabel("Training views", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="major", color="#d0d0d0", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_clean(summary, out_path, title, show_points, show_baselines):
    series = series_from_summary(summary)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for method, rows in sorted(series.items()):
        xs = [row["train_views"] for row in rows]
        ys = [row["mean_psnr"] for row in rows]
        label = LABELS.get(method, method.replace("_", " "))
        color = COLORS.get(method)
        ax.plot(xs, ys, linewidth=3.0, marker="o" if show_points else None, markersize=6, label=label, color=color)

    style_axes(ax, title)
    if show_baselines:
        add_paper_reference_points(ax)

    ax.legend(
        frameon=True,
        fontsize=10,
        loc="upper left",
        framealpha=0.92,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_shaded(summary, out_path, title, show_points, show_baselines):
    series = series_from_summary(summary)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    for method, rows in sorted(series.items()):
        xs = [row["train_views"] for row in rows]
        ys = [row["mean_psnr"] for row in rows]
        std = [row["std_psnr"] for row in rows]
        lo = [y - s for y, s in zip(ys, std)]
        hi = [y + s for y, s in zip(ys, std)]
        label = LABELS.get(method, method.replace("_", " "))
        color = COLORS.get(method)
        ax.plot(xs, ys, linewidth=3.0, marker="o" if show_points else None, markersize=6, label=label, color=color)
        if any(s > 0 for s in std):
            ax.fill_between(xs, lo, hi, color=color, alpha=0.16, linewidth=0)

    style_axes(ax, title)
    if show_baselines:
        add_paper_reference_points(ax)

    ax.legend(
        frameon=True,
        fontsize=10,
        loc="upper left",
        framealpha=0.92,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

def add_paper_reference_points(ax):
    # Mip-NeRF360 average 20-view values from prior work, not per-scene Bicycle/Garden values.
    refs = BASELINE_POINTS

    # Force axes to include the reference points.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, 3.8), max(x1, 20.6))
    ax.set_ylim(min(y0, min(r["y"] for r in refs) - 0.4), max(y1, max(r["y"] for r in refs) + 0.4))

    for ref in refs:
        ax.scatter(
            ref["x"],
            ref["y"],
            marker=ref["marker"],
            s=220,
            color=ref["color"],
            edgecolor="white",
            linewidth=1.4,
            zorder=50,
            clip_on=False,
            label=ref["label"],
        )
        ax.axhline(
            ref["y"],
            color=ref["color"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.45,
            zorder=1,
        )

    ax.text(
        0.98,
        0.03,
        "FisherRF / POP-GS are Mip-NeRF360 averages",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    
def main():
    parser = argparse.ArgumentParser(description="Plot PSNR vs train views from existing AL round folders")
    parser.add_argument("--al_root", required=True, type=Path)
    parser.add_argument("--out_dir", default=None, type=Path)
    parser.add_argument("--title", default="PSNR vs Training Views")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--no_points", action="store_true")
    parser.add_argument("--no_baselines", action="store_true", help="Do not plot FisherRF / POP-GS reference markers.")
    args = parser.parse_args()

    al_root = args.al_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else al_root / "figures_poster"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(al_root, args.methods)
    if not rows:
        raise SystemExit(f"No completed PSNR results found under {al_root}")
    summary = summarize(rows)

    write_csv(out_dir / "psnr_vs_train_views.csv", rows, summary)
    plot_clean(summary, out_dir / "psnr_vs_train_views_clean.png", args.title, not args.no_points, not args.no_baselines)
    plot_shaded(summary, out_dir / "psnr_vs_train_views_shaded.png", args.title, not args.no_points, not args.no_baselines)

    print(f"Wrote plots to {out_dir}")
    print(f"  {out_dir / 'psnr_vs_train_views_clean.png'}")
    print(f"  {out_dir / 'psnr_vs_train_views_shaded.png'}")
    print(f"  {out_dir / 'psnr_vs_train_views.csv'}")


if __name__ == "__main__":
    main()
