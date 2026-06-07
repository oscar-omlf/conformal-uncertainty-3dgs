"""
Aggregate active-learning archive folders across scenes.

Inputs are compact archive roots such as:
  al_archives/garden_popgs20/
  al_archives/bicycle_popgs20/

Outputs:
  - final_metrics_by_scene.csv
  - final_metrics_mean.csv
  - final_metrics_mean.md
  - psnr_vs_train_views_by_scene.csv
  - psnr_vs_train_views_mean.csv
  - psnr_vs_train_views_std.png
  - psnr_vs_train_views_ci95.png
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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
        "y": 20.568,
        "color": "#111111",
        "marker": "X",
    },
    {
        "label": "POP-GS avg. (20 views)",
        "x": 20.0,
        "y": 21.32,
        "color": "#e91e63",
        "marker": "P",
    },
]

BASELINE_METRICS = [
    {
        "method": "fisherrf_paper_avg",
        "label": "FisherRF avg. (20 views)",
        "psnr_mean": 20.568,
        "ssim_mean": 0.608,
        "lpips_mean": 0.365,
    },
    {
        "method": "popgs_paper_avg",
        "label": "POP-GS avg. (20 views)",
        "psnr_mean": 21.32,
        "ssim_mean": 0.636,
        "lpips_mean": 0.397,
    },
]


def scene_name(scene_dir):
    name = scene_dir.name
    for suffix in ("_popgs20", "_popgs20_camera", "_popgs20_fixed", "_minimal"):
        name = name.replace(suffix, "")
    return name


def read_float(value):
    if value is None or value == "":
        return None
    return float(value)


def mean_std(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None, 0
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var), len(vals)


def ci95(std, n):
    if std is None or n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(n)


def read_train_count(round_dir):
    path = round_dir / "splits" / "train.txt"
    if not path.exists():
        return None
    with open(path, "r") as handle:
        return sum(1 for line in handle if line.strip())


def read_metrics_json(round_dir):
    path = round_dir / "results.json"
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        data = json.load(handle)
    for payload in data.values():
        if isinstance(payload, dict):
            return {
                "psnr": read_float(payload.get("PSNR")),
                "ssim": read_float(payload.get("SSIM")),
                "lpips": read_float(payload.get("LPIPS")),
            }
    return {}


def discover_scene_dirs(archives_root, scene_dirs):
    if scene_dirs:
        return [Path(path).resolve() for path in scene_dirs]
    return sorted(path for path in Path(archives_root).resolve().iterdir() if path.is_dir() and (path / "final_metrics.csv").exists())


def collect_final_rows(scene_dirs, methods):
    rows = []
    method_filter = set(methods) if methods else None
    for scene_dir in scene_dirs:
        final_path = scene_dir / "final_metrics.csv"
        if not final_path.exists():
            continue
        with open(final_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                method = row.get("method", "")
                if method_filter and method not in method_filter:
                    continue
                rows.append(
                    {
                        "scene": scene_name(scene_dir),
                        "method": method,
                        "seed": row.get("seed", ""),
                        "round": int(row.get("round", 0) or 0),
                        "train_views": int(row.get("train_views", 0) or 0),
                        "psnr": read_float(row.get("psnr")),
                        "ssim": read_float(row.get("ssim")),
                        "lpips": read_float(row.get("lpips")),
                    }
                )
    return rows


def summarize_final(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary = []
    for method, method_rows in sorted(grouped.items()):
        psnr_mean, psnr_std, n_psnr = mean_std([row["psnr"] for row in method_rows])
        ssim_mean, ssim_std, n_ssim = mean_std([row["ssim"] for row in method_rows])
        lpips_mean, lpips_std, n_lpips = mean_std([row["lpips"] for row in method_rows])
        scenes = sorted({row["scene"] for row in method_rows})
        summary.append(
            {
                "method": method,
                "label": LABELS.get(method, method.replace("_", " ")),
                "n_scenes": len(scenes),
                "scenes": ",".join(scenes),
                "psnr_mean": psnr_mean,
                "psnr_std": psnr_std,
                "psnr_ci95": ci95(psnr_std, n_psnr),
                "ssim_mean": ssim_mean,
                "ssim_std": ssim_std,
                "ssim_ci95": ci95(ssim_std, n_ssim),
                "lpips_mean": lpips_mean,
                "lpips_std": lpips_std,
                "lpips_ci95": ci95(lpips_std, n_lpips),
            }
        )
    return summary


def collect_curve_rows(scene_dirs, methods):
    rows = []
    method_filter = set(methods) if methods else None
    for scene_dir in scene_dirs:
        scene = scene_name(scene_dir)
        for method_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            method = method_dir.name
            if method in {"figures", "figures_poster"}:
                continue
            if method_filter and method not in method_filter:
                continue
            for seed_dir in sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
                seed = seed_dir.name.replace("seed_", "")
                for round_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir() and path.name.startswith("round_")):
                    metrics = read_metrics_json(round_dir)
                    train_views = read_train_count(round_dir)
                    if train_views is None or metrics.get("psnr") is None:
                        continue
                    rows.append(
                        {
                            "scene": scene,
                            "method": method,
                            "seed": seed,
                            "round": int(round_dir.name.replace("round_", "")),
                            "train_views": train_views,
                            "psnr": metrics["psnr"],
                        }
                    )
    return rows


def summarize_curve(rows):
    # Average duplicate seeds within a scene first, then average across scenes.
    per_scene_grouped = defaultdict(list)
    for row in rows:
        per_scene_grouped[(row["scene"], row["method"], row["train_views"])].append(row["psnr"])

    per_scene = []
    for (scene, method, train_views), values in sorted(per_scene_grouped.items()):
        mean, _, n = mean_std(values)
        per_scene.append({"scene": scene, "method": method, "train_views": train_views, "psnr": mean, "n_seeds": n})

    grouped = defaultdict(list)
    for row in per_scene:
        grouped[(row["method"], row["train_views"])].append(row["psnr"])

    summary = []
    for (method, train_views), values in sorted(grouped.items()):
        mean, std, n = mean_std(values)
        summary.append(
            {
                "method": method,
                "label": LABELS.get(method, method.replace("_", " ")),
                "train_views": train_views,
                "mean_psnr": mean,
                "std_psnr": std,
                "ci95_psnr": ci95(std, n),
                "n_scenes": n,
            }
        )
    return per_scene, summary


def fmt(value, digits):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_mean_std(mean, std, digits):
    if mean is None:
        return ""
    if std is None:
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} ± {fmt(std, digits)}"


def write_dict_csv(path, rows, fields):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_final_markdown(path, rows):
    lines = ["# Final Metrics Averaged Across Scenes", ""]
    lines.append("| method | scenes | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['n_scenes']} | "
            f"{fmt_mean_std(row['psnr_mean'], row['psnr_std'], 3)} | "
            f"{fmt_mean_std(row['ssim_mean'], row['ssim_std'], 4)} | "
            f"{fmt_mean_std(row['lpips_mean'], row['lpips_std'], 4)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def add_baseline_metric_rows(rows):
    enriched = list(rows)
    for baseline in BASELINE_METRICS:
        enriched.append(
            {
                "method": baseline["method"],
                "label": baseline["label"],
                "n_scenes": "paper avg.",
                "scenes": "Mip-NeRF360 paper average",
                "psnr_mean": baseline["psnr_mean"],
                "psnr_std": None,
                "psnr_ci95": None,
                "ssim_mean": baseline["ssim_mean"],
                "ssim_std": None,
                "ssim_ci95": None,
                "lpips_mean": baseline["lpips_mean"],
                "lpips_std": None,
                "lpips_ci95": None,
            }
        )
    return enriched


def svg_color(method):
    return COLORS.get(method, "#333333")


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def rgba(color, alpha):
    r, g, b = hex_to_rgb(color)
    return f"rgba({r},{g},{b},{alpha})"


def plot_curve_svg(summary, out_path, title, shade_key, shade_label, show_baselines):
    by_method = defaultdict(list)
    for row in summary:
        by_method[row["method"]].append(row)

    all_x = [row["train_views"] for row in summary]
    if show_baselines:
        all_x.extend(point["x"] for point in BASELINE_POINTS)
    all_y = []
    for row in summary:
        shade = row[shade_key] or 0.0
        all_y.extend([row["mean_psnr"] - shade, row["mean_psnr"] + shade])
    if show_baselines:
        all_y.extend(point["y"] for point in BASELINE_POINTS)
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = max(0.25, 0.05 * (y_max - y_min))
    y_min -= y_pad
    y_max += y_pad

    width, height = 1100, 700
    left, right, top, bottom = 90, 260, 70, 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x):
        return left + (x - x_min) / max(x_max - x_min, 1) * plot_w

    def sy(y):
        return top + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h

    def polyline(points):
        return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:26px;font-weight:700}.label{font-size:18px}.tick{font-size:14px}.legend{font-size:15px}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="38" text-anchor="middle" class="title">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
    ]

    # Grid and ticks.
    for x in range(int(x_min), int(x_max) + 1, 2):
        px = sx(x)
        lines.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" stroke="#d0d0d0" stroke-opacity="0.55"/>')
        lines.append(f'<text x="{px:.1f}" y="{top + plot_h + 25}" text-anchor="middle" class="tick">{x}</text>')
    y_step = 1.0
    y_tick = math.ceil(y_min)
    while y_tick <= y_max:
        py = sy(y_tick)
        lines.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#d0d0d0" stroke-opacity="0.55"/>')
        lines.append(f'<text x="{left - 12}" y="{py + 5:.1f}" text-anchor="end" class="tick">{y_tick:.0f}</text>')
        y_tick += y_step

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 25}" text-anchor="middle" class="label">Training views</text>')
    lines.append(f'<text x="25" y="{top + plot_h / 2:.1f}" transform="rotate(-90 25 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">PSNR (dB)</text>')

    legend_x = left + plot_w + 35
    legend_y = top + 20
    legend_idx = 0
    for method, method_rows in sorted(by_method.items()):
        method_rows = sorted(method_rows, key=lambda row: row["train_views"])
        color = svg_color(method)
        points = [(row["train_views"], row["mean_psnr"]) for row in method_rows]
        upper = [(row["train_views"], row["mean_psnr"] + (row[shade_key] or 0.0)) for row in method_rows]
        lower = [(row["train_views"], row["mean_psnr"] - (row[shade_key] or 0.0)) for row in reversed(method_rows)]
        band = upper + lower
        if any((row[shade_key] or 0.0) > 0 for row in method_rows):
            lines.append(f'<polygon points="{polyline(band)}" fill="{rgba(color, 0.14)}" stroke="none"/>')
        lines.append(f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="3.2"/>')
        for x, y in points:
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{color}" stroke="white" stroke-width="1"/>')
        ly = legend_y + legend_idx * 28
        lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 28}" y2="{ly}" stroke="{color}" stroke-width="3.2"/>')
        lines.append(f'<circle cx="{legend_x + 14}" cy="{ly}" r="4.5" fill="{color}" stroke="white" stroke-width="1"/>')
        lines.append(f'<text x="{legend_x + 38}" y="{ly + 5}" class="legend">{LABELS.get(method, method.replace("_", " "))}</text>')
        legend_idx += 1

    if show_baselines:
        for point in BASELINE_POINTS:
            px, py = sx(point["x"]), sy(point["y"])
            color = point["color"]
            lines.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="{color}" stroke-width="1.2" stroke-dasharray="4 4" stroke-opacity="0.45"/>')
            marker_text = "×" if point["marker"] == "X" else "✚"
            lines.append(f'<text x="{px:.1f}" y="{py + 7:.1f}" text-anchor="middle" font-size="25" font-weight="700" fill="{color}" stroke="white" stroke-width="0.6">{marker_text}</text>')
            ly = legend_y + legend_idx * 28
            lines.append(f'<text x="{legend_x + 14}" y="{ly + 8}" text-anchor="middle" font-size="22" font-weight="700" fill="{color}">{marker_text}</text>')
            lines.append(f'<text x="{legend_x + 38}" y="{ly + 5}" class="legend">{point["label"]}</text>')
            legend_idx += 1

    lines.append(f'<text x="{left + plot_w - 5}" y="{top + plot_h - 10}" text-anchor="end" class="tick" fill="#555">{shade_label}</text>')
    if show_baselines:
        lines.append(f'<text x="{left + plot_w - 5}" y="{top + plot_h - 30}" text-anchor="end" class="tick" fill="#555">FisherRF / POP-GS are Mip-NeRF360 averages</text>')
    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n")


def add_paper_reference_points(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, 3.8), max(x1, 20.6))
    ax.set_ylim(
        min(y0, min(point["y"] for point in BASELINE_POINTS) - 0.35),
        max(y1, max(point["y"] for point in BASELINE_POINTS) + 0.35),
    )
    for point in BASELINE_POINTS:
        ax.scatter(
            [point["x"]],
            [point["y"]],
            marker=point["marker"],
            s=180,
            color=point["color"],
            edgecolor="white",
            linewidth=1.2,
            zorder=50,
            label=point["label"],
        )
        ax.axhline(point["y"], color=point["color"], linestyle=":", linewidth=1.2, alpha=0.45, zorder=1)
    ax.text(
        0.99,
        0.06,
        "FisherRF / POP-GS are Mip-NeRF360 averages",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )


def plot_curve(summary, out_path, title, shade_key, shade_label, show_baselines):
    if plt is None:
        svg_path = out_path.with_suffix(".svg")
        plot_curve_svg(summary, svg_path, title, shade_key, shade_label, show_baselines)
        return svg_path
    by_method = defaultdict(list)
    for row in summary:
        by_method[row["method"]].append(row)

    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    for method, method_rows in sorted(by_method.items()):
        method_rows = sorted(method_rows, key=lambda row: row["train_views"])
        xs = [row["train_views"] for row in method_rows]
        ys = [row["mean_psnr"] for row in method_rows]
        shade = [row[shade_key] or 0.0 for row in method_rows]
        lo = [y - s for y, s in zip(ys, shade)]
        hi = [y + s for y, s in zip(ys, shade)]
        color = COLORS.get(method)
        label = LABELS.get(method, method.replace("_", " "))
        ax.plot(xs, ys, linewidth=2.8, marker="o", markersize=5.5, color=color, label=label)
        if any(s > 0 for s in shade):
            ax.fill_between(xs, lo, hi, color=color, alpha=0.14, linewidth=0)

    ax.set_title(title, fontsize=17, weight="bold", pad=12)
    ax.set_xlabel("Training views", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_baselines:
        add_paper_reference_points(ax)
    ax.legend(frameon=True, fontsize=10, loc="best")
    ax.text(
        0.99,
        0.02,
        shade_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path


def filter_methods(rows, methods):
    keep = set(methods)
    return [row for row in rows if row["method"] in keep]


def signal_plot_name(method):
    return method.replace("conformal_", "").replace("raw_", "raw_")


def main():
    parser = argparse.ArgumentParser(description="Aggregate AL archives across scenes")
    parser.add_argument("--archives_root", default="al_archives", type=Path)
    parser.add_argument("--scene_dirs", nargs="+", default=None, help="Explicit scene archive dirs. Defaults to all dirs with final_metrics.csv.")
    parser.add_argument("--out_dir", default=None, type=Path)
    parser.add_argument("--methods", nargs="+", default=["conformal_color", "conformal_sensitivity", "conformal_visibility", "raw_sensitivity", "uniform"])
    parser.add_argument("--title", default="MipNeRF360 20-view Active Learning: PSNR vs Training Views")
    parser.add_argument("--no_baselines", action="store_true", help="Do not plot FisherRF / POP-GS reference markers.")
    args = parser.parse_args()

    scene_dirs = discover_scene_dirs(args.archives_root, args.scene_dirs)
    if not scene_dirs:
        raise SystemExit(f"No scene archives found under {args.archives_root}")
    out_dir = args.out_dir.resolve() if args.out_dir else Path(args.archives_root).resolve() / "summary_9scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_rows = collect_final_rows(scene_dirs, args.methods)
    final_summary = summarize_final(final_rows)
    final_summary_for_report = final_summary if args.no_baselines else add_baseline_metric_rows(final_summary)
    curve_rows = collect_curve_rows(scene_dirs, args.methods)
    curve_by_scene, curve_summary = summarize_curve(curve_rows)

    write_dict_csv(
        out_dir / "final_metrics_by_scene.csv",
        final_rows,
        ["scene", "method", "seed", "round", "train_views", "psnr", "ssim", "lpips"],
    )
    write_dict_csv(
        out_dir / "final_metrics_mean.csv",
        final_summary_for_report,
        [
            "method",
            "label",
            "n_scenes",
            "psnr_mean",
            "psnr_std",
            "psnr_ci95",
            "ssim_mean",
            "ssim_std",
            "ssim_ci95",
            "lpips_mean",
            "lpips_std",
            "lpips_ci95",
            "scenes",
        ],
    )
    write_final_markdown(out_dir / "final_metrics_mean.md", final_summary_for_report)
    write_dict_csv(out_dir / "psnr_vs_train_views_by_scene.csv", curve_by_scene, ["scene", "method", "train_views", "psnr", "n_seeds"])
    write_dict_csv(out_dir / "psnr_vs_train_views_mean.csv", curve_summary, ["method", "label", "train_views", "mean_psnr", "std_psnr", "ci95_psnr", "n_scenes"])
    std_plot = plot_curve(curve_summary, out_dir / "psnr_vs_train_views_std.png", args.title, "std_psnr", "Shading: ±1 std across scenes", not args.no_baselines)
    ci_plot = plot_curve(curve_summary, out_dir / "psnr_vs_train_views_ci95.png", args.title, "ci95_psnr", "Shading: 95% CI across scenes", not args.no_baselines)
    per_signal_dir = out_dir / "per_signal"
    per_signal_dir.mkdir(parents=True, exist_ok=True)
    per_signal_with_uniform_dir = out_dir / "per_signal_with_uniform"
    per_signal_with_uniform_dir.mkdir(parents=True, exist_ok=True)
    per_signal_plots = []
    for method in args.methods:
        if method == "uniform":
            continue
        rows_for_method = filter_methods(curve_summary, [method])
        if not rows_for_method:
            continue
        label = LABELS.get(method, method.replace("_", " "))
        basename = signal_plot_name(method)
        per_signal_plots.append(
            plot_curve(
                rows_for_method,
                per_signal_dir / f"psnr_vs_train_views_{basename}_ci95.png",
                f"PSNR vs Training Views ({label})",
                "ci95_psnr",
                "Shading: 95% CI across scenes",
                not args.no_baselines,
            )
        )
        per_signal_plots.append(
            plot_curve(
                rows_for_method,
                per_signal_dir / f"psnr_vs_train_views_{basename}_std.png",
                f"PSNR vs Training Views ({label})",
                "std_psnr",
                "Shading: ±1 std across scenes",
                not args.no_baselines,
            )
        )
        rows_with_uniform = filter_methods(curve_summary, [method, "uniform"])
        if len({row["method"] for row in rows_with_uniform}) > 1:
            per_signal_plots.append(
                plot_curve(
                    rows_with_uniform,
                    per_signal_with_uniform_dir / f"psnr_vs_train_views_{basename}_vs_uniform_ci95.png",
                    f"PSNR vs Training Views ({label} vs Uniform)",
                    "ci95_psnr",
                    "Shading: 95% CI across scenes",
                    not args.no_baselines,
                )
            )
            per_signal_plots.append(
                plot_curve(
                    rows_with_uniform,
                    per_signal_with_uniform_dir / f"psnr_vs_train_views_{basename}_vs_uniform_std.png",
                    f"PSNR vs Training Views ({label} vs Uniform)",
                    "std_psnr",
                    "Shading: ±1 std across scenes",
                    not args.no_baselines,
                )
            )

    print(f"Scenes ({len(scene_dirs)}):")
    for path in scene_dirs:
        print(f"  {scene_name(path)}: {path}")
    print(f"\nWrote summary outputs to {out_dir}")
    print(f"  {out_dir / 'final_metrics_mean.md'}")
    print(f"  {std_plot}")
    print(f"  {ci_plot}")
    print(f"  per-signal plots: {per_signal_dir}")
    print(f"  per-signal + uniform plots: {per_signal_with_uniform_dir}")


if __name__ == "__main__":
    main()
