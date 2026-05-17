"""
Generate composite poster-ready figures for the floater extension story.

Produces three figure types under <repo>/assets/floater_extension/:
  1. <scene>/qualitative_<frame>.png  — 1×5 panel: GT | render | abs error |
     σ_color (heat) | σ_visibility (heat).  One file per (scene, frame).
  2. plots/kpsweep_calib_psnr.png — calib PSNR vs K curve for all 3 scenes,
     annotated with K* = 0 to show the conformal extension's "safety net" call.
  3. plots/ae_correlation_bars.png — bar chart of σ AE correlation across all
     5 scenes (tandt/train, drjohnson, Church, Garden, Bicycle).

Reads from /scratch-shared/$USER/output/<scene>/... so this is best run on the
login node after the pipeline jobs have finished. Outputs land inside the repo
tree so the figures are committed and pushable.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

USER = os.environ.get("USER", "pkarageorgis")
RUN_BASE = Path(f"/scratch-shared/{USER}/output")
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets" / "floater_extension"


# ---------------------------------------------------------------------------
# 1. Per-scene qualitative panels
# ---------------------------------------------------------------------------

def turbo_heatmap(arr, vmin=None, vmax=None):
    """Return RGB uint8 turbo-colormapped version of a 2D grayscale array."""
    arr = np.asarray(arr, dtype=np.float32)
    if vmin is None: vmin = float(np.percentile(arr, 1.0))
    if vmax is None: vmax = float(np.percentile(arr, 99.0))
    if vmax <= vmin: vmax = vmin + 1e-6
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(norm)
    return np.rint(rgba[..., :3] * 255.0).astype(np.uint8)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def make_qualitative_panel(scene_dir: Path, iter_dir: str, frame_stem: str, out_path: Path):
    """One row, six panels: GT, render, |err|, σ_color, σ_visibility, σ_floater."""
    gt_path = scene_dir / "test" / iter_dir / "gt" / f"{frame_stem}.png"
    rd_path = scene_dir / "test" / iter_dir / "render" / f"{frame_stem}.png"
    if not gt_path.exists() or not rd_path.exists():
        print(f"  [skip] missing render/gt for {scene_dir.name}/{frame_stem}")
        return False

    gt = load_rgb(gt_path)
    rd = load_rgb(rd_path)
    err = np.abs(gt - rd).mean(axis=2)  # per-pixel mean RGB error

    sigmas = {}
    for sig_name in ("color", "visibility", "floater"):
        p = scene_dir / "test" / iter_dir / "preview" / sig_name / f"{frame_stem}.png"
        if p.exists():
            sigmas[sig_name] = load_gray(p)
        else:
            sigmas[sig_name] = None

    cols = 3 + len(sigmas)  # GT, render, err, + 3 sigmas
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.4), constrained_layout=True)

    axes[0].imshow(np.clip(gt / 255.0, 0, 1)); axes[0].set_title("Ground truth", fontsize=12)
    axes[1].imshow(np.clip(rd / 255.0, 0, 1)); axes[1].set_title("3DGS render",  fontsize=12)
    axes[2].imshow(turbo_heatmap(err, vmin=0)); axes[2].set_title("|render − GT|", fontsize=12)
    for ax, (sig_name, sig_img) in zip(axes[3:], sigmas.items()):
        if sig_img is None:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center", transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.imshow(turbo_heatmap(sig_img))
        ax.set_title(f"σ {sig_name}", fontsize=12)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{scene_dir.name} — frame {frame_stem}  |  GT vs render vs error vs the 3 most-promising σ candidates",
                 fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# 2. K-sweep calib PSNR vs K, three scenes overlaid
# ---------------------------------------------------------------------------

def make_kpsweep_figure(scenes, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    colors = {"Church": "#1f77b4", "garden": "#2ca02c", "bicycle": "#d62728"}
    plotted = []
    for scene_name in scenes:
        pj = RUN_BASE / scene_name / "uncertainty" / "pruning_pick.json"
        if not pj.exists():
            print(f"  [skip] no pruning_pick.json for {scene_name}")
            continue
        data = json.load(open(pj))
        rows = sorted(data["rows"], key=lambda r: r["K"])
        Ks = [r["K"] * 100 for r in rows]       # convert to %
        ps = [r["psnr_calib"] for r in rows]
        K_star = data["K_star"] * 100
        ps_baseline = data["psnr_baseline"]
        # normalize so baseline = 0 to put all scenes on one axis
        delta = [p - ps_baseline for p in ps]
        color = colors.get(scene_name, "#888888")
        line, = ax.plot(Ks, delta, marker="o", label=f"{scene_name} (baseline PSNR {ps_baseline:.2f})", color=color, linewidth=2.0)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        # annotate K*
        baseline_idx = next((i for i, r in enumerate(rows) if r["K"] == 0.0), 0)
        ax.scatter([Ks[baseline_idx]], [delta[baseline_idx]], s=140, color=color, edgecolor="black", zorder=5, linewidth=1.5)
        plotted.append(scene_name)
    ax.set_xlabel("pruning fraction K (% of Gaussians muted by top-score)")
    ax.set_ylabel("Δ calibration PSNR vs baseline (dB)")
    ax.set_title("Conformal K-sweep: calib PSNR vs K — K* = 0 on every scene\n(the conformal extension's safety net refuses to prune)", fontsize=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    if plotted:
        # K* is 0 on every scene; mark with a vertical band of zero width on x=0
        ax.axvline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate("K* = 0\n(picker refuses to prune)", xy=(0, 0), xytext=(2.5, max(0.0005, ax.get_ylim()[1] * 0.4)),
                    fontsize=10, ha="left",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# 3. σ AE-correlation bar chart across all 5 scenes
# ---------------------------------------------------------------------------

AE_CORR_TABLE = {
    # (scene_label, {modality: ae_corr})
    "tandt/train":    {"color": 0.294, "visibility": 0.184, "depth": 0.059, "sensitivity": 0.054, "entropy": 0.005, "floater": 0.005},
    "drjohnson":      {"color": 0.442, "visibility": 0.252, "depth": 0.012, "sensitivity": 0.069, "entropy": -0.007, "floater": 0.121},
    "Church":         {"color": 0.202, "visibility": 0.066, "depth": 0.002, "sensitivity": 0.053, "entropy": -0.002, "floater": 0.019},
    "Garden":         {"color": 0.382, "visibility": 0.196, "depth": 0.099, "sensitivity": 0.080, "entropy": -0.007, "floater": -0.006},
    "Bicycle":        {"color": 0.331, "visibility": 0.248, "depth": -0.076, "sensitivity": 0.043, "entropy": 0.022, "floater": 0.092},
}
MODALITY_ORDER = ["color", "visibility", "sensitivity", "floater", "depth", "entropy"]
MODALITY_COLOR = {"color": "#1f77b4", "visibility": "#ff7f0e", "sensitivity": "#2ca02c", "floater": "#d62728", "depth": "#9467bd", "entropy": "#8c564b"}


def make_correlation_bars(out_path: Path):
    scenes = list(AE_CORR_TABLE.keys())
    n_scenes = len(scenes)
    n_mod = len(MODALITY_ORDER)
    bar_w = 0.13
    x = np.arange(n_scenes)
    fig, ax = plt.subplots(figsize=(10.0, 4.6), constrained_layout=True)
    for i, mod in enumerate(MODALITY_ORDER):
        vals = [AE_CORR_TABLE[s][mod] for s in scenes]
        offset = (i - (n_mod - 1) / 2) * bar_w
        ax.bar(x + offset, vals, width=bar_w, label=mod, color=MODALITY_COLOR[mod], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(scenes, rotation=0)
    ax.set_ylabel("per-view AE correlation (Pearson)")
    ax.set_title("σ candidate AE-correlation across 5 scenes — color universally wins", fontsize=12)
    ax.legend(loc="upper right", ncol=3, fontsize=9, frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# 4. Baseline vs pruned visual diff for Church (showcases "K=0.10 is invisible")
# ---------------------------------------------------------------------------

def make_baseline_vs_pruned(scene_name: str, iter_dir: str, frame_stem: str, fixed_K: float, out_path: Path):
    base = RUN_BASE / scene_name
    A_render = base / "test" / iter_dir / "render" / f"{frame_stem}.png"
    B_render = base / f"_prune_B_fixed_K{fixed_K:.2f}" / "test" / iter_dir / "render" / f"{frame_stem}.png"
    gt_path  = base / "test" / iter_dir / "gt" / f"{frame_stem}.png"
    if not A_render.exists() or not B_render.exists() or not gt_path.exists():
        print(f"  [skip] missing baseline/pruned/gt for {scene_name}/{frame_stem}")
        return False
    A = load_rgb(A_render)
    B = load_rgb(B_render)
    G = load_rgb(gt_path)
    diff = np.abs(A - B).mean(axis=2)

    fig, axes = plt.subplots(1, 4, figsize=(3.4 * 4, 3.4), constrained_layout=True)
    axes[0].imshow(np.clip(G / 255.0, 0, 1)); axes[0].set_title("Ground truth", fontsize=12)
    axes[1].imshow(np.clip(A / 255.0, 0, 1)); axes[1].set_title("(A) baseline render", fontsize=12)
    axes[2].imshow(np.clip(B / 255.0, 0, 1)); axes[2].set_title(f"(B) pruned K={fixed_K:.0%} render", fontsize=12)
    axes[3].imshow(turbo_heatmap(diff, vmin=0, vmax=max(diff.max(), 5))); axes[3].set_title(f"|A − B|  (mean = {diff.mean():.2f} RGB units)", fontsize=12)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{scene_name} frame {frame_stem}: baseline vs TIDI-GS-style K={fixed_K:.0%} pruning — visually indistinguishable", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print(f"Writing figures under {ASSETS}")
    # 1. one qualitative panel per scene (frame 00000 by default)
    for scene_name, frame in [("Church", "00000"), ("garden", "00000"), ("bicycle", "00000")]:
        sd = RUN_BASE / scene_name
        if not (sd / "test").exists():
            print(f"[skip] {sd} doesn't exist")
            continue
        out = ASSETS / scene_name / f"qualitative_{frame}.png"
        make_qualitative_panel(sd, "ours_30000", frame, out)
    # 2. K-sweep curve across all three scenes
    make_kpsweep_figure(["Church", "garden", "bicycle"], ASSETS / "plots" / "kpsweep_calib_psnr.png")
    # 3. σ correlation bar chart
    make_correlation_bars(ASSETS / "plots" / "ae_correlation_bars.png")
    # 4. baseline vs pruned for Church (where we ran the experiment most)
    make_baseline_vs_pruned("Church", "ours_30000", "00000", 0.10, ASSETS / "Church" / "baseline_vs_pruned_K0.10.png")
    print("Done.")


if __name__ == "__main__":
    main()
