"""Figure: where the factored model gains. Bars = ADE(direct) - ADE(FIF) per difficulty stratum,
for four matched comparisons (stereo fold A, stereo fold B, single view fold A, joint SHOW3D+HOT3D training).
All numbers come from experiments/<run>/metrics_val.json; nothing is typed by hand.
Column-width figure, three panels stacked."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif.paths import EXPERIMENTS_ROOT
import paperstyle as S
import matplotlib.pyplot as plt

PAIRS = [("stereo, fold A", "B3_direct_stereo_rays_A", "FIF_hybrid_stereo_A", S.FACTORED),
         ("stereo, fold B", "B3_direct_stereo_rays_B", "FIF_hybrid_stereo_B", S.ORANGE),
         ("single view, fold A", "B2_direct_mono_A", "FIF_hybrid_mono_A", S.PURPLE),
         ("SHOW3D + HOT3D training, fold A", "T3_B3_joint", "T3_FIF_joint", S.GREY)]
PANELS = [("a", "by field magnitude", "magnitude", [("near(<15)", "< 15 mm"), ("mid(15-60)", "15 to 60 mm"), ("far(>60)", "> 60 mm")]),
          ("b", "by joint visibility in the reference view", "visibility0", [("visible", "visible"), ("occluded_or_outside", "occluded or outside")]),
          ("c", "by fraction of the hand inside the reference image", "hand_fov0", [("in_view", "in view"), ("partial", "partial"), ("out_of_view", "out of view")])]


def strat(run):
    return json.load(open(EXPERIMENTS_ROOT / run / "metrics_val.json"))["stratified"]


def main():
    S.setup()
    R = {r: strat(r) for _, a, b, _ in PAIRS for r in (a, b)}
    fig, axes = plt.subplots(3, 1, figsize=(S.COL, 3.55))
    w = 0.8 / len(PAIRS)
    for ax, (letter, ttl, grp, keys) in zip(axes, PANELS):
        vals = np.array([[R[a][grp][k]["ade_mm"] - R[b][grp][k]["ade_mm"] for k, _ in keys] for _, a, b, _ in PAIRS])
        span = max(vals.max(), 0) - min(vals.min(), 0)
        for i, (lab, a, b, col) in enumerate(PAIRS):
            xs = np.arange(len(keys)) + (i - (len(PAIRS) - 1) / 2) * w
            ax.bar(xs, vals[i], width=w * 0.92, color=col, label=lab if ax is axes[0] else None, zorder=3)
            for x, g in zip(xs, vals[i]):
                ax.text(x, g + (0.02 if g >= 0 else -0.02) * span, f"{g:+.1f}", ha="center", va="bottom" if g >= 0 else "top", fontsize=5.2, color="#333333")
        ax.axhline(0, c="#333333", lw=0.5, zorder=2)
        lo, hi = min(vals.min(), 0), max(vals.max(), 0); ax.set_ylim(lo - 0.16 * span, hi + 0.22 * span)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys]); ax.set_xlim(-0.6, len(keys) - 0.4)
        S.n_labels(ax, range(len(keys)), [R[PAIRS[0][1]][grp][k].get("n", 0) for k, _ in keys], y=-0.30)
        S.title(ax, letter, ttl)
        ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_yticks([t for t in ax.get_yticks() if lo - 0.16 * span <= t <= hi + 0.22 * span])
    fig.text(0.012, 0.47, "ADE(direct) minus ADE(factored), mm", rotation=90, va="center", ha="left", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.10, 1.005), ncol=2, fontsize=6.3, columnspacing=1.2)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.855, bottom=0.075, hspace=0.72)
    S.save(fig, "fig6_strata_gain")


if __name__ == "__main__":
    main()
