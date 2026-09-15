"""Figure: fold-A ablation ladder. Horizontal bars of held-out ADE for every fold-A variant, sorted,
with the two seeds of the main comparison marked and the official InterField reference as a line.
Reads tables/all_runs.json (written by make_tables.py). Column-width figure."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif.paths import PROJECT_ROOT
import paperstyle as S
import matplotlib.pyplot as plt

NAMES = {"B2_direct_mono": ("direct, single view", S.DIRECT_LIGHT), "B3b_direct_stereo_norays": ("direct, stereo, no ray encoding", "#6F9CC9"),
         "B3_direct_stereo_rays": ("direct, stereo + ray encoding", S.DIRECT), "FIF_geo_stereo": ("factored, geometric head only", S.GEO),
         "FIF_hybrid_meanfusion": ("factored, equal-weight fusion", "#F2B8AE"), "FIF_hybrid_nonll": ("factored, no likelihood terms", "#EBA093"),
         "FIF_hybrid_noaux": ("factored, no auxiliary losses", "#E68877"), "FIF_hybrid_mono": ("factored, single view", "#DE6E5B"),
         "FIF_hybrid_gatefusion": ("factored, learned-gate fusion", "#D5513C"), "FIF_hybrid_stereo": ("factored, precision fusion (full)", S.FACTORED)}
SEEDS = {"B3_direct_stereo_rays": "B3_direct_stereo_rays_s1", "FIF_hybrid_stereo": "FIF_hybrid_stereo_s1"}


def main():
    S.setup()
    rows = {r["run"].replace("_A", ""): r for r in json.load(open(PROJECT_ROOT / "tables/all_runs.json")) if r.get("fold") == "A"}
    items = sorted([(k, rows[k]) for k in NAMES if k in rows], key=lambda kv: -kv[1]["mean_ade_mm"])
    fig, ax = plt.subplots(figsize=(S.COL, 2.45))
    for i, (k, r) in enumerate(items):
        lab, col = NAMES[k]; y = r["mean_ade_mm"]
        ax.barh(i, y - 40, left=40, color=col, height=0.72, zorder=3)
        txt = f"{y:.2f}"
        if k in SEEDS and SEEDS[k] in rows:
            y2 = rows[SEEDS[k]]["mean_ade_mm"]; ax.plot([y2], [i], marker="|", c="k", ms=6, mew=0.8, zorder=5); txt += f"   seed 1: {y2:.2f}"
        ax.text(y + 0.3, i, txt, va="center", fontsize=5.8, color="#222222")
    ref = 60.47
    ax.axvline(ref, c="#444444", ls=(0, (3, 2)), lw=0.6, zorder=2)
    ax.text(ref - 0.25, -0.55, f"official InterField baseline {ref:.1f}", fontsize=5.6, ha="right", va="bottom", color="#444444")
    ax.set_yticks(range(len(items))); ax.set_yticklabels([NAMES[k][0] for k, _ in items], fontsize=6.3)
    ax.set_xlim(40, 63); ax.set_xticks([40, 45, 50, 55, 60]); ax.set_xlabel("mean ADE on fold A (mm)"); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.445, right=0.99, top=0.97, bottom=0.15)
    S.save(fig, "fig8_ablation_ladder"); print(len(items), "variants")


if __name__ == "__main__":
    main()
