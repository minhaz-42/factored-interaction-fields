"""Figure: dataset analysis from derived labels (full text width, one row).

(a) field-magnitude histogram (log-x) with median and mean; (b) per-joint mean magnitude (21 bars, coloured by
group); (c) observability: object/hand field-of-view strata and joint visibility; (d) labelled frames per object."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif.paths import PROJECT_ROOT
from fif.metrics import JOINT_GROUPS
import paperstyle as S
import matplotlib.pyplot as plt

GROUP_COL = {"fingertips": "#1B7F5C", "distal": "#4FAF8A", "intermediate": "#8FC7B0", "proximal": "#6D6DAF", "thumb": "#C77CA8", "wrist": "#D9782D", "palm": "#D4A72C"}
JOINT_NAMES = ["T-tip", "I-tip", "M-tip", "R-tip", "P-tip", "wrist", "T-int", "T-dist", "I-prox", "I-int", "I-dist", "M-prox", "M-int", "M-dist", "R-prox", "R-int", "R-dist", "P-prox", "P-int", "P-dist", "palm"]


def group_of(j):
    return next(g for g, js in JOINT_GROUPS.items() if j in js)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aux", type=Path, default=PROJECT_ROOT / "data/aux/aux_labels_10fps.jsonl"); ap.add_argument("--name", default="fig8_dataset_stats"); a = ap.parse_args()
    S.setup()
    rows = [json.loads(l) for l in open(a.aux) if l.strip()]; lab = [r for r in rows if any(r["field_valid"])]
    mags = np.concatenate([np.asarray(r["field_mag"][k]) for r in lab for k in range(2) if r["field_mag"][k] is not None])
    per_joint = np.array([np.mean([r["field_mag"][k][j] for r in lab for k in range(2) if r["field_mag"][k] is not None]) for j in range(21)])
    vis = np.array([v for r in lab for k in range(2) if r["joint_visible0"][k] is not None for v in r["joint_visible0"][k]])
    of0 = np.array([r["obj_fov_frac"].get("headset0", np.nan) for r in lab]); hf = np.array([r["hand_fov_frac"]["headset0"][k] for r in lab for k in range(2) if r["hand_fov_frac"].get("headset0") and r["hand_fov_frac"]["headset0"][k] is not None])
    per_obj = Counter(r["object_alias"] for r in lab)
    fig, ax = plt.subplots(1, 4, figsize=(S.FULL, 1.85), gridspec_kw={"width_ratios": [1.05, 1.25, 1.0, 1.35]})
    # (a)
    bins = np.logspace(0, np.log10(max(mags.max(), 10)), 50)
    ax[0].hist(mags, bins=bins, weights=np.full(mags.size, 1e-3), color="#B0B0B0", zorder=2); ax[0].set_xscale("log"); ax[0].set_xlabel("field magnitude (mm)"); ax[0].set_ylabel("joints (thousands)")
    top = ax[0].get_ylim()[1]
    for x, lab_, c, dy in ((np.median(mags), "median", "#222222", 0.98), (mags.mean(), "mean", S.FACTORED, 0.84)):
        ax[0].axvline(x, c=c, lw=0.7, ls=(0, (3, 2)), zorder=3); ax[0].text(x * 1.15, top * dy, f"{lab_} {x:.0f} mm", fontsize=5.8, color=c, va="top")
    ax[0].set_xticks([1, 10, 100, 1000]); ax[0].set_xticklabels(["1", "10", "100", "1000"])
    S.title(ax[0], "a", "field magnitude")
    # (b)
    order = np.argsort(per_joint)
    ax[1].barh(range(21), per_joint[order], color=[GROUP_COL[group_of(j)] for j in order], height=0.75, zorder=2)
    ax[1].set_yticks(range(21)); ax[1].set_yticklabels([JOINT_NAMES[j] for j in order], fontsize=4.8); ax[1].set_xlabel("mean magnitude (mm)"); ax[1].tick_params(axis="y", length=0)
    for g, c in GROUP_COL.items(): ax[1].bar(0, 0, color=c, label=g)
    ax[1].legend(fontsize=5.0, loc="lower right", ncol=1, bbox_to_anchor=(1.03, -0.02), handlelength=0.9, labelspacing=0.25); ax[1].set_ylim(-0.7, 20.7)
    S.title(ax[1], "b", "per-joint mean magnitude")
    # (c)
    strata = {"object in view": np.nanmean(of0 >= 0.9), "object partial": np.nanmean((of0 < 0.9) & (of0 >= 0.1)), "object out": np.nanmean(of0 < 0.1), "hand in view": (hf >= 0.9).mean(), "hand partial": ((hf < 0.9) & (hf >= 0.1)).mean(), "hand out": (hf < 0.1).mean(), "joint visible": vis.mean(), "joint occluded/out": 1 - vis.mean()}
    cols = [S.OBJ] * 3 + [S.RIGHT] * 3 + ["#777777"] * 2
    ax[2].bar(range(len(strata)), list(strata.values()), color=cols, width=0.72, zorder=2)
    ax[2].set_xticks(range(len(strata))); ax[2].set_xticklabels(list(strata.keys()), fontsize=5.0, rotation=55, ha="right", rotation_mode="anchor"); ax[2].set_ylabel("fraction"); ax[2].set_ylim(0, 1.08); ax[2].tick_params(axis="x", length=0)
    for i, v in enumerate(strata.values()): ax[2].text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=4.9, color="#333333")
    S.title(ax[2], "c", "observability, reference view")
    # (d)
    objs, cnts = zip(*sorted(per_obj.items(), key=lambda x: -x[1]))
    ax[3].bar(range(len(objs)), np.array(cnts) / 1000, color="#B0B0B0", width=0.75, zorder=2); ax[3].set_xticks(range(len(objs))); ax[3].set_xticklabels(objs, rotation=55, fontsize=4.9, ha="right", rotation_mode="anchor"); ax[3].set_ylabel("labelled frames (thousands)"); ax[3].tick_params(axis="x", length=0)
    S.title(ax[3], "d", "labelled frames per object")
    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.30, wspace=0.42)
    S.save(fig, a.name); print("labelled frames", len(lab), "| joints", mags.size)


if __name__ == "__main__":
    main()
