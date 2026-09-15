"""Graphical abstract (Elsevier: at least 1328 x 531 px). Left: a held-out SHOW3D frame with the
ground-truth field (grey) and the factored prediction (red). Middle: the factorisation, rendered
from the same frame's ground truth. Right: held-out error of the official baseline, the direct
regressor and the factored model. All content from real data and real runs."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif import viz, metrics as M
from fif.paths import PROJECT_ROOT, EXPERIMENTS_ROOT
import paperstyle as S
import matplotlib.pyplot as plt

SAMPLE = "XYZ109/vase_inspecting_1f82:000168"


def main():
    S.setup(size=7)
    rec = viz.load_aux(PROJECT_ROOT / "data/aux/aux_labels_10fps.jsonl", {SAMPLE})[SAMPLE]
    subj, rest = SAMPLE.split("/"); scene, fidx = rest.split(":")
    joints, fgt, verts, F, T0 = viz.cam_geometry(rec)
    preds = M.read_predictions_jsonl(EXPERIMENTS_ROOT / "FIF_hybrid_stereo_A/predictions_val.jsonl")[SAMPLE]
    pc = [None if fgt[k] is None else viz.pred_to_cam(preds.get(key), T0) for k, key in enumerate(M.FIELD_KEYS)]
    W, H = 13.4 / 2.54, 5.4 / 2.54
    fig = plt.figure(figsize=(W, H))

    # left: the held-out frame, ground truth in grey and the prediction in red
    ax = fig.add_axes([0.015, 0.09, 0.285, 0.74]); gray = viz.read_frame(subj, scene, int(fidx))
    viz.draw_overlay(ax, gray, rec, joints, fgt, pc, verts, F, show_mesh=True)
    fig.text(0.157, 0.875, "truth (grey) and prediction (red)", ha="center", fontsize=6.0, color="#222222")

    # middle: the factorisation itself
    ax3 = fig.add_axes([0.315, 0.13, 0.335, 0.72], projection="3d")
    S.scene_3d(ax3, joints, fgt, None, verts, F, elev=12, azim=10, triad=False, zoom=0.98, vec_lw=0.5, mesh_alpha=0.92)
    fig.text(0.482, 0.875, "field = nearest mesh point minus joint", ha="center", fontsize=6.0, color="#222222")
    fig.text(0.482, 0.015, "predict the joints and the object pose, apply the known mesh,\nfuse with a direct head by predicted precision",
             ha="center", va="bottom", fontsize=5.4, color="#444444", linespacing=1.35)

    # right: the headline numbers
    axb = fig.add_axes([0.735, 0.24, 0.245, 0.58])
    vals = [("InterField\n(official)", 60.47, S.GREY), ("direct\nregression", 46.15, S.DIRECT), ("factored\n(ours)", 44.43, S.FACTORED)]
    for i, (lab, v, c) in enumerate(vals):
        axb.bar(i, v, color=c, width=0.68, zorder=3); axb.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=6.2, color="#222222")
    axb.set_xticks(range(3)); axb.set_xticklabels([v[0] for v in vals], fontsize=5.8)
    axb.set_ylabel("mean ADE (mm)", fontsize=6.0); axb.set_ylim(0, 70); axb.set_yticks([0, 20, 40, 60])
    axb.tick_params(axis="x", length=0, labelsize=5.8); axb.tick_params(axis="y", labelsize=5.8)
    fig.text(0.857, 0.875, "error on held-out subjects", ha="center", fontsize=6.0, color="#222222")
    S.save(fig, "graphical_abstract", png_dpi=300)


if __name__ == "__main__":
    main()
