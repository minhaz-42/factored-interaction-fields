"""Figure: geometry of the joint-anchored interaction field on a real SHOW3D frame (column width).

(a) reference view cropped around the interaction, with the projected 21-joint skeletons, the posed
    object mesh and the nearest-vertex vectors; (b) soft nearest-vertex assignment weights of one joint
    at the initial and final training temperature. All content from the released ground truth."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif import viz
from fif.paths import PROJECT_ROOT
import paperstyle as S
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aux", type=Path, default=PROJECT_ROOT / "data/aux/aux_labels_10fps.jsonl")
    ap.add_argument("--sample", default="ASC023/aria_inspecting_f5e9:000864"); ap.add_argument("--name", default="fig3_field_geometry"); a = ap.parse_args()
    S.setup()
    rec = viz.load_aux(a.aux, {a.sample})[a.sample]
    subj, rest = a.sample.split("/"); scene, fidx = rest.split(":"); fidx = int(fidx)
    joints, fields, verts, F, T0 = viz.cam_geometry(rec); gray = viz.read_frame(subj, scene, fidx)
    FH = 3.35; fig = plt.figure(figsize=(S.COL, FH))
    # (a) crop with skeletons, mesh and vectors; the axes is fitted to the crop so labels align with the image
    ax = fig.add_axes([0.0, 0.335, 1.0, 0.63])
    x0, y0, x1, y1 = viz.draw_overlay(ax, gray, rec, joints, fields, None, verts, F, show_gt=True, show_mesh=True)
    hin = 0.63 * FH; win = hin * (x1 - x0) / (y1 - y0); ax.set_position([(S.COL - win) / 2 / S.COL, 0.335, win / S.COL, 0.63])
    S.label_panel(fig, ax, "a", dx=0.0); fig.text(ax.get_position().x0 + 0.045, ax.get_position().y1 + 0.004, "reference view: skeletons, posed mesh and the 42 field vectors", fontsize=6.5, va="bottom", color="#333333")
    # colour key for the overlay
    for i, (col, lab) in enumerate(((S.LEFT, "left hand"), (S.RIGHT, "right hand"), (S.OBJ, "object mesh"), (S.GTVEC, "field vector"))):
        xk = ax.get_position().x0 + 0.01 + i * 0.205
        fig.patches.append(plt.Rectangle((xk, 0.306), 0.022, 0.012, transform=fig.transFigure, color=col, lw=0))
        fig.text(xk + 0.03, 0.312, lab, fontsize=6, va="center", color="#333333")
    # (b) soft assignment for one joint (index fingertip, joint 1) at two temperatures
    ax2 = fig.add_axes([0.13, 0.085, 0.85, 0.165])
    J = joints[1] if joints[1] is not None else joints[0]; p = J[1]
    d2 = ((verts - p) ** 2).sum(1); order = np.argsort(d2)[:60]
    for tau, col, lab in ((400.0, "#AAAAAA", r"$\tau = 400\ \mathrm{mm^2}$ (start of training)"), (25.0, "#222222", r"$\tau = 25\ \mathrm{mm^2}$ (end of training)")):
        wgt = np.exp(-(d2 - d2.min()) / tau); wgt /= wgt.sum()
        ax2.plot(np.sqrt(d2[order]), wgt[order], ".-", ms=2.5, lw=0.8, c=col, label=lab)
    ax2.set_xlabel("distance from the index fingertip to the vertex (mm)"); ax2.set_ylabel("weight"); ax2.set_yscale("log"); ax2.legend(loc="upper right", fontsize=5.8)
    ax2.set_ylim(1e-4, 2)
    S.title(ax2, "b", "soft nearest-vertex weights of the 60 nearest vertices")
    S.save(fig, a.name); print("frame", a.sample, "object", rec["object_alias"], "vectors:", [None if f is None else np.round(np.linalg.norm(f, axis=1).mean(), 1) for f in fields])


if __name__ == "__main__":
    main()
