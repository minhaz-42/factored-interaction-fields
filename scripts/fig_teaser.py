"""Figure 1 (teaser): the task on one real SHOW3D stereo frame.

(a, b) both headset views with the ground-truth hand skeletons and the posed object mesh;
(c) the reference view cropped around the interaction with the ground-truth field vectors;
(d) the same quantities in the camera frame: joints, posed mesh and nearest-vertex vectors.
All content is ground truth from the released annotations. Full text width."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif import viz, geometry as G
from fif.paths import PROJECT_ROOT
import paperstyle as S
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def wire(ax, uv, inside, F, offset=(0, 0), n=1400, lw=0.3, alpha=0.5, seed=0):
    e = F[np.random.default_rng(seed).choice(len(F), size=min(n, len(F)), replace=False)]
    segs = [[uv[i] - offset, uv[j] - offset] for tri in e for i, j in ((tri[0], tri[1]), (tri[1], tri[2])) if inside[i] and inside[j]]
    ax.add_collection(LineCollection(segs, colors=S.OBJ, linewidths=lw, alpha=alpha))


def skeleton(ax, uv, col, offset=(0, 0), lw=1.0, s=5):
    uv = uv - offset
    ax.add_collection(LineCollection([[uv[i], uv[j]] for i, j in S.BONES], colors=col, linewidths=lw))
    ax.scatter(uv[:, 0], uv[:, 1], s=s, c=col, zorder=3, linewidths=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aux", type=Path, default=PROJECT_ROOT / "data/aux/aux_labels_10fps.jsonl")
    ap.add_argument("--sample", default="LYA722/dinotoy_inspecting_4cb1:000744"); ap.add_argument("--name", default="fig1_teaser"); a = ap.parse_args()
    S.setup()
    rec = viz.load_aux(a.aux, {a.sample})[a.sample]
    subj, rest = a.sample.split("/"); scene, fidx = rest.split(":"); fidx = int(fidx)
    joints, fgt, verts, F, T0 = viz.cam_geometry(rec)
    fig = plt.figure(figsize=(S.FULL, 2.3))
    H = 0.86; y0 = 0.02
    # (a, b) full frames of both views, portrait 1024 x 1280
    fw = H * 2.3 * (1024 / 1280) / S.FULL
    for i, v in enumerate(("headset0", "headset1")):
        ax = fig.add_axes([0.003 + i * (fw + 0.008), y0, fw, H]); gray = viz.read_frame(subj, scene, fidx, v)
        Tv = np.asarray(rec["T_world_from_cam"][v]); fx, fy, cx, cy, w, h = rec["intrinsics"][v]
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="bilinear"); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off()
        if verts is not None:
            vv = G.world_to_cam_points(G.cam_to_world_points(verts, T0), Tv); uv_o, in_o, _ = G.project_pinhole(vv, fx, fy, cx, cy, w, h); wire(ax, uv_o, in_o, F, lw=0.25, alpha=0.55)
        for k, col in enumerate((S.LEFT, S.RIGHT)):
            if joints[k] is None: continue
            jv = G.world_to_cam_points(G.cam_to_world_points(joints[k], T0), Tv); uv, _, _ = G.project_pinhole(jv, fx, fy, cx, cy, w, h); skeleton(ax, uv, col, lw=0.9, s=3.5)
        S.label_panel(fig, ax, "ab"[i]); fig.text(ax.get_position().x0 + 0.018, ax.get_position().y1 + 0.006, f"headset camera {i}", fontsize=6.5, va="bottom", color="#333333")
    # (c) crop of the reference view with the ground-truth field
    x_c = 0.003 + 2 * (fw + 0.008) + 0.012; cw = H * 2.3 / S.FULL
    ax = fig.add_axes([x_c, y0, cw, H]); gray = viz.read_frame(subj, scene, fidx)
    viz.draw_overlay(ax, gray, rec, joints, fgt, None, verts, F, show_gt=True, show_mesh=True)
    S.label_panel(fig, ax, "c"); fig.text(ax.get_position().x0 + 0.018, ax.get_position().y1 + 0.006, "target: joint to nearest object point", fontsize=6.5, va="bottom", color="#333333")
    # (d) camera-frame geometry
    x_d = x_c + cw + 0.005; dw = 1.0 - x_d - 0.002
    ax3 = fig.add_axes([x_d - 0.01, -0.08, dw + 0.02, H + 0.16], projection="3d")
    S.scene_3d(ax3, joints, fgt, None, verts, F, elev=16, azim=-58, zoom=1.5, triad=False)
    S.triad_overlay(fig, ax3, anchor=(0.84, 0.13), length_in=0.17)
    fig.text(x_d + 0.005, y0 + H + 0.006, r"$\mathbf{(d)}$", fontsize=7.5, va="bottom")
    fig.text(x_d + 0.035, y0 + H + 0.006, "the same geometry in 3-D", fontsize=6.5, va="bottom", color="#333333")
    S.save(fig, a.name); print("frame", a.sample, "object", rec["object_alias"])


if __name__ == "__main__":
    main()
