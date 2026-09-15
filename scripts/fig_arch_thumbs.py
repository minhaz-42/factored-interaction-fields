"""Image assets for the architecture figure (paper/fig/).

thumb_in0/1   the two raw headset frames of a held-out sample
thumb_patches a 3x3 grid of real image patches, standing for the ViT patch tokens
thumb_meshgrey the canonical object mesh as a grey shaded wireframe
thumb_out     the factored model's prediction on the reference view
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif import viz, metrics as M, geometry as G
from fif.paths import PROJECT_ROOT, EXPERIMENTS_ROOT
import paperstyle as S
import matplotlib.pyplot as plt

SAMPLE = "XYZ109/vase_inspecting_1f82:000168"
OUT = PROJECT_ROOT / "paper/fig"


def main():
    S.setup()
    rec = viz.load_aux(PROJECT_ROOT / "data/aux/aux_labels_10fps.jsonl", {SAMPLE})[SAMPLE]
    subj, rest = SAMPLE.split("/"); scene, fidx = rest.split(":"); fidx = int(fidx)
    joints, fgt, verts, F, T0 = viz.cam_geometry(rec)

    # --- the two input frames -------------------------------------------------------------
    for v in (0, 1):
        gray = viz.read_frame(subj, scene, fidx, f"headset{v}")
        cv2.imwrite(str(OUT / f"thumb_in{v}.png"), cv2.resize(gray, (256, 320), interpolation=cv2.INTER_AREA))

    # --- 3 x 3 grid of real patches, with white gutters -----------------------------------
    gray = viz.read_frame(subj, scene, fidx)
    h, w = gray.shape
    uv = [G.project_pinhole(j, *rec["intrinsics"]["headset0"])[0] for j in joints if j is not None]
    c = np.concatenate(uv).mean(0) if uv else np.array([w / 2, h / 2])
    s = int(min(h, w) * 0.72)
    x0 = int(np.clip(c[0] - s / 2, 0, w - s)); y0 = int(np.clip(c[1] - s / 2, 0, h - s))
    crop = cv2.resize(gray[y0:y0 + s, x0:x0 + s], (240, 240), interpolation=cv2.INTER_AREA)
    tile = np.full((252, 252), 255, np.uint8)
    for i in range(3):
        for j in range(3):
            tile[i * 84 + 2:i * 84 + 82, j * 84 + 2:j * 84 + 82] = crop[i * 80:i * 80 + 80, j * 80:j * 80 + 80]
    cv2.imwrite(str(OUT / "thumb_patches.png"), tile)

    # --- canonical mesh as a grey shaded wireframe ----------------------------------------
    V, Fc = G.load_object_mesh(rec["object_alias"])
    fig = plt.figure(figsize=(1.7, 1.7)); ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.plot_trisurf(V[:, 0], V[:, 2], -V[:, 1], triangles=Fc, color="#E9ECEF", edgecolor="#9AA2AA",
                    linewidth=0.06, shade=True, antialiased=True)
    c3 = V.mean(0); r = (V.max(0) - V.min(0)).max() / 2 * 0.88
    ax.set_xlim(c3[0] - r, c3[0] + r); ax.set_ylim(c3[2] - r, c3[2] + r); ax.set_zlim(-c3[1] - r, -c3[1] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=-52); ax.set_axis_off()
    fig.savefig(OUT / "thumb_meshgrey.png", dpi=300, transparent=True); plt.close(fig)

    # --- the model's prediction on the reference view --------------------------------------
    preds = M.read_predictions_jsonl(EXPERIMENTS_ROOT / "FIF_hybrid_stereo_A/predictions_val.jsonl")[SAMPLE]
    pc = [None if fgt[k] is None else viz.pred_to_cam(preds.get(key), T0) for k, key in enumerate(M.FIELD_KEYS)]
    fig = plt.figure(figsize=(1.4, 1.4)); ax = fig.add_axes([0, 0, 1, 1])
    viz.draw_overlay(ax, gray, rec, joints, None, pc, verts, F, show_gt=False, show_mesh=True)
    fig.savefig(OUT / "thumb_out.png", dpi=300); plt.close(fig)
    print("wrote assets to", OUT)


if __name__ == "__main__":
    main()
