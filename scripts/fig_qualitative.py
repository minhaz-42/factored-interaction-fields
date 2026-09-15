"""Figure 4 / 5: qualitative comparison grid from real predictions (full text width).

Two samples per row, each shown as [ground truth | direct (B3) | factored (FIF)]; predicted vectors are red over
the ground-truth skeleton, ground-truth vectors grey. Per-panel ADE is printed. Regions outside the image are light
grey so that hands drawn there are visibly outside the camera's field of view. Sample selection: --samples explicit
ids, --auto k (k samples spread over the error quantiles of the last method, distinct objects), or --worst k."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif import viz, metrics as M
from fif.paths import PROJECT_ROOT
import paperstyle as S
import matplotlib.pyplot as plt

COLNAMES = {"B3": "direct (B3)", "FIF": "factored (FIF)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux", type=Path, default=PROJECT_ROOT / "data/aux/aux_labels_10fps.jsonl")
    ap.add_argument("--pred", nargs="+", required=True, help="name=path/to/predictions.jsonl (last one is 'ours')")
    ap.add_argument("--samples", nargs="*", default=None); ap.add_argument("--auto", type=int, default=0); ap.add_argument("--worst", type=int, default=0)
    ap.add_argument("--name", default="fig4_qualitative"); ap.add_argument("--labels", nargs="*", default=None, help="one caption per sample, printed in the GT panel")
    ap.add_argument("--per-row", type=int, default=2)
    a = ap.parse_args()
    S.setup()
    methods = [(s.split("=", 1)[0], M.read_predictions_jsonl(s.split("=", 1)[1])) for s in a.pred]
    common = set.intersection(*[set(p.keys()) for _, p in methods])
    aux = viz.load_aux(a.aux, common)
    refs = [aux[s] for s in sorted(common) if any(aux[s]["field_valid"])]
    last = methods[-1][1]

    def sample_ade(r, preds):
        e = []
        for k, key in enumerate(M.FIELD_KEYS):
            if r["field_valid"][k] and preds[r["sample_id"]].get(key) is not None:
                e.append(np.linalg.norm(preds[r["sample_id"]][key] - np.asarray(r["field_world"][k]), axis=1).mean())
        return float(np.mean(e)) if e else np.nan
    ades = np.array([sample_ade(r, last) for r in refs])
    if a.samples: sel = [aux[s] for s in a.samples]
    elif a.worst: sel = [refs[i] for i in np.argsort(-ades)[: a.worst]]
    else:
        k = a.auto or 4; qs = np.quantile(ades[~np.isnan(ades)], np.linspace(0.1, 0.9, k)); sel, used = [], set()
        for q in qs:  # nearest sample to the quantile whose object was not used yet and whose object is in view
            for i in np.argsort(np.abs(ades - q)):
                r = refs[i]
                if r["object_alias"] in used or r["obj_R_cam0"] is None or r["obj_fov_frac"].get("headset0", 0) < 0.9: continue
                sel.append(r); used.add(r["object_alias"]); break
    n = len(sel); per_row = a.per_row; nrow = int(np.ceil(n / per_row)); ncol = 1 + len(methods)
    inner, group, header, rowgap = 0.03, 0.14, 0.15, 0.06  # inches
    s = (S.FULL - (per_row * (ncol - 1)) * inner - (per_row - 1) * group) / (per_row * ncol)
    Hin = nrow * s + (nrow - 1) * rowgap + header
    fig = plt.figure(figsize=(S.FULL, Hin))
    for idx, r in enumerate(sel):
        ri, gi = divmod(idx, per_row)
        subj, rest = r["sample_id"].split("/"); scene, fidx = rest.split(":")
        gray = viz.read_frame(subj, scene, int(fidx)); joints, fgt, verts, F, T0 = viz.cam_geometry(r)
        box = None
        for c in range(ncol):
            x = gi * (ncol * s + (ncol - 1) * inner + group) + c * (s + inner); y = Hin - header - (ri + 1) * s - ri * rowgap
            ax = fig.add_axes([x / S.FULL, y / Hin, s / S.FULL, s / Hin]); ax.set_facecolor("#E9E9E9")
            if c == 0:
                box = viz.draw_overlay(ax, gray, r, joints, fgt, None, verts, F)
                cap = a.labels[idx] if a.labels and idx < len(a.labels) else f"{r['object_alias']}, {subj}"
                ax.text(0.03, 0.965, cap, transform=ax.transAxes, fontsize=5.6, color="white", va="top", bbox=dict(facecolor="black", alpha=0.55, lw=0, pad=1.6))
            else:
                name, preds = methods[c - 1]
                pc = [viz.pred_to_cam(preds[r["sample_id"]].get(key), T0) for key in M.FIELD_KEYS]; pc = [None if fgt[k] is None else pc[k] for k in range(2)]
                viz.draw_overlay(ax, gray, r, joints, None, pc, verts, F, box=box, show_gt=False, err_text=f"ADE {viz.per_hand_ade(pc, fgt):.1f} mm")
            ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            # keep the crop square: pad the shorter side symmetrically
            x0, y0, x1, y1 = box; wbox, hbox = x1 - x0, y1 - y0; side = max(wbox, hbox)
            ax.set_xlim((wbox - side) / 2, (wbox + side) / 2); ax.set_ylim((hbox + side) / 2, (hbox - side) / 2)
            if ri == 0:
                ttl = "ground truth" if c == 0 else COLNAMES.get(methods[c - 1][0], methods[c - 1][0])
                fig.text((x + s / 2) / S.FULL, (y + s + 0.03) / Hin, ttl, ha="center", va="bottom", fontsize=7, color="#222222")
    S.save(fig, a.name); print("samples:", [r["sample_id"] for r in sel])


if __name__ == "__main__":
    main()
