"""Figure: what the factored model actually does.
(a) error decomposition of the geometric route: direct head, geometric head, geometric head with the
    ground-truth object pose, with the ground-truth joints, and the fused output; dashed = the standalone
    direct baseline trained without the geometric head.
(b) fusion weight on the geometric head: fold A (scale head saturated) vs fold B (calibrated).
(c) reliability of both heads on fold B, where neither scale saturated.
All values are read from run artefacts. Column-width figure."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif.paths import EXPERIMENTS_ROOT
import paperstyle as S
import matplotlib.pyplot as plt


def M(run): return json.load(open(EXPERIMENTS_ROOT / run / "metrics_val.json"))


def main():
    S.setup()
    groups = [("stereo, fold A", "FIF_hybrid_stereo_A", "B3_direct_stereo_rays_A"), ("single view, fold A", "FIF_hybrid_mono_A", "B2_direct_mono_A"), ("stereo, fold B", "FIF_hybrid_stereo_B", "B3_direct_stereo_rays_B")]
    comps = [("official_dir_head", "direct head", S.DIRECT), ("official_geo_head", "geometric head", S.GEO), ("oracle_hand_pred_obj_gt", "geometric head, true object pose", S.GEO_LIGHT),
             ("oracle_hand_gt_obj_pred", "geometric head, true joints", S.GEO_PALE), ("official", "fused output", S.FACTORED)]
    fig = plt.figure(figsize=(S.COL, 3.5))
    ax = fig.add_axes([0.105, 0.555, 0.885, 0.385])
    w = 0.8 / len(comps)
    for gi, (glab, fif, base) in enumerate(groups):
        m = M(fif)
        for ci, (key, clab, col) in enumerate(comps):
            x = gi + (ci - (len(comps) - 1) / 2) * w; y = m[key]["mean_ade_mm"]
            ax.bar(x, y, width=w * 0.92, color=col, label=clab if gi == 0 else None, zorder=3)
        b = M(base)["official"]["mean_ade_mm"]
        ax.plot([gi - 0.44, gi + 0.44], [b, b], ls=(0, (3, 2)), c="#222222", lw=0.7, label="standalone direct regressor" if gi == 0 else None, zorder=4)
        ax.text(gi + 0.45, b, f"{b:.1f}", fontsize=5.2, va="center", ha="left", color="#222222")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.tick_params(axis="x", length=0, pad=3); ax.set_xlim(-0.55, len(groups) - 0.35)
    ax.set_ylabel("mean ADE (mm)"); ax.set_ylim(0, 80); ax.set_yticks([0, 20, 40, 60])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.7, columnspacing=0.8, handlelength=1.0)
    S.title(ax, "a", "heads, oracle swaps and fused output")
    # (b) fusion weight distributions
    ax2 = fig.add_axes([0.105, 0.10, 0.40, 0.33])
    for run, col, lab in (("FIF_hybrid_stereo_A", S.FACTORED, "fold A"), ("FIF_hybrid_stereo_B", S.ORANGE, "fold B")):
        z = np.load(EXPERIMENTS_ROOT / run / "uncertainty_val.npz", allow_pickle=True)
        m = np.repeat(z["field_mask"].astype(bool)[:, :, None], 21, 2)
        ld, lg = np.exp(-2 * z["s_dir"][m]), np.exp(-2 * z["s_geo"][m]); wg = lg / (ld + lg)
        ax2.hist(wg, bins=np.linspace(0, 0.6, 37), color=col, alpha=0.8, density=True, label=f"{lab}, median {np.median(wg):.2f}", zorder=3)
    ax2.set_xlabel("weight on the geometric head"); ax2.set_ylabel("density"); ax2.legend(loc="upper right", fontsize=5.7); ax2.set_xlim(0, 0.6); ax2.set_xticks([0, 0.2, 0.4, 0.6])
    S.title(ax2, "b", "fusion weights")
    # (c) reliability on fold B
    ax3 = fig.add_axes([0.62, 0.10, 0.37, 0.33])
    z = np.load(EXPERIMENTS_ROOT / "FIF_hybrid_stereo_B" / "uncertainty_val.npz", allow_pickle=True)
    m = np.repeat(z["field_mask"].astype(bool)[:, :, None], 21, 2)
    for key, ek, col, lab in (("s_dir", "err_dir", S.DIRECT, "direct head"), ("s_geo", "err_geo", S.GEO, "geometric head")):
        sig, err = np.exp(z[key][m]), z[ek][m]; qs = np.quantile(sig, np.linspace(0, 1, 13)); c, r = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            s = (sig >= lo) & (sig < hi)
            if s.sum() > 20: c.append(sig[s].mean()); r.append(np.sqrt((err[s] ** 2).mean()))
        ax3.plot(c, r, "o-", ms=2.4, lw=0.9, c=col, label=lab, zorder=3)
    lim = [6, 200]; ax3.plot(lim, [np.sqrt(3) * x for x in lim], ls=(0, (1, 1.5)), c="#666666", lw=0.7, label=r"$\sqrt{3}\,\sigma$ (ideal)", zorder=2)
    ax3.set_xscale("log"); ax3.set_yscale("log"); ax3.set_xlim(*lim); ax3.set_ylim(*lim)
    ax3.set_xlabel(r"predicted scale $\sigma$ (mm)"); ax3.set_ylabel("RMS error (mm)"); ax3.legend(loc="lower right", fontsize=5.7)
    ax3.set_xticks([10, 100]); ax3.set_yticks([10, 100]); ax3.set_xticklabels(["10", "100"]); ax3.set_yticklabels(["10", "100"])
    S.title(ax3, "c", "reliability, fold B")
    S.save(fig, "fig7_decomposition")


if __name__ == "__main__":
    main()
