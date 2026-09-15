"""Figure: uncertainty calibration and fusion behaviour from a run's uncertainty_val.npz.

(a) reliability: binned predicted scale exp(s) vs observed RMS error for the direct and geometric heads;
(b) sparsification: ADE of the fused output when discarding the k% most uncertain joints (predicted vs oracle);
(c) fusion weight histogram and the error of dir/geo/fused as a function of that weight. Column-width."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperstyle as S
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--name", default=None); a = ap.parse_args()
    S.setup()
    z = np.load(a.run / "uncertainty_val.npz", allow_pickle=True)
    m = np.repeat(z["field_mask"].astype(bool)[:, :, None], 21, 2); ef = z["err_fused"][m]; has_geo = "s_geo" in z.files
    fig = plt.figure(figsize=(S.COL, 3.45))
    ax0 = fig.add_axes([0.115, 0.605, 0.375, 0.33]); ax1 = fig.add_axes([0.63, 0.605, 0.36, 0.33]); ax2 = fig.add_axes([0.115, 0.10, 0.76, 0.33])
    # (a) reliability
    lim = [8, 400]
    for key, ek, col, lab in (("s_dir", "err_dir", S.DIRECT, "direct head"), ("s_geo", "err_geo", S.GEO, "geometric head")):
        if key not in z.files: continue
        sig = np.exp(z[key][m]); err = z[ek][m]
        if sig.std() < 1e-6:  # constant (saturated) scale: a single point
            ax0.plot([sig.mean()], [np.sqrt((err ** 2).mean())], "s", ms=3.5, c=col, label=lab + " (constant)", zorder=4); continue
        qs = np.quantile(sig, np.linspace(0, 1, 13)); c, r = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            sel = (sig >= lo) & (sig < hi)
            if sel.sum() > 10: c.append(sig[sel].mean()); r.append(np.sqrt((err[sel] ** 2).mean()))
        ax0.plot(c, r, "o-", ms=2.4, lw=0.9, c=col, label=lab, zorder=3)
    ax0.plot(lim, [np.sqrt(3) * x for x in lim], ls=(0, (1, 1.5)), c="#666666", lw=0.7, label=r"$\sqrt{3}\,\sigma$ (ideal)", zorder=2)
    ax0.set_xscale("log"); ax0.set_yscale("log"); ax0.set_xlim(*lim); ax0.set_ylim(*lim)
    ax0.set_xticks([10, 100]); ax0.set_xticklabels(["10", "100"]); ax0.set_yticks([10, 100]); ax0.set_yticklabels(["10", "100"])
    ax0.set_xlabel(r"predicted scale $\sigma$ (mm)"); ax0.set_ylabel("RMS error (mm)"); ax0.legend(loc="lower right", fontsize=5.6)
    S.title(ax0, "a", "reliability")
    # (b) sparsification
    s_pred = -0.5 * np.log(np.exp(-2 * z["s_dir"][m]) + (np.exp(-2 * z["s_geo"][m]) if has_geo else 0.0))
    fr = np.linspace(0, 0.5, 26)
    for order, col, ls, lab in ((np.argsort(-s_pred), S.FACTORED, "-", "predicted uncertainty"), (np.argsort(-ef), "#666666", (0, (3, 2)), "oracle (true error)")):
        ys = [ef[order[int(f * len(ef)):]].mean() for f in fr]
        ax1.plot(fr * 100, ys, lw=1.0, ls=ls, c=col, label=lab, zorder=3)
    ax1.set_xlabel("joints removed (%)"); ax1.set_ylabel("ADE of the rest (mm)"); ax1.legend(loc="upper right", fontsize=5.6); ax1.set_xlim(0, 50); ax1.set_ylim(0, 48)
    S.title(ax1, "b", "sparsification")
    # (c) fusion weight vs error
    if has_geo:
        lam_d, lam_g = np.exp(-2 * z["s_dir"][m]), np.exp(-2 * z["s_geo"][m]); wgeo = lam_g / (lam_d + lam_g)
        ax2.hist(wgeo, bins=np.linspace(0, 0.55, 34), color="#CCCCCC", zorder=2); ax2.set_xlabel("fusion weight on the geometric head"); ax2.set_ylabel("joints")
        ax2.set_xlim(0, 0.55)
        axr = ax2.twinx(); axr.spines["right"].set_visible(True); bins = np.linspace(0, 0.55, 12); cen = 0.5 * (bins[:-1] + bins[1:])
        for ek, col, lab in (("err_dir", S.DIRECT, "direct head"), ("err_geo", S.GEO, "geometric head"), ("err_fused", S.FACTORED, "fused")):
            e = z[ek][m]; y = [e[(wgeo >= lo) & (wgeo < hi)].mean() if ((wgeo >= lo) & (wgeo < hi)).sum() > 10 else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
            axr.plot(cen, y, "o-", ms=2.4, lw=0.9, c=col, label=lab, zorder=3)
        axr.set_ylabel("ADE in bin (mm)"); axr.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=5.6); axr.set_ylim(0, 260)
        ax2.set_yticks([0, 25000, 50000, 75000]); ax2.set_yticklabels(["0", "25k", "50k", "75k"])
        S.title(ax2, "c", "fusion weight against the error of each estimate")
    else:
        ax2.set_axis_off()
    S.save(fig, a.name or f"fig7_uncertainty_{a.run.name}")


if __name__ == "__main__":
    main()
