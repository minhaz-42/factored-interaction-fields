"""Figure: cross-dataset transfer between SHOW3D and HOT3D-IF, direct vs factored.
(a) the four cross-dataset settings; (b) the effect of adding HOT3D-train to the SHOW3D training set,
evaluated on both datasets. Reads tables/transfer.json (written by make_transfer_table.py). Column-width."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fif.paths import PROJECT_ROOT
import paperstyle as S
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, default=PROJECT_ROOT / "tables/transfer.json"); a = ap.parse_args()
    S.setup(); d = json.load(open(a.json))
    settings = [("SHOW3D\nto HOT3D\n(zero-shot)", ("B3_direct_stereo_rays", "SHOW3D->HOT3D (fold A)"), ("FIF_hybrid_stereo", "SHOW3D->HOT3D (fold A)")),
                ("HOT3D\nto HOT3D", ("T2_B3", "HOT3D->HOT3D"), ("T2_FIF", "HOT3D->HOT3D")),
                ("HOT3D\nto SHOW3D", ("T2_B3", "HOT3D->SHOW3D"), ("T2_FIF", "HOT3D->SHOW3D")),
                ("SHOW3D + HOT3D\nto HOT3D", ("T3_B3", "SHOW3D+HOT3D->HOT3D"), ("T3_FIF", "SHOW3D+HOT3D->HOT3D"))]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(S.COL, 3.7), gridspec_kw={"height_ratios": [1.15, 1]})
    w = 0.36
    for i, (lab, (dr, dk), (fr, fk)) in enumerate(settings):
        yd, yf = d[dr][dk], d[fr][fk]
        ax.bar(i - w / 2, yd, w, color=S.DIRECT, label="direct (B3)" if i == 0 else None, zorder=3)
        ax.bar(i + w / 2, yf, w, color=S.FACTORED, label="factored (FIF)" if i == 0 else None, zorder=3)
        ax.text(i - w / 2, yd + 1.2, f"{yd:.1f}", ha="center", fontsize=5.6, color="#222222"); ax.text(i + w / 2, yf + 1.2, f"{yf:.1f}", ha="center", fontsize=5.6, color="#222222")
        ax.text(i, max(yd, yf) + 8.5, f"{yf - yd:+.1f} mm", ha="center", fontsize=5.8, color=S.FACTORED)
    ax.set_xticks(range(len(settings))); ax.set_xticklabels([s[0] for s in settings], fontsize=6.2, linespacing=1.1); ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("mean ADE (mm)"); ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80])
    ax.axhline(82.6, ls=(0, (1, 1.5)), c="#777777", lw=0.6, zorder=2); ax.text(-0.55, 84, "zero-vector predictor on SHOW3D, 82.6", fontsize=5.4, ha="left", color="#555555")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.83), ncol=1)
    S.title(ax, "a", "training set to evaluation set")
    # (b) adding HOT3D-train to SHOW3D training
    xs = [0, 1]; xl = ["SHOW3D only", "SHOW3D + HOT3D-train"]
    series = [("direct, evaluated on SHOW3D", S.DIRECT, "-", [d["B3_direct_stereo_rays"]["SHOW3D->SHOW3D (fold A)"], d["T3_B3"]["SHOW3D+HOT3D->SHOW3D"]]),
              ("factored, evaluated on SHOW3D", S.FACTORED, "-", [d["FIF_hybrid_stereo"]["SHOW3D->SHOW3D (fold A)"], d["T3_FIF"]["SHOW3D+HOT3D->SHOW3D"]]),
              ("direct, evaluated on HOT3D", S.DIRECT, (0, (3, 1.5)), [d["B3_direct_stereo_rays"]["SHOW3D->HOT3D (fold A)"], d["T3_B3"]["SHOW3D+HOT3D->HOT3D"]]),
              ("factored, evaluated on HOT3D", S.FACTORED, (0, (3, 1.5)), [d["FIF_hybrid_stereo"]["SHOW3D->HOT3D (fold A)"], d["T3_FIF"]["SHOW3D+HOT3D->HOT3D"]])]
    labels = []  # (x, y, text) collected, then pushed apart vertically where they would collide
    for si, (lab, col, ls, ys) in enumerate(series):
        ax2.plot(xs, ys, marker="o", ms=3.2, lw=1.0, c=col, ls=ls, label=lab, zorder=3)
        for x, y in zip(xs, ys): labels.append([x, y, f"{y:.1f}"])
    for x in xs:
        L = sorted([l for l in labels if l[0] == x], key=lambda l: l[1]); i = 0
        while i < len(L):  # merge identical strings, then enforce a minimum vertical gap of 1.0 mm
            if i + 1 < len(L) and L[i][2] == L[i + 1][2]: L.pop(i + 1); continue
            i += 1
        for i in range(1, len(L)):
            if L[i][1] - L[i - 1][1] < 1.0: L[i][1] = L[i - 1][1] + 1.0
        for lx, ly, txt in L:
            ax2.text(lx + (-0.06 if lx == 0 else 0.06), ly, txt, fontsize=5.6, ha="right" if lx == 0 else "left", va="center", color="#222222")
    ax2.set_xticks(xs); ax2.set_xticklabels(xl); ax2.set_xlim(-0.45, 1.45); ax2.set_ylabel("mean ADE (mm)"); ax2.set_ylim(33, 48); ax2.set_yticks([35, 40, 45])
    ax2.tick_params(axis="x", length=0, pad=3)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=5.9, columnspacing=1.0)
    S.title(ax2, "b", "adding 2,889 HOT3D frames to the SHOW3D training set")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.955, bottom=0.12, hspace=0.5)
    S.save(fig, "fig9_transfer")


if __name__ == "__main__":
    main()
