"""Figure 7a: accuracy vs cost for all runs (from tables/all_runs.json written by make_tables.py).
x = trainable parameters (M) or inference time; y = mean ADE (mm); markers by model family; labels by run name.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import PROJECT_ROOT
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "pdf.fonttype": 42, "axes.linewidth": 0.6})
FAM = {"B1": ("InterField", "s", "#6e6e6e"), "B2": ("direct mono", "o", "#9ecae1"), "B3": ("direct stereo", "o", "#1f78b4"), "FIF_geo": ("geometric", "^", "#33a02c"), "FIF_hybrid": ("FIF hybrid", "D", "#b2182b")}


def fam(run):
    for k, v in sorted(FAM.items(), key=lambda x: -len(x[0])):
        if run.startswith(k): return v
    return ("other", "x", "#000000")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", default="A"); ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "figures/fig7a_progression"); a = ap.parse_args()
    rows = [r for r in json.load(open(PROJECT_ROOT / "tables/all_runs.json")) if r.get("fold") == a.fold and r.get("mean_ade_mm") is not None and r["params_M"] != "-"]
    fig, ax = plt.subplots(figsize=(3.3, 2.2)); seen = set()
    for r in rows:
        name, mk, col = fam(r["run"]); ax.scatter(float(r["params_M"]), r["mean_ade_mm"], marker=mk, c=col, s=22, lw=0, label=(name if name not in seen else None)); seen.add(name)
        ax.annotate(r["run"].replace(f"_{a.fold}", ""), (float(r["params_M"]), r["mean_ade_mm"]), fontsize=4.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("trainable parameters (M)"); ax.set_ylabel("mean ADE (mm), held-out subjects"); ax.legend(frameon=False, fontsize=6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); a.out.parent.mkdir(exist_ok=True); fig.savefig(str(a.out) + ".pdf"); fig.savefig(str(a.out) + ".png", dpi=220); print("wrote", a.out, len(rows), "runs")


if __name__ == "__main__":
    main()
