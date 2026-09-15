"""Paper-quality LaTeX tables for the IVC manuscript, read from run artefacts only.
Writes paper/tab/{main,strata,ablation,oracle,transfer,uncertainty}.tex as bare tabulars.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, yaml
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import EXPERIMENTS_ROOT, PROJECT_ROOT
OUT = PROJECT_ROOT / "paper/tab"; OUT.mkdir(exist_ok=True)
E = EXPERIMENTS_ROOT


def J(run, f="metrics_val.json"):
    p = E / run / f
    return json.load(open(p)) if p.exists() else None


def S(run): return J(run, "summary.json")
def C(run): return yaml.safe_load(open(E / run / "config.yaml"))
def f2(x): return "--" if x is None else f"{x:.2f}"
def f3(x): return "--" if x is None else f"{x:.3f}"
def f1(x): return "--" if x is None else f"{x:.1f}"


def off(run, key="official", f="metrics_val.json"):
    m = J(run, f); return None if m is None else m.get(key)


def sd(run, key="surface_distance_vertex", f="metrics_val.json"):
    m = J(run, f); return None if m is None or key not in m else m[key]["surface_distance_mm"]


ref = json.load(open(PROJECT_ROOT / "tables/b1_official_reference.json"))

# ------------------------------------------------------------------ main table
rows = [("Zero-vector predictor", "--", None, ref["reference_predictors"]["zero_mm"]), ("Training-mean predictor", "--", None, ref["reference_predictors"]["train_mean_mm"]),
        ("Hand-crop oracle box, InterField~\\cite{rim2026show3dapi}", "h0", None, ref["reference_predictors"]["hand_crop_oracle_box_mm"])]
L = ["\\begin{tabular}{@{}llrrrrrrrr@{}}", "\\toprule", " & & \\multicolumn{5}{c}{Fold A (XYZ109, LYA722)} & \\multicolumn{3}{c}{Fold B (MHA016, MMO925)} \\\\",
     "\\cmidrule(lr){3-7}\\cmidrule(lr){8-10}", "Method & Views & ADE & acc@10 & acc@50 & acc@100 & SD & ADE & acc@50 & SD \\\\", "\\midrule"]
for name, views, _, a in rows: L.append(f"{name} & {views} & {f1(a)} & -- & -- & -- & -- & -- & -- & -- \\\\")
L.append(f"InterField ResNet-50, official~\\cite{{rim2026show3dapi}} & h0 & {ref['mean_ade_mm']:.2f} & {ref['mean_acc@10mm']:.3f} & {ref['mean_acc@50mm']:.3f} & {ref['mean_acc@100mm']:.3f} & -- & -- & -- & -- \\\\")
L.append("\\midrule")
def mrow(name, views, A, B=None, A2=None, bold=False):
    oa = off(A); sa = sd(A); ade = f2(oa["mean_ade_mm"])
    if A2: ade += f" / {off(A2)['mean_ade_mm']:.2f}"
    cells = [ade, f3(oa["mean_acc@10mm"]), f3(oa["mean_acc@50mm"]), f3(oa["mean_acc@100mm"]), f2(sa)]
    if B: ob = off(B); cells += [f2(ob["mean_ade_mm"]), f3(ob["mean_acc@50mm"]), f2(sd(B))]
    else: cells += ["--", "--", "--"]
    if bold: name = "\\textbf{" + name + "}"; cells = ["\\textbf{" + c + "}" for c in cells]
    return f"{name} & {views} & " + " & ".join(cells) + " \\\\"
L.append(mrow("Direct regression (B2)", "h0", "B2_direct_mono_A"))
L.append(mrow("Direct, stereo, no ray encoding (B3b)", "h0+h1", "B3b_direct_stereo_norays_A"))
L.append(mrow("Direct, stereo + Pl\\\"ucker rays (B3)", "h0+h1", "B3_direct_stereo_rays_A", "B3_direct_stereo_rays_B", "B3_direct_stereo_rays_s1_A"))
L.append("\\midrule")
L.append(mrow("FIF, geometric head only", "h0+h1", "FIF_geo_stereo_A"))
L.append(mrow("FIF, single view", "h0", "FIF_hybrid_mono_A"))
L.append(mrow("FIF (full)", "h0+h1", "FIF_hybrid_stereo_A", "FIF_hybrid_stereo_B", "FIF_hybrid_stereo_s1_A", bold=True))
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "main.tex").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ strata table (fold A, four models)
runs = [("B2_direct_mono_A", "Direct, h0"), ("FIF_hybrid_mono_A", "FIF, h0"), ("B3_direct_stereo_rays_A", "Direct, h0+h1"), ("FIF_hybrid_stereo_A", "FIF, h0+h1")]
st = {r: J(r)["stratified"] for r, _ in runs}
groups = [("Field magnitude", "magnitude", [("near(<15)", "$<15$ mm"), ("mid(15-60)", "15 to 60 mm"), ("far(>60)", "$>60$ mm")]),
          ("Joint visibility", "visibility0", [("visible", "visible"), ("occluded_or_outside", "occluded or outside")]),
          ("Hand in view", "hand_fov0", [("in_view", "$\\geq 90\\%$"), ("partial", "10 to 90\\%"), ("out_of_view", "$<10\\%$")]),
          ("Object in view", "object_fov0", [("in_view", "$\\geq 90\\%$"), ("partial", "10 to 90\\%"), ("out_of_view", "$<10\\%$")]),
          ("Joint group", "joint_group", [("fingertips", "fingertips"), ("distal", "distal"), ("intermediate", "intermediate"), ("proximal", "proximal"), ("thumb", "thumb"), ("palm", "palm"), ("wrist", "wrist")])]
L = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule", "Stratum & & $n$ (joints) & " + " & ".join(n for _, n in runs) + " \\\\", "\\midrule"]
for gname, g, keys in groups:
    for i, (k, lab) in enumerate(keys):
        n = st[runs[0][0]][g][k].get("n", 0)
        L.append(f"{gname if i == 0 else ''} & {lab} & {n:,} & " + " & ".join(f2(st[r][g][k]["ade_mm"]) for r, _ in runs) + " \\\\")
    L.append("\\midrule")
L.append("All joints & mean ADE & " + f"{sum(st[runs[0][0]]['magnitude'][k]['n'] for k in ('near(<15)','mid(15-60)','far(>60)')):,}" + " & " + " & ".join(f2(off(r)["mean_ade_mm"]) for r, _ in runs) + " \\\\")
L.append("Error component & along the viewing ray & & " + " & ".join(f2(st[r]["along_ray_vs_lateral"]["mean_err_par_mm"]) for r, _ in runs) + " \\\\")
L.append(" & lateral & & " + " & ".join(f2(st[r]["along_ray_vs_lateral"]["mean_err_lat_mm"]) for r, _ in runs) + " \\\\")
L.append("Direction error ($\\|\\field\\|>20$ mm) & degrees & & " + " & ".join(f1(st[r]["direction_err_deg_mag>20"]) for r, _ in runs) + " \\\\")
L.append("Magnitude error & mm & & " + " & ".join(f2(st[r]["magnitude_err_mm"]) for r, _ in runs) + " \\\\")
L.append("Surface distance & mm & & " + " & ".join(f2(sd(r)) for r, _ in runs) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "strata.tex").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ ablation table (fold A)
abl = [("B2_direct_mono_A", "Direct regression, single view"), ("B3b_direct_stereo_norays_A", "Direct, stereo, no ray encoding"), ("B3_direct_stereo_rays_A", "Direct, stereo + rays (B3)"),
       ("B3_direct_stereo_rays_s1_A", "\\quad same, seed 1"), ("FIF_geo_stereo_A", "FIF, geometric head only"), ("FIF_hybrid_meanfusion_A", "FIF, mean fusion"),
       ("FIF_hybrid_nonll_A", "FIF, no likelihood terms (mean fusion)"), ("FIF_hybrid_noaux_A", "FIF, no auxiliary losses"), ("FIF_hybrid_mono_A", "FIF, single view"),
       ("FIF_hybrid_gatefusion_A", "FIF, learned gate fusion"), ("FIF_hybrid_stereo_A", "FIF, precision fusion (full)"), ("FIF_hybrid_stereo_s1_A", "\\quad same, seed 1")]
L = ["\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lrrrrrrrrrr@{}}", "\\toprule", "Variant & ADE & acc@10 & acc@50 & acc@100 & SD & along ray & lateral & Params (M) & Train (h) & Infer.\\ (ms) \\\\", "\\midrule"]
for r, name in abl:
    o = off(r); s = S(r); c = C(r); a = J(r)["stratified"]["along_ray_vs_lateral"]
    L.append(f"{name} & {f2(o['mean_ade_mm'])} & {f3(o['mean_acc@10mm'])} & {f3(o['mean_acc@50mm'])} & {f3(o['mean_acc@100mm'])} & {f2(sd(r))} & {f2(a['mean_err_par_mm'])} & {f2(a['mean_err_lat_mm'])} & {c['params_trainable']/1e6:.1f} & {s['train_time_s']/3600:.1f} & {s['inference_s_per_frame_incl_metrics']*1000:.1f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "ablation.tex").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ oracle / head decomposition
def wgeo_stats(run):
    p = E / run / "uncertainty_val.npz"
    if not p.exists(): return None, None
    z = np.load(p, allow_pickle=True); m = np.repeat(z["field_mask"].astype(bool)[:, :, None], 21, 2)
    sg = z["s_geo"][m]; ld, lg = np.exp(-2 * z["s_dir"][m]), np.exp(-2 * sg)
    return float(np.median(lg / (ld + lg))), float(np.mean(sg >= 4.99))
dec = [("FIF, fold A, seed 0", "FIF_hybrid_stereo_A", "B3_direct_stereo_rays_A"), ("FIF, fold A, seed 1", "FIF_hybrid_stereo_s1_A", "B3_direct_stereo_rays_s1_A"),
       ("FIF single view, fold A", "FIF_hybrid_mono_A", "B2_direct_mono_A"), ("FIF, fold B", "FIF_hybrid_stereo_B", "B3_direct_stereo_rays_B"),
       ("FIF gate fusion, fold A", "FIF_hybrid_gatefusion_A", "B3_direct_stereo_rays_A"), ("FIF, SHOW3D+HOT3D, fold A", "T3_FIF_joint", "T3_B3_joint")]
L = ["\\footnotesize\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule", "Model & Direct alone & Direct head & Geo.\\ head & Geo., GT object & Geo., GT joints & Fused & med.\\ $w^{\\rm geo}$ & $s^{\\rm geo}$ saturated \\\\", "\\midrule"]
for name, fif, base in dec:
    m = J(fif); w, sat = wgeo_stats(fif)
    L.append(f"{name} & {f2(off(base)['mean_ade_mm'])} & {f2(m['official_dir_head']['mean_ade_mm'])} & {f2(m['official_geo_head']['mean_ade_mm'])} & {f2(m['oracle_hand_pred_obj_gt']['mean_ade_mm'])} & {f2(m['oracle_hand_gt_obj_pred']['mean_ade_mm'])} & \\textbf{{{f2(m['official']['mean_ade_mm'])}}} & {f2(w)} & {'--' if sat is None else f'{100*sat:.0f}\\%'} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "oracle.tex").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ transfer table
def hot(run, key="official"):
    m = J(run, "metrics_hot3d.json"); return None if m is None else m.get(key, m)
def show(run):
    m = J(run, "metrics_show3d.json"); return None if m is None else m.get("official", m)
L = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule", " & & \\multicolumn{2}{c}{SHOW3D fold A} & \\multicolumn{3}{c}{HOT3D-IF eval} \\\\", "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
     "Training data & Model & ADE & acc@50 & ADE & acc@10 & acc@50 \\\\", "\\midrule"]
def trow(td, name, run, sh, ho):
    return f"{td} & {name} & {f2(sh['mean_ade_mm']) if sh else '--'} & {f3(sh['mean_acc@50mm']) if sh else '--'} & {f2(ho['mean_ade_mm']) if ho else '--'} & {f3(ho['mean_acc@10mm']) if ho else '--'} & {f3(ho['mean_acc@50mm']) if ho else '--'} \\\\"
L.append(trow("SHOW3D (37,645 frames)", "Direct, single view (B2)", "B2_direct_mono_A", off("B2_direct_mono_A"), hot("B2_direct_mono_A")))
L.append(trow("", "Direct, stereo (B3)", "B3_direct_stereo_rays_A", off("B3_direct_stereo_rays_A"), hot("B3_direct_stereo_rays_A")))
L.append(trow("", "FIF, geometric head only", "FIF_geo_stereo_A", off("FIF_geo_stereo_A"), hot("FIF_geo_stereo_A")))
L.append(trow("", "FIF (full)", "FIF_hybrid_stereo_A", off("FIF_hybrid_stereo_A"), hot("FIF_hybrid_stereo_A")))
L.append("\\midrule")
L.append(trow("HOT3D-train (2,889 frames)", "Direct, stereo", "T2_B3_hot3d", show("T2_B3_hot3d"), off("T2_B3_hot3d")))
L.append(trow("", "FIF (full)", "T2_FIF_hot3d", show("T2_FIF_hot3d"), off("T2_FIF_hot3d")))
L.append("\\midrule")
L.append(trow("SHOW3D + HOT3D-train (40,534)", "Direct, stereo", "T3_B3_joint", off("T3_B3_joint"), hot("T3_B3_joint")))
L.append(trow("", "FIF (full)", "T3_FIF_joint", off("T3_FIF_joint"), hot("T3_FIF_joint")))
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "transfer.tex").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ uncertainty table
unc = [("FIF (full), fold A, seed 0", "FIF_hybrid_stereo_A"), ("FIF (full), fold A, seed 1", "FIF_hybrid_stereo_s1_A"), ("FIF, single view, fold A", "FIF_hybrid_mono_A"),
       ("FIF, learned gate, fold A", "FIF_hybrid_gatefusion_A"), ("FIF (full), fold B", "FIF_hybrid_stereo_B")]
L = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule", "Model & $\\rho_{\\rm dir}$ & $\\rho_{\\rm geo}$ & $\\rho_{\\rm fused}$ & ADE, all joints & drop 10\\% & drop 50\\% & AUSE \\\\", "\\midrule"]
for name, run in unc:
    z = np.load(E / run / "uncertainty_val.npz", allow_pickle=True); m = np.repeat(z["field_mask"].astype(bool)[:, :, None], 21, 2)
    ef, ed, eg = z["err_fused"][m], z["err_dir"][m], z["err_geo"][m]; sd_, sg = np.exp(z["s_dir"][m]), np.exp(z["s_geo"][m])
    rd = spearmanr(sd_, ed).correlation; rg = spearmanr(sg, eg).correlation if len(np.unique(np.round(sg, 3))) > 3 else None
    sf = np.exp(-0.5 * np.log(np.exp(-2 * z["s_dir"][m]) + np.exp(-2 * z["s_geo"][m]))); rf = spearmanr(sf, ef).correlation
    op, oo = np.argsort(-sf), np.argsort(-ef); fr = np.linspace(0, 0.5, 51)
    pred = np.array([ef[op[int(f * len(ef)):]].mean() for f in fr]); orc = np.array([ef[oo[int(f * len(ef)):]].mean() for f in fr])
    ause = np.trapz(pred - orc, fr) / 0.5
    L.append(f"{name} & {rd:.2f} & {'const.' if rg is None else f'{rg:.2f}'} & {rf:.2f} & {ef.mean():.2f} & {pred[10]:.2f} ({orc[10]:.2f}) & {pred[50]:.2f} ({orc[50]:.2f}) & {ause:.2f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "uncertainty.tex").write_text("\n".join(L) + "\n")
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
