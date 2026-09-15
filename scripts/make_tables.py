"""Collect experiments/<run>/{config.yaml|config.json, metrics_val.json|summary.json} into
Markdown and LaTeX tables under tables/. Every number is read from run artefacts;
nothing is typed by hand.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import EXPERIMENTS_ROOT, PROJECT_ROOT

COLS = [("mean_ade_mm", "ADE", "{:.2f}"), ("left_to_object.ade_mm", "L", "{:.2f}"), ("right_to_object.ade_mm", "R", "{:.2f}"),
        ("mean_acc@10mm", "acc@10", "{:.3f}"), ("mean_acc@50mm", "acc@50", "{:.3f}"), ("mean_acc@100mm", "acc@100", "{:.3f}"), ("mean_recall", "recall", "{:.3f}")]


def get(d, path):
    for k in path.split("."):
        if d is None: return None
        d = d.get(k)
    return d


def load_run(run: Path):
    cfg = None
    if (run / "config.yaml").exists(): cfg = yaml.safe_load(open(run / "config.yaml"))
    elif (run / "config.json").exists(): cfg = json.load(open(run / "config.json"))
    met = json.load(open(run / "metrics_val.json")) if (run / "metrics_val.json").exists() else None
    summ = json.load(open(run / "summary.json")) if (run / "summary.json").exists() else None
    return cfg, met, summ


def main():
    runs = sorted(p for p in EXPERIMENTS_ROOT.iterdir() if p.is_dir() and not p.name.startswith("smoke"))
    ref_p = PROJECT_ROOT / "tables" / "b1_official_reference.json"
    ref = json.load(open(ref_p)) if ref_p.exists() else None
    out_dir = PROJECT_ROOT / "tables"; out_dir.mkdir(exist_ok=True)
    md = ["| run | model | views | fold | " + " | ".join(c[1] for c in COLS) + " | SD (mm) | params (M) | train h |", "|" + "---|" * (len(COLS) + 7)]
    tex = ["\\begin{tabular}{ll" + "r" * (len(COLS) + 3) + "}", "\\toprule", "Run & Model & Views & " + " & ".join(c[1] for c in COLS) + " & SD & Params (M) \\\\", "\\midrule"]
    rows = []
    if ref is not None:  # organisers' published baseline on the identical fold-A split (not retrained)
        vals = [c[2].format(ref[c[0]]) for c in COLS]
        md.append(f"| {ref['run']} | {ref['model']} | {ref['views']} | A | " + " | ".join(vals) + f" | - | {ref['params_M']} | {ref['train_h']} |")
        tex.append(f"{ref['run'].replace('_', chr(92)+'_')} & {ref['model'].replace('_', chr(92)+'_')} & {ref['views']} & " + " & ".join(vals) + f" & - & {ref['params_M']} \\\\")
        rows.append({"run": ref["run"], "model": ref["model"], "views": ref["views"], "fold": "A", **{c[0]: ref[c[0]] for c in COLS},
                     "surface_distance_mm": "-", "params_M": ref["params_M"], "train_h": ref["train_h"], "stratified": None, "oracles": None, "source": "official RESULTS.md"})
    for run in runs:
        cfg, met, summ = load_run(run)
        if cfg is None: continue
        off = (met or {}).get("official") if met else (summ or {}).get("final")
        if off is None:
            continue
        vals = [c[2].format(get(off, c[0])) if get(off, c[0]) is not None else "-" for c in COLS]
        sd = get(met or {}, "surface_distance_vertex.surface_distance_mm"); sd = f"{sd:.2f}" if sd is not None else "-"
        params = cfg.get("params_trainable") or cfg.get("params_total"); params = f"{params/1e6:.1f}" if params else "-"
        th = (summ or {}).get("train_time_s"); th = f"{th/3600:.1f}" if th else "-"
        model = cfg.get("model") or f'{cfg.get("mode","?")}/{cfg.get("fusion","-")}/rays={cfg.get("use_rays","-")}'
        views = "+".join(v.replace("headset", "h") for v in cfg.get("views", []))
        md.append(f"| {run.name} | {model} | {views} | {cfg.get('fold','-')} | " + " | ".join(vals) + f" | {sd} | {params} | {th} |")
        tex.append(f"{run.name.replace('_', chr(92)+'_')} & {model.replace('_', chr(92)+'_')} & {views} & " + " & ".join(vals) + f" & {sd} & {params} \\\\")
        rows.append({"run": run.name, "model": model, "views": views, "fold": cfg.get("fold"), **{c[0]: get(off, c[0]) for c in COLS}, "surface_distance_mm": sd, "params_M": params, "train_h": th,
                     "stratified": (met or {}).get("stratified"), "oracles": {k: v.get("mean_ade_mm") for k, v in (met or {}).items() if k.startswith("oracle") or k.startswith("official_")}})
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "main_results.md").write_text("\n".join(md) + "\n"); (out_dir / "main_results.tex").write_text("\n".join(tex) + "\n")
    (out_dir / "all_runs.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md)); print(f"wrote {len(rows)} runs -> tables/")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# additional tables: ablation (fold A FIF variants), strata (one run), efficiency (all runs)
# ---------------------------------------------------------------------------
def _fmt(x, f="{:.2f}"):
    return f.format(x) if isinstance(x, (int, float)) else "-"


def extra_tables():
    out_dir = PROJECT_ROOT / "tables"
    rows = json.load(open(out_dir / "all_runs.json"))
    # ablation: FIF variants and direct baselines on fold A
    abl = [r for r in rows if r["fold"] == "A" and (r["run"].startswith("FIF") or r["run"].startswith("B3") or r["run"].startswith("B2"))]
    tex = ["\\begin{tabular}{lrrrr}", "\\toprule", "Variant & ADE & acc@10 & acc@50 & SD \\\\", "\\midrule"]
    for r in abl:
        tex.append(f"{r['run'].replace('_A','').replace('_', chr(92)+'_')} & {_fmt(r['mean_ade_mm'])} & {_fmt(r['mean_acc@10mm'], '{:.3f}')} & {_fmt(r['mean_acc@50mm'], '{:.3f}')} & {r['surface_distance_mm']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]; (out_dir / "ablation.tex").write_text("\n".join(tex) + "\n")
    # strata: for the main FIF run and the B3 baseline on fold A, side by side
    picks = [r for r in rows if r["run"] in ("B3_direct_stereo_rays_A", "FIF_hybrid_stereo_A") and r.get("stratified")]
    if picks:
        keys = [("magnitude", "near(<15)"), ("magnitude", "mid(15-60)"), ("magnitude", "far(>60)"), ("visibility0", "visible"), ("visibility0", "occluded_or_outside"),
                ("object_fov0", "in_view"), ("object_fov0", "partial"), ("object_fov0", "out_of_view"), ("joint_group", "fingertips"), ("joint_group", "wrist"), ("joint_group", "palm"), ("hand", "left"), ("hand", "right")]
        tex = ["\\begin{tabular}{ll" + "r" * len(picks) + "r}", "\\toprule", "Stratum & & " + " & ".join(p["run"].replace("_A", "").replace("_", chr(92) + "_") for p in picks) + " & n \\\\", "\\midrule"]
        for g, k in keys:
            vals = [p["stratified"].get(g, {}).get(k) for p in picks]
            if all(v is None for v in vals): continue
            tex.append(f"{g.replace('_', chr(92)+'_')} & {k.replace('_', chr(92)+'_')} & " + " & ".join(_fmt(v["ade_mm"]) if v else "-" for v in vals) + f" & {next(v['n'] for v in vals if v):,} \\\\")
        for p in picks:
            arv = p["stratified"].get("along_ray_vs_lateral", {})
        tex.append("\\midrule")
        tex.append("along-ray / lateral & & " + " & ".join(f"{_fmt(p['stratified']['along_ray_vs_lateral']['mean_err_par_mm'])} / {_fmt(p['stratified']['along_ray_vs_lateral']['mean_err_lat_mm'])}" for p in picks) + " & \\\\")
        tex.append("direction err (deg, $|v|>20$) & & " + " & ".join(_fmt(p["stratified"].get("direction_err_deg_mag>20"), "{:.1f}") for p in picks) + " & \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]; (out_dir / "strata.tex").write_text("\n".join(tex) + "\n")
    # efficiency
    tex = ["\\begin{tabular}{lrrr}", "\\toprule", "Run & Params (M) & Train (h) & Inference (ms/frame) \\\\", "\\midrule"]
    for r in rows:
        if r["fold"] != "A": continue
        summ_p = EXPERIMENTS_ROOT / r["run"] / "summary.json"; inf = "-"
        if summ_p.exists():
            s = json.load(open(summ_p)); t = s.get("inference_s_per_frame_incl_metrics") or s.get("inference_s_per_frame"); inf = f"{t*1000:.1f}" if t else "-"
        tex.append(f"{r['run'].replace('_A','').replace('_', chr(92)+'_')} & {r['params_M']} & {r['train_h']} & {inf} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]; (out_dir / "efficiency.tex").write_text("\n".join(tex) + "\n")
    print("wrote ablation.tex, strata.tex, efficiency.tex")


if __name__ == "__main__":
    extra_tables()
