"""Export a run's analysis tables (tables/analysis_<run>.md): official metrics, head/oracle
decomposition, stratified ADE by magnitude, visibility, observability, joint group, hand,
object and subject, along-ray vs lateral error, direction/magnitude errors, surface distance.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import PROJECT_ROOT


def table(d: dict, title: str, cols=("ade_mm", "n")):
    lines = [f"**{title}**", "", "| stratum | ADE (mm) | n |", "|---|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {v['ade_mm']:.2f} | {v['n']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); a = ap.parse_args()
    met = json.load(open(a.run / "metrics_val.json"))
    out = [f"# Analysis: {a.run.name}", ""]
    off = met["official"]
    out.append(f"Official: mean ADE {off['mean_ade_mm']:.2f} mm | recall {off['mean_recall']:.3f} | acc@10 {off['mean_acc@10mm']:.3f} | acc@50 {off['mean_acc@50mm']:.3f} | acc@100 {off['mean_acc@100mm']:.3f} | left {off['left_to_object']['ade_mm']:.2f} | right {off['right_to_object']['ade_mm']:.2f}\n")
    for k in ("official_dir_head", "official_geo_head", "oracle_hand_pred_obj_gt", "oracle_hand_gt_obj_pred"):
        if k in met and met[k].get("mean_ade_mm") is not None: out.append(f"- {k}: {met[k]['mean_ade_mm']:.2f} mm")
    for k in ("surface_distance_vertex", "surface_distance_vertex_dir_head", "surface_distance_vertex_geo_head"):
        if k in met and met[k].get("surface_distance_mm") is not None: out.append(f"- {k}: {met[k]['surface_distance_mm']:.2f} mm (n={met[k]['n']})")
    st = met.get("stratified", {})
    out.append(""); out.append(f"Along-ray error {st['along_ray_vs_lateral']['mean_err_par_mm']:.2f} mm vs lateral {st['along_ray_vs_lateral']['mean_err_lat_mm']:.2f} mm; direction error (|v|>20 mm) {st['direction_err_deg_mag>20']:.1f} deg; magnitude error {st['magnitude_err_mm']:.2f} mm\n")
    for key, title in (("magnitude", "By field magnitude"), ("visibility0", "By joint visibility (view 0)"), ("object_fov0", "By object field-of-view fraction (view 0)"), ("hand_fov0", "By hand field-of-view fraction"), ("joint_group", "By joint group"), ("hand", "By hand"), ("object", "By object"), ("subject", "By subject")):
        if key in st: out.append(table(st[key], title))
    p = PROJECT_ROOT / "tables" / f"analysis_{a.run.name}.md"; p.parent.mkdir(exist_ok=True); p.write_text("\n".join(out)); print("wrote", p)


if __name__ == "__main__":
    main()
