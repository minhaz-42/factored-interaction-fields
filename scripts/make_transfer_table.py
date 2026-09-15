"""Collect cross-dataset results into tables/transfer.json and tables/transfer.md.

Sources: experiments/<run>/metrics_<tag>.json where tag encodes the evaluation set, e.g.
  metrics_val.json            -> in-domain validation of the run's training set(s)
  metrics_hot3d.json          -> zero-shot evaluation on HOT3D-eval
Run names encode the training set: *_A / *_B (SHOW3D folds), *_hot3d (HOT3D-train), *_joint (SHOW3D+HOT3D).
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import EXPERIMENTS_ROOT, PROJECT_ROOT


def main():
    table = {}
    for run in sorted(p for p in EXPERIMENTS_ROOT.iterdir() if p.is_dir() and not p.name.startswith("smoke")):
        fam = re.sub(r"_(A|B|hot3d|joint)$", "", run.name); train = run.name.split("_")[-1]
        train_set = {"A": "SHOW3D", "B": "SHOW3D", "hot3d": "HOT3D", "joint": "SHOW3D+HOT3D"}.get(train)
        if train_set is None: continue
        for mp in run.glob("metrics_*.json"):
            tag = mp.stem.replace("metrics_", ""); met = json.load(open(mp))["official"]
            eval_set = {"val": ("SHOW3D" if train in ("A", "B", "joint") else "HOT3D"), "hot3d": "HOT3D", "show3d": "SHOW3D"}.get(tag, tag)
            key = f"{train_set}->{eval_set}" + (f" (fold {train})" if train in ("A", "B") else "")
            table.setdefault(fam, {})[key] = met["mean_ade_mm"]
    (PROJECT_ROOT / "tables").mkdir(exist_ok=True)
    (PROJECT_ROOT / "tables/transfer.json").write_text(json.dumps(table, indent=1))
    keys = sorted({k for v in table.values() for k in v})
    md = ["| method | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)] + [f"| {m} | " + " | ".join(f"{table[m][k]:.2f}" if k in table[m] else "-" for k in keys) + " |" for m in table]
    (PROJECT_ROOT / "tables/transfer.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
