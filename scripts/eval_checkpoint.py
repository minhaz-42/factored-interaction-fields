"""Evaluate a trained FIF checkpoint on any (cache, aux, subjects) set, e.g. zero-shot HOT3D-IF.

  python scripts/eval_checkpoint.py --run experiments/FIF_hybrid_stereo_A --cache <cache tag> --aux aux_labels_hot3d_10fps.jsonl --subjects P0017 P0018 P0021 --tag hot3d
Writes experiments/<run>/metrics_<tag>.json, predictions_<tag>.jsonl, uncertainty_<tag>.npz.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fif  # noqa
from fif.paths import CACHE_ROOT, DATA_ROOT
from fif.data_cache import CachedFieldDataset
from fif.models import FIFModel
from fif.train_fif import evaluate, pick_device


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--cache", required=True); ap.add_argument("--aux", required=True)
    ap.add_argument("--subjects", nargs="+", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--ckpt", default="best.pt"); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.run / "config.yaml")); dev = pick_device(a.device)
    views = tuple(cfg["views"]); ds = CachedFieldDataset(CACHE_ROOT / a.cache, DATA_ROOT / "aux" / a.aux, tuple(a.subjects), views, n_ctrl=cfg.get("n_ctrl", 16), n_verts=cfg.get("n_verts", 1024))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("workers", 4))
    meta = json.loads((CACHE_ROOT / a.cache / "meta.json").read_text())
    model = FIFModel(c_in=meta["pca_dim"] or 768, d=cfg.get("d_model", 384), heads=cfg.get("heads", 8), layers=cfg.get("layers", 6), ff=cfg.get("ff", 1536), drop=cfg.get("dropout", 0.1),
                     mode=cfg["mode"], use_rays=cfg.get("use_rays", True), n_ctrl=cfg.get("n_ctrl", 16), fusion=cfg.get("fusion", "precision"), n_views=len(views), n_tokens=(meta["height"] // 16) * (meta["width"] // 16)).to(dev)
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location=dev, weights_only=False)["model_state"])
    res = evaluate(model, ds, dl, dev, cfg, a.run, a.tag, full=True)
    print(json.dumps({"tag": a.tag, "frames": len(ds), **{k: v for k, v in res["official"].items() if k.startswith("mean")}}))


if __name__ == "__main__":
    main()
