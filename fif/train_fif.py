"""Train and evaluate FIF variants on cached tokens (config-driven, docs/07-08).

  python -m fif.train_fif --config configs/FIF_hybrid_stereo.yaml --out experiments/<run>
"""
from __future__ import annotations
import argparse, json, math, sys, time, platform
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fif  # noqa: F401  (warning filters)
from fif.paths import fold_subjects, CACHE_ROOT, DATA_ROOT
from fif.data_cache import CachedFieldDataset, object_geometry
from fif.models import FIFModel, count_params
from fif.losses import total_loss
from fif import metrics as M, geometry as G

FIELD_KEYS = ("left_to_object", "right_to_object")


def pick_device(name=None):
    if name: return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def mem_gb(dev):
    return torch.mps.driver_allocated_memory() / 1e9 if dev.type == "mps" else (torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan"))


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def tau_at(cfg, ep):
    t0, t1 = float(cfg.get("tau_start", 400.0)), float(cfg.get("tau_end", 25.0)); E = max(cfg["epochs"] - 1, 1)
    return float(t0 * (t1 / t0) ** (min(ep, E) / E))


@torch.no_grad()
def evaluate(model, ds, loader, dev, cfg, out_dir: Path | None, tag: str, full: bool = False):
    model.eval()
    preds, pred_geo, pred_dir, oracle_hand, oracle_obj = {}, {}, {}, {}, {}
    unc = {"sample_id": [], "s_dir": [], "s_geo": [], "err_dir": [], "err_geo": [], "err_fused": [], "field_mask": [], "vis_logit": [], "pres_logit": []}
    for b in loader:
        bd = to_dev(b, dev)
        out = model(bd["tokens"], bd["rays"], torch.arange(len(ds.views), device=dev), ds.grid_hw, bd["alias"], bd["ctrl_canon"], bd["verts_canon"], bd["verts_mask"], tau=float(cfg.get("tau_end", 25.0)), hard_nn=True)
        v = out["v"].cpu().numpy(); R0 = b["R0"].numpy()
        if full:
            gt = bd["field"]; fm = bd["field_mask"]
            unc["sample_id"] += [ds.rows[int(i)]["sample_id"] for i in b["idx"]]
            unc["field_mask"].append(fm.cpu().numpy()); unc["err_fused"].append(torch.linalg.norm(out["v"] - gt, dim=-1).cpu().numpy())
            unc["vis_logit"].append(out["vis_logit"].cpu().numpy()); unc["pres_logit"].append(out["pres_logit"].cpu().numpy())
            if "v_geo" in out:
                unc["s_dir"].append(out["s_dir"].cpu().numpy()); unc["s_geo"].append(out["s_geo"].cpu().numpy())
                unc["err_dir"].append(torch.linalg.norm(out["v_dir"] - gt, dim=-1).cpu().numpy()); unc["err_geo"].append(torch.linalg.norm(out["v_geo"] - gt, dim=-1).cpu().numpy())
            elif "s_dir" in out:
                unc["s_dir"].append(out["s_dir"].cpu().numpy()); unc["err_dir"].append(torch.linalg.norm(out["v_dir"] - gt, dim=-1).cpu().numpy())
        vg = out["v_geo"].cpu().numpy() if "v_geo" in out else None
        vd = out["v_dir"].cpu().numpy() if "v_geo" in out else None
        for i in range(v.shape[0]):
            r = ds.rows[int(b["idx"][i])]; sid = r["sample_id"]
            world = lambda x: x @ R0[i].T  # camera-0 -> world (row vectors)
            preds[sid] = {k: world(v[i, h]) for h, k in enumerate(FIELD_KEYS)}
            if vg is not None:
                pred_geo[sid] = {k: world(vg[i, h]) for h, k in enumerate(FIELD_KEYS)}
                pred_dir[sid] = {k: world(vd[i, h]) for h, k in enumerate(FIELD_KEYS)}
                if full and r["obj_R_cam0"] is not None:
                    _, _, Vfull = object_geometry(r["object_alias"], model.n_ctrl, ds.n_verts)
                    Rg, tg = np.asarray(r["obj_R_cam0"]), np.asarray(r["obj_t_cam0"]); verts_gt = Vfull @ Rg.T + tg
                    Rp, tp = out["R_obj"][i].cpu().numpy().astype(np.float64), out["t_obj"][i].cpu().numpy().astype(np.float64); verts_pr = Vfull @ Rp.T + tp
                    pj = out["joints"][i].cpu().numpy().astype(np.float64)
                    oh, oo = {}, {}
                    for h, k in enumerate(FIELD_KEYS):
                        oh[k] = world(G.nearest_vertex_field(pj[h], verts_gt)[0])
                        oo[k] = world(G.nearest_vertex_field(np.asarray(r["joints_cam0"][h]), verts_pr)[0]) if r["joints_cam0"][h] is not None else None
                    oracle_hand[sid], oracle_obj[sid] = oh, oo
    refs = ds.rows
    res = {"official": M.official_metrics(refs, preds)}
    if full:
        rows = M.per_joint_errors(refs, preds); res["stratified"] = M.stratified(rows)
        res["surface_distance_vertex"] = M.surface_distance(refs, preds, "vertex")
        if pred_geo:
            res["official_geo_head"] = M.official_metrics(refs, pred_geo); res["official_dir_head"] = M.official_metrics(refs, pred_dir)
            res["oracle_hand_pred_obj_gt"] = M.official_metrics(refs, oracle_hand); res["oracle_hand_gt_obj_pred"] = M.official_metrics(refs, oracle_obj)
            res["surface_distance_vertex_geo_head"] = M.surface_distance(refs, pred_geo, "vertex")
            res["surface_distance_vertex_dir_head"] = M.surface_distance(refs, pred_dir, "vertex")
    if out_dir is not None and full:
        np.savez_compressed(out_dir / f"uncertainty_{tag}.npz", **{k: (np.concatenate(v) if (v and not isinstance(v[0], str)) else np.asarray(v)) for k, v in unc.items() if len(v)})
    if out_dir is not None:
        with open(out_dir / f"predictions_{tag}.jsonl", "w") as f:
            for sid, d in preds.items():
                f.write(json.dumps({"sample_id": sid, **{k: v.tolist() for k, v in d.items()}}) + "\n")
        (out_dir / f"metrics_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None); ap.add_argument("--override", nargs="*", default=[], help="key=value (yaml) overrides")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.override:
        k, v = kv.split("=", 1); cfg[k] = yaml.safe_load(v)
    torch.manual_seed(cfg.get("seed", 0)); np.random.seed(cfg.get("seed", 0))
    dev = pick_device(a.device); a.out.mkdir(parents=True, exist_ok=True)
    tr_s, va_s = fold_subjects(cfg["fold"])
    if cfg.get("train_subjects"): tr_s = tuple(cfg["train_subjects"])
    if cfg.get("val_subjects"): va_s = tuple(cfg["val_subjects"])
    cache = CACHE_ROOT / cfg["cache"]; aux = DATA_ROOT / "aux" / cfg["aux"]
    views = tuple(cfg.get("views", ["headset0", "headset1"]))
    kw = dict(n_ctrl=cfg.get("n_ctrl", 16), n_verts=cfg.get("n_verts", 1024))
    # training sources: the primary (cache, aux, train subjects) plus optional extra datasets, e.g. HOT3D-IF
    sources = [(cache, aux, tr_s)] + [(CACHE_ROOT / e["cache"], DATA_ROOT / "aux" / e["aux"], tuple(e["subjects"])) for e in cfg.get("extra_train", [])]
    tr_sets = [CachedFieldDataset(c, x, subj, views, require_field=cfg.get("require_field", True), token_drop=cfg.get("token_drop", 0.0), **kw) for c, x, subj in sources]
    tr = tr_sets[0] if len(tr_sets) == 1 else torch.utils.data.ConcatDataset(tr_sets)
    ev = cfg.get("eval")  # optional cross-dataset evaluation set {cache, aux, subjects}
    va = CachedFieldDataset(CACHE_ROOT / ev["cache"], DATA_ROOT / "aux" / ev["aux"], tuple(ev["subjects"]), views, **kw) if ev else CachedFieldDataset(cache, aux, va_s, views, **kw)
    nw = cfg.get("workers", 4)
    tl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    vl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw)
    meta = json.loads((cache / "meta.json").read_text())
    grid_hw = tr_sets[0].grid_hw
    model = FIFModel(c_in=meta["pca_dim"] or 768, d=cfg.get("d_model", 384), heads=cfg.get("heads", 8), layers=cfg.get("layers", 6), ff=cfg.get("ff", 1536),
                     drop=cfg.get("dropout", 0.1), mode=cfg["mode"], use_rays=cfg.get("use_rays", True), n_ctrl=cfg.get("n_ctrl", 16), fusion=cfg.get("fusion", "precision"), n_views=len(views), n_tokens=(meta["height"] // 16) * (meta["width"] // 16)).to(dev)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.05)), betas=(0.9, 0.95))
    steps = cfg["epochs"] * len(tl); warm = cfg.get("warmup_steps", 200)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    config = dict(cfg, train_subjects=list(tr_s), val_subjects=(list(ev["subjects"]) if ev else list(va_s)), train_frames=len(tr), val_frames=len(va), train_frames_per_source=[len(t) for t in tr_sets], params_trainable=n_params, backbone=meta["model"],
                  image_resolution=f'{meta["height"]}x{meta["width"]}', views=list(views), device=str(dev), platform=platform.platform(), torch=str(torch.__version__), optimizer="AdamW", scheduler="warmup+cosine")
    (a.out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({k: config[k] for k in ("mode", "fusion", "use_rays", "views", "train_frames", "val_frames", "params_trainable", "device")}), flush=True)
    w = cfg.get("loss_weights", {"dir": 0.1, "geo": 0.1, "joint": 0.02, "obj": 0.02, "vis": 0.1, "pres": 0.1, "cat": 0.0})
    history, best, t_all = [], float("inf"), time.time()
    patience = cfg.get("patience", 3); since_best = 0
    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time(); agg = {}; n = 0; tau = tau_at(cfg, ep)
        for it, b in enumerate(tl):
            bd = to_dev(b, dev)
            out = model(bd["tokens"], bd["rays"], torch.arange(len(views), device=dev), grid_hw, bd["alias"], bd["ctrl_canon"], bd["verts_canon"], bd["verts_mask"], tau=tau)
            loss, logs = total_loss(out, bd, w, cfg["mode"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0)); opt.step(); sched.step()
            for k, v in logs.items(): agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
            if it % 50 == 0: print(f"  ep{ep} it{it}/{len(tl)} " + " ".join(f"{k}={float(v):.2f}" for k, v in logs.items()) + f" | tau={tau:.0f} | {(it+1)*cfg['batch_size']/(time.time()-t0):.1f} samp/s", flush=True)
        res = evaluate(model, va, vl, dev, cfg, None, "val", full=False)
        val_ade = res["official"]["mean_ade_mm"]
        rec = {"epoch": ep, **{f"train_{k}": v / max(n, 1) for k, v in agg.items()}, "val_ade_mm": val_ade, "val_recall": res["official"]["mean_recall"], "tau": tau, "epoch_time_s": time.time() - t0, "mem_gb": mem_gb(dev)}
        history.append(rec); print(json.dumps(rec), flush=True); (a.out / "history.json").write_text(json.dumps(history, indent=1))
        torch.save({"model_state": model.state_dict(), "epoch": ep, "val_ade": val_ade, "config": config}, a.out / "last.pt")
        if val_ade is not None and val_ade < best - 0.05:
            best = val_ade; since_best = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep, "val_ade": val_ade, "config": config}, a.out / "best.pt")
        else:
            since_best += 1
            if val_ade is not None and val_ade < best:
                best = val_ade; torch.save({"model_state": model.state_dict(), "epoch": ep, "val_ade": val_ade, "config": config}, a.out / "best.pt")
            if since_best >= patience:
                print(json.dumps({"early_stop": ep, "best_val_ade_mm": best, "patience": patience}), flush=True); break
    model.load_state_dict(torch.load(a.out / "best.pt", map_location=dev, weights_only=False)["model_state"])
    t_inf = time.time(); res = evaluate(model, va, vl, dev, cfg, a.out, "val", full=True); t_inf = (time.time() - t_inf) / max(len(va), 1)
    summary = {"best_val_ade_mm": best, "epochs_run": len(history), "early_stopped": len(history) < cfg["epochs"], "final": res["official"], "train_time_s": time.time() - t_all, "inference_s_per_frame_incl_metrics": t_inf, "peak_mem_gb": mem_gb(dev), "params_trainable": n_params}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary["final"]))


if __name__ == "__main__":
    main()
