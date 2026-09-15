"""Train/evaluate the official InterField ResNet-50 baseline on Apple MPS (or CUDA/CPU).

Re-uses the organisers' model, dataset and loss verbatim (third_party) and only
changes device handling, logging and experiment tracking. Usage:

  python -m fif.train_interfield --frames-dir data/frames/train_10fps --fold A \
      --epochs 20 --batch-size 64 --out experiments/B1_interfield_foldA
"""
from __future__ import annotations
import argparse, json, sys, time, platform
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "SHOW3D-dataset-api"))
from show3d.interaction_field.baseline.data import InterFieldFrameDataset, collate  # noqa: E402
from show3d.interaction_field.baseline.model import InterFieldModel, masked_field_loss  # noqa: E402
from show3d.interaction_field.baseline.predict import camera_to_world  # noqa: E402
from show3d.interaction_field import PredictionRecord, write_submission_jsonl  # noqa: E402
from fif.paths import fold_subjects, SHOW3D_ROOT  # noqa: E402


def pick_device(name: str | None):
    if name: return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def mem_gb(dev):
    if dev.type == "mps": return torch.mps.driver_allocated_memory() / 1e9
    if dev.type == "cuda": return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


@torch.no_grad()
def run_eval(model, loader, dev, frames_ds, out_path: Path | None):
    """Mean camera-frame ADE (== world ADE) plus a world-frame predictions.jsonl."""
    from show3d.interaction_field.baseline.data import _read_rotation_world_from_camera
    model.eval(); tot = 0.0; n = 0; records = []; cache = {}
    for images, targets, masks, metas in loader:
        pred = model(images.to(dev)).float().cpu()
        per_joint = torch.linalg.norm(pred - targets, dim=-1).mean(-1)
        tot += float((per_joint * masks).sum()); n += int(masks.sum())
        if out_path is not None:
            for i, m in enumerate(metas):
                from show3d.dataset import Show3DFrameRef
                ref = Show3DFrameRef(subject_id=m["subject_id"], scene_id=m["scene_id"], frame_index=m["frame_index"])
                R = _read_rotation_world_from_camera(frames_ds.paths.camera_calibration_path(ref, "headset0"), m["frame_index"], cache)
                if R is None: continue
                fields = {"left_to_object": camera_to_world(pred[i, 0].numpy(), R), "right_to_object": camera_to_world(pred[i, 1].numpy(), R)}
                records.append(PredictionRecord(sample_id=m["sample_id"], fields=fields))
    if out_path is not None:
        write_submission_jsonl(out_path, records)
    return tot / max(n, 1), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", type=Path, required=True); p.add_argument("--root", type=Path, default=SHOW3D_ROOT)
    p.add_argument("--fold", default="A"); p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--image-size", type=int, default=224); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-train-batches", type=int, default=None, help="smoke tests")
    p.add_argument("--train-subjects", nargs="*", default=None); p.add_argument("--val-subjects", nargs="*", default=None)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = pick_device(a.device)
    tr_s, va_s = fold_subjects(a.fold)
    if a.train_subjects: tr_s = tuple(a.train_subjects)
    if a.val_subjects: va_s = tuple(a.val_subjects)
    a.out.mkdir(parents=True, exist_ok=True)
    train_ds = InterFieldFrameDataset(a.frames_dir, a.root, subjects=tr_s, image_size=a.image_size, train=True)
    val_ds = InterFieldFrameDataset(a.frames_dir, a.root, subjects=va_s, image_size=a.image_size, train=False)
    tl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, collate_fn=collate, persistent_workers=a.workers > 0)
    vl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0)
    model = InterFieldModel(pretrained=True).to(dev)
    n_params = sum(x.numel() for x in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    config = {"model": "InterField ResNet-50 (official baseline code)", "dataset": "SHOW3D", "fold": a.fold, "train_subjects": tr_s, "val_subjects": va_s,
              "train_frames": len(train_ds), "val_frames": len(val_ds), "backbone": "resnet50 ImageNet1K_V2", "image_size": a.image_size, "views": ["headset0"],
              "lr": a.lr, "batch_size": a.batch_size, "epochs": a.epochs, "optimizer": "AdamW", "weight_decay": a.weight_decay, "scheduler": "cosine",
              "augmentation": "photometric (brightness/contrast/gamma/noise)", "params_total": n_params, "params_trainable": n_params,
              "device": str(dev), "platform": platform.platform(), "torch": torch.__version__, "seed": a.seed, "frames_dir": str(a.frames_dir)}
    (a.out / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps({k: v for k, v in config.items() if k in ("fold", "train_frames", "val_frames", "device", "params_total")}))
    history = []; best = float("inf"); t_train0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0; seen = 0
        for it, (images, targets, masks, _) in enumerate(tl):
            if a.max_train_batches and it >= a.max_train_batches: break
            images, targets, masks = images.to(dev), targets.to(dev), masks.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = masked_field_loss(model(images), targets, masks)
            loss.backward(); opt.step()
            run += loss.item() * images.size(0); seen += images.size(0)
            if it % 100 == 0: print(f"  ep{ep} it{it}/{len(tl)} loss {float(loss):.2f} mm | {(it+1)*a.batch_size/(time.time()-t0):.1f} img/s", flush=True)
        sched.step()
        t_eval = time.time(); val_ade, n_hands = run_eval(model, vl, dev, val_ds, None); t_eval = time.time() - t_eval
        rec = {"epoch": ep, "train_loss_mm": run / max(seen, 1), "val_ade_mm": val_ade, "val_hands": n_hands, "epoch_time_s": time.time() - t0, "eval_time_s": t_eval, "mem_gb": mem_gb(dev)}
        history.append(rec); print(json.dumps(rec), flush=True)
        (a.out / "history.json").write_text(json.dumps(history, indent=1))
        torch.save({"model_state": model.state_dict(), "epoch": ep, "val_ade": val_ade, "config": config}, a.out / "last.pt")
        if val_ade < best:
            best = val_ade; torch.save({"model_state": model.state_dict(), "epoch": ep, "val_ade": val_ade, "config": config}, a.out / "best.pt")
    model.load_state_dict(torch.load(a.out / "best.pt", map_location=dev, weights_only=False)["model_state"])
    t_inf = time.time(); val_ade, n_hands = run_eval(model, vl, dev, val_ds, a.out / "predictions_val.jsonl"); t_inf = (time.time() - t_inf) / max(len(val_ds), 1)
    summary = {"best_val_ade_mm": best, "final_val_ade_mm": val_ade, "val_hands": n_hands, "train_time_s": time.time() - t_train0, "inference_s_per_frame": t_inf, "peak_mem_gb": mem_gb(dev)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))


if __name__ == "__main__":
    main()
