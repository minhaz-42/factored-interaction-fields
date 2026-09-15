"""Cache frozen DINOv3 patch tokens for extracted frames.

For each (subject, scene, view) writes data/cache/<model>_<H>x<W>[_pcaK]/<subject>/<scene>/<view>/{tokens,scale,frame_index}.npy with
  tokens: int8 (n_frames, N_patches, C_pca), scale: float16 (n_frames, N_patches),
  frame_index: int32 (n_frames,), plus a global pca.npz (mean, components).
Input frames are the grayscale JPEGs written by the official extract_images tool
(portrait 1280x1024), resized to H x W and replicated to three channels.
PCA is fitted on tokens of a random sample of training-subject frames only.
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch, cv2, timm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import FRAMES_ROOT, CACHE_ROOT, TRAIN_SUBJECTS

MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def load_batch(paths, H, W):
    arr = np.empty((len(paths), 3, H, W), np.float32)
    for i, p in enumerate(paths):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        arr[i] = (np.repeat(g[None], 3, 0) - MEAN) / STD
    return torch.from_numpy(arr)


@torch.no_grad()
def tokens(model, x, dev):
    f = model.forward_features(x.to(dev).half())  # (B, 1+reg+N, C)
    n_prefix = model.num_prefix_tokens
    return f[:, n_prefix:, :].float()


def fit_pca(model, frames_dir, index_rows, H, W, dev, n_imgs, dim, seed=0):
    rng = np.random.default_rng(seed)
    rows = [r for r in index_rows if r["subject_id"] in TRAIN_SUBJECTS]
    pick = rng.choice(len(rows), size=min(n_imgs, len(rows)), replace=False)
    feats = []
    for i in range(0, len(pick), 16):
        paths = [frames_dir / rows[j]["image"] for j in pick[i:i + 16]]
        t = tokens(model, load_batch(paths, H, W), dev)
        feats.append(t.reshape(-1, t.shape[-1]).cpu().numpy())
    X = np.concatenate(feats, 0)
    sub = X[rng.choice(len(X), size=min(200000, len(X)), replace=False)]
    mean = sub.mean(0)
    U, S, Vt = np.linalg.svd(sub - mean, full_matrices=False)
    comps = Vt[:dim]
    var = (S ** 2); print(f"PCA fitted on {len(sub)} tokens: {dim} comps keep {var[:dim].sum()/var.sum():.3f} of variance", flush=True)
    return mean.astype(np.float32), comps.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, default=FRAMES_ROOT / "train_10fps")
    ap.add_argument("--model", default="vit_base_patch16_dinov3.lvd1689m")
    ap.add_argument("--height", type=int, default=480); ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--pca-dim", type=int, default=256); ap.add_argument("--pca-imgs", type=int, default=300)
    ap.add_argument("--stride", type=int, default=2, help="keep every k-th extracted frame (e.g. 10 fps frames, stride 2 -> 5 fps)")
    ap.add_argument("--source-fps", type=int, default=60); ap.add_argument("--extract-fps", type=int, default=10)
    ap.add_argument("--views", nargs="+", default=["headset0", "headset1"])
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--aux", type=Path, default=None, help="aux_labels jsonl: cache only frames that are tracking-valid and have a field, a confident hand or a confident object")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(a.model, pretrained=True, num_classes=0, img_size=(a.height, a.width)).to(dev).eval().half()
    tag = f"{a.model.split('.')[0]}_{a.height}x{a.width}" + ("" if a.no_pca else f"_pca{a.pca_dim}")
    out_root = CACHE_ROOT / tag; out_root.mkdir(parents=True, exist_ok=True)
    index = [json.loads(l) for l in open(a.frames_dir / "index.jsonl") if l.strip()]
    index = [r for r in index if r["view"] in a.views and (a.subjects is None or r["subject_id"] in a.subjects)]
    if a.aux is not None:
        keep = set()
        for l in open(a.aux):
            if not l.strip(): continue
            r = json.loads(l)
            if r["tracking_valid"] and (any(r["field_valid"]) or max(r["hand_conf"]) > 0.5 or r["object_conf"] > 0.5):
                keep.add(r["sample_id"])
        n0 = len(index); index = [r for r in index if r["sample_id"] in keep]
        print(f"aux filter: {n0} -> {len(index)} images", flush=True)
    step = max(1, a.source_fps // a.extract_fps)
    index = [r for r in index if (int(r["frame_index"]) // step) % a.stride == 0]  # extracted at extract_fps: stride k keeps extract_fps/k
    pca_path = out_root / "pca.npz"
    if not a.no_pca:
        if pca_path.exists():
            z = np.load(pca_path); mean, comps = z["mean"], z["components"]
        else:
            mean, comps = fit_pca(model, a.frames_dir, index, a.height, a.width, dev, a.pca_imgs, a.pca_dim)
            np.savez(pca_path, mean=mean, components=comps)
        mean_t, comps_t = torch.from_numpy(mean).to(dev), torch.from_numpy(comps).to(dev)
    groups = {}
    for r in index:
        groups.setdefault((r["subject_id"], r["scene_id"], r["view"]), []).append(r)
    t0 = time.time(); done = 0; fresh = 0
    meta = {"model": a.model, "height": a.height, "width": a.width, "pca_dim": (None if a.no_pca else a.pca_dim), "stride": a.stride,
            "frames_dir": str(a.frames_dir), "views": a.views, "n_images": len(index)}
    (out_root / "meta.json").write_text(json.dumps(meta, indent=1))
    pool = ThreadPoolExecutor(max_workers=3)
    for (subj, scene, view), rows in sorted(groups.items()):
        outp = out_root / subj / scene / view
        if (outp / "tokens.npy").exists():
            done += len(rows); continue
        rows = sorted(rows, key=lambda r: int(r["frame_index"]))
        toks, scales = [], []
        batches = [rows[i:i + a.batch] for i in range(0, len(rows), a.batch)]
        futs = [pool.submit(load_batch, [a.frames_dir / r["image"] for r in b], a.height, a.width) for b in batches[:3]]
        for bi in range(len(batches)):
            x = futs[bi].result()
            if bi + 3 < len(batches):
                futs.append(pool.submit(load_batch, [a.frames_dir / r["image"] for r in batches[bi + 3]], a.height, a.width))
            t = tokens(model, x, dev)
            if not a.no_pca:
                t = (t - mean_t) @ comps_t.T
            s = t.abs().amax(-1).clamp_min(1e-6) / 127.0
            q = torch.round(t / s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
            toks.append(q.cpu().numpy()); scales.append(s.half().cpu().numpy())
        outp.mkdir(parents=True, exist_ok=True)
        np.save(outp / "tokens.npy", np.concatenate(toks)); np.save(outp / "scale.npy", np.concatenate(scales))
        np.save(outp / "frame_index.npy", np.array([int(r["frame_index"]) for r in rows], np.int32))
        done += len(rows); fresh += len(rows)
        el = time.time() - t0; rate = fresh / max(el, 1e-6)
        print(f"{done}/{len(index)} imgs | {rate:.1f} img/s | eta {(len(index)-done)/max(rate,1e-6)/60:.0f} min | {subj}/{scene}/{view}", flush=True)
    print(f"CACHE_COMPLETE {done}/{len(index)} images in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
