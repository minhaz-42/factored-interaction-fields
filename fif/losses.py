"""Training objective of docs/07 Sec. 7.5."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def masked_ade(pred, gt, mask):
    """pred/gt (B,2,J,3); mask (B,2) -> mean per-joint L2 over valid hands (mm)."""
    e = torch.linalg.norm(pred - gt, dim=-1).mean(-1)  # (B,2)
    return (e * mask).sum() / mask.sum().clamp_min(1.0)


def gaussian_nll(pred, gt, log_s, mask):
    """Isotropic Gaussian NLL on the 3-D residual; log_s (B,2,J)."""
    r2 = ((pred - gt) ** 2).sum(-1)  # (B,2,J)
    nll = 0.5 * r2 * torch.exp(-2 * log_s) + 3 * log_s
    m = mask.unsqueeze(-1).expand_as(nll)
    return (nll * m).sum() / m.sum().clamp_min(1.0)


def joint_l1(pred, gt, mask):
    e = (pred - gt).abs().sum(-1).mean(-1)  # (B,2)
    return (e * mask).sum() / mask.sum().clamp_min(1.0)


def add_s(verts_pred, verts_gt, vmask, obj_mask):
    """Symmetry-agnostic ADD-S: mean over predicted vertices of min distance to GT vertices."""
    d = torch.cdist(verts_pred, verts_gt)  # (B,M,M)
    d = d.masked_fill(~vmask.unsqueeze(1), float("inf"))
    mind = d.min(-1).values  # (B,M)
    per = (mind * vmask).sum(-1) / vmask.sum(-1).clamp_min(1)
    return (per * obj_mask).sum() / obj_mask.sum().clamp_min(1.0)


def bce_masked(logit, target, mask):
    l = F.binary_cross_entropy_with_logits(logit, target.float(), reduction="none")
    return (l * mask).sum() / mask.sum().clamp_min(1.0)


def total_loss(out, batch, w, mode: str):
    """out: model dict; batch: dict of tensors; w: dict of weights. Returns (loss, logs)."""
    logs = {}
    fm = batch["field_mask"]  # (B,2)
    L = masked_ade(out["v"], batch["field"], fm); logs["ade"] = L.detach()
    total = L
    if mode in ("direct", "hybrid") and w.get("dir", 0) > 0:
        l = gaussian_nll(out["v_dir"], batch["field"], out["s_dir"], fm); logs["nll_dir"] = l.detach(); total = total + w["dir"] * l
    if mode in ("geo", "hybrid"):
        if w.get("geo", 0) > 0:
            l = gaussian_nll(out["v_geo"], batch["field"], out["s_geo"], fm); logs["nll_geo"] = l.detach(); total = total + w["geo"] * l
        if w.get("obj", 0) > 0:
            l = add_s(out["verts"], batch["verts_gt"], batch["verts_mask"], batch["obj_mask"]) + \
                ((out["t_obj"] - batch["t_obj"]).abs().sum(-1) * batch["obj_mask"]).sum() / batch["obj_mask"].sum().clamp_min(1.0)
            logs["obj"] = l.detach(); total = total + w["obj"] * l
    if w.get("joint", 0) > 0:
        l = joint_l1(out["joints"], batch["joints"], batch["hand_mask"]); logs["joint"] = l.detach(); total = total + w["joint"] * l
    if w.get("vis", 0) > 0:
        l = bce_masked(out["vis_logit"], batch["vis"], batch["vis_mask"]); logs["vis"] = l.detach(); total = total + w["vis"] * l
    if w.get("pres", 0) > 0:
        l = F.binary_cross_entropy_with_logits(out["pres_logit"], fm.float()); logs["pres"] = l.detach(); total = total + w["pres"] * l
    if w.get("cat", 0) > 0:
        l = F.cross_entropy(out["cat_logit"], batch["alias"]); logs["cat"] = l.detach(); total = total + w["cat"] * l
    logs["total"] = total.detach()
    return total, logs
