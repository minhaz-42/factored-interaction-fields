"""FIF model: joint/object queries over camera-aware frozen tokens, with direct and
factored (geometric) field heads and uncertainty-weighted fusion (docs/07).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import geometry as G

J = 21
N_OBJ = 21  # object aliases in the interaction-field task


class TokenEmbed(nn.Module):
    def __init__(self, c_in: int, d: int, n_views: int = 2, use_rays: bool = True, n_tokens: int = 720):
        super().__init__()
        self.proj = nn.Linear(c_in, d)
        self.use_rays = use_rays
        self.ray_mlp = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d)) if use_rays else None
        self.view_emb = nn.Embedding(n_views, d)
        self.pos = nn.Parameter(torch.zeros(n_tokens, d)); nn.init.trunc_normal_(self.pos, std=0.02)  # learned per-token positional embedding

    def forward(self, tokens, rays, view_ids, grid_hw):
        """tokens (B,V,N,C); rays (B,V,N,6); view_ids (V,) long; grid_hw (h,w)."""
        B, V, N, C = tokens.shape
        x = self.proj(tokens)
        assert self.pos.shape[0] == N, f"positional embedding sized for {self.pos.shape[0]} tokens, got {N}"
        x = x + self.pos.view(1, 1, N, -1)
        if self.use_rays:
            x = x + self.ray_mlp(rays)
        x = x + self.view_emb(view_ids).view(1, V, 1, -1)
        return x.reshape(B, V * N, -1)


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ff, drop=0.0):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, mem):
        h = self.n1(q); q = q + self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(q); q = q + self.ca(h, mem, mem, need_weights=False)[0]
        return q + self.ff(self.n3(q))


class FIFModel(nn.Module):
    """Configurable: mode in {'direct','geo','hybrid'}; use_rays; n_ctrl control points."""

    def __init__(self, c_in: int, d: int = 384, heads: int = 8, layers: int = 6, ff: int = 1536, drop: float = 0.1,
                 mode: str = "hybrid", use_rays: bool = True, n_ctrl: int = 16, fusion: str = "precision", n_views: int = 2, n_tokens: int = 720):
        super().__init__()
        assert mode in ("direct", "geo", "hybrid") and fusion in ("precision", "mean", "gate")
        self.mode, self.fusion, self.n_ctrl = mode, fusion, n_ctrl
        self.embed = TokenEmbed(c_in, d, n_views, use_rays, n_tokens)
        self.joint_q = nn.Parameter(torch.randn(2 * J, d) * 0.02)
        self.obj_q = nn.Parameter(torch.randn(n_ctrl, d) * 0.02)
        self.alias_emb = nn.Embedding(N_OBJ, d)
        self.layers = nn.ModuleList([DecoderLayer(d, heads, ff, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        # heads
        self.h_dir = nn.Linear(d, 4)         # vector (3) + log-scale
        self.h_joint = nn.Linear(d, 3)       # camera-0 joints
        self.h_vis = nn.Linear(d, 1)
        self.h_pres = nn.Linear(d, 1)
        self.h_ctrl = nn.Linear(d, 4)        # control point (3) + confidence logit
        self.h_cat = nn.Linear(d, N_OBJ)
        self.h_geo_scale = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Linear(d, 1))
        self.h_gate = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Linear(d, 1)) if fusion == "gate" else None
        # scale of regression outputs (mm): start near dataset scale
        self.out_scale = 100.0

    def forward(self, tokens, rays, view_ids, grid_hw, alias, ctrl_canon, verts_canon, verts_mask, tau: float, hard_nn: bool = False):
        """
        tokens (B,V,N,C) float; rays (B,V,N,6); alias (B,) long; ctrl_canon (B,n_ctrl,3) canonical control points;
        verts_canon (B,M,3) canonical (sub-sampled) vertices with verts_mask (B,M) bool; tau soft-NN temperature (mm^2).
        Returns dict of predictions in the camera-0 frame (mm).
        """
        B = tokens.shape[0]
        mem = self.embed(tokens, rays, view_ids, grid_hw)
        q = torch.cat([self.joint_q.unsqueeze(0).expand(B, -1, -1), (self.obj_q.unsqueeze(0) + self.alias_emb(alias).unsqueeze(1)).expand(B, -1, -1)], 1)
        for layer in self.layers:
            q = layer(q, mem)
        q = self.norm(q)
        zj, zo = q[:, : 2 * J], q[:, 2 * J:]
        out = {}
        d = self.h_dir(zj)
        out["v_dir"] = d[..., :3].view(B, 2, J, 3) * self.out_scale
        out["s_dir"] = d[..., 3].view(B, 2, J).clamp(-3, 5)
        out["joints"] = self.h_joint(zj).view(B, 2, J, 3) * self.out_scale + torch.tensor([0., 0., 300.], device=zj.device)
        out["vis_logit"] = self.h_vis(zj).view(B, 2, J)
        out["pres_logit"] = self.h_pres(zj.view(B, 2, J, -1).mean(2)).squeeze(-1)
        out["cat_logit"] = self.h_cat(zo.mean(1))
        if self.mode in ("geo", "hybrid"):
            c = self.h_ctrl(zo)
            ctrl = c[..., :3] * self.out_scale + torch.tensor([0., 0., 300.], device=zo.device)
            w_ctrl = torch.softmax(c[..., 3], -1)
            out["ctrl"], out["ctrl_w"] = ctrl, w_ctrl
            R, t = G.weighted_procrustes(ctrl_canon, ctrl, w_ctrl)
            out["R_obj"], out["t_obj"] = R, t
            verts = torch.einsum("bij,bmj->bmi", R, verts_canon) + t.unsqueeze(1)  # (B,M,3)
            out["verts"] = verts
            joints = out["joints"].reshape(B, 2 * J, 3)
            d2 = torch.cdist(joints, verts) ** 2
            d2 = d2.masked_fill(~verts_mask.unsqueeze(1), float("inf"))
            if hard_nn:
                idx = d2.argmin(-1); qstar = torch.gather(verts, 1, idx.unsqueeze(-1).expand(-1, -1, 3)); w = None
                ent = torch.zeros(B, 2 * J, device=zj.device)
            else:
                w = torch.softmax(-d2 / tau, -1); qstar = torch.einsum("bjm,bmd->bjd", w, verts)
                ent = -(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum(-1)
            v_geo = (qstar - joints).view(B, 2, J, 3)
            out["v_geo"] = v_geo
            feat = torch.cat([zj, zo.mean(1, keepdim=True).expand(-1, 2 * J, -1), ent.unsqueeze(-1), v_geo.reshape(B, 2 * J, 3).norm(dim=-1, keepdim=True).add(1.0).log()], -1)
            out["s_geo"] = self.h_geo_scale(feat).view(B, 2, J).clamp(-3, 5)
            if self.h_gate is not None:
                out["gate"] = torch.sigmoid(self.h_gate(feat)).view(B, 2, J, 1)
        # fused output
        if self.mode == "direct":
            out["v"] = out["v_dir"]
        elif self.mode == "geo":
            out["v"] = out["v_geo"]
        else:
            if self.fusion == "mean":
                out["v"] = 0.5 * (out["v_dir"] + out["v_geo"])
            elif self.fusion == "gate":
                g = out["gate"]; out["v"] = (1 - g) * out["v_dir"] + g * out["v_geo"]
            else:
                lam_d = torch.exp(-2 * out["s_dir"]).unsqueeze(-1); lam_g = torch.exp(-2 * out["s_geo"]).unsqueeze(-1)
                out["v"] = (lam_d * out["v_dir"] + lam_g * out["v_geo"]) / (lam_d + lam_g)
        return out


def count_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
