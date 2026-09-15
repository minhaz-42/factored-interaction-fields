"""Fisheye624 camera model (Project Aria / Quest 3 calibration convention) and
rectification of fisheye images to a virtual pinhole camera.

projection_params = [f, cx, cy, k0..k5, p0, p1, s0..s3] (15 values).
Forward projection of a camera-frame point (x, y, z):
  a = x/z, b = y/z, r = sqrt(a^2 + b^2), th = atan(r)
  th_d = th * (1 + k0 th^2 + k1 th^4 + k2 th^6 + k3 th^8 + k4 th^10 + k5 th^12)
  (xr, yr) = th_d / r * (a, b)
  tangential: xr += 2 p0 xr yr + p1 (rd^2 + 2 xr^2);  yr += p0 (rd^2 + 2 yr^2) + 2 p1 xr yr  (rd^2 = xr^2 + yr^2 before tangential)
  thin prism: xr += s0 rd^2 + s1 rd^4;  yr += s2 rd^2 + s3 rd^4
  u = f xr + cx, v = f yr + cy
"""
from __future__ import annotations
import numpy as np
import cv2


def fisheye624_project(points_c: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
    f, cx, cy = params[0], params[1], params[2]
    k = np.asarray(params[3:9]); p0, p1 = params[9], params[10]; s = np.asarray(params[11:15])
    x, y, z = points_c[:, 0], points_c[:, 1], points_c[:, 2]
    valid = z > 1e-6
    zs = np.where(valid, z, 1.0)
    a, b = x / zs, y / zs
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    th2 = th * th
    th_d = th * (1 + k[0] * th2 + k[1] * th2**2 + k[2] * th2**3 + k[3] * th2**4 + k[4] * th2**5 + k[5] * th2**6)
    scale = np.where(r > 1e-9, th_d / np.maximum(r, 1e-9), 1.0)
    xr, yr = a * scale, b * scale
    rd2 = xr * xr + yr * yr
    xt = xr + 2 * p0 * xr * yr + p1 * (rd2 + 2 * xr * xr) + s[0] * rd2 + s[1] * rd2 * rd2
    yt = yr + p0 * (rd2 + 2 * yr * yr) + 2 * p1 * xr * yr + s[2] * rd2 + s[3] * rd2 * rd2
    return np.stack([f * xt + cx, f * yt + cy], 1), valid


def rectification_map(params, src_hw: tuple[int, int], dst_K: np.ndarray, dst_hw: tuple[int, int], R_src_from_dst: np.ndarray | None = None):
    """Pixel map (dst -> src) to build a virtual pinhole image with intrinsics dst_K from a fisheye image.

    R_src_from_dst rotates rays of the virtual camera into the fisheye camera frame
    (identity = same optical axis; use a 90-degree roll to make a portrait image).
    """
    H, W = dst_hw
    ys, xs = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    rays = np.stack([(xs - dst_K[0, 2]) / dst_K[0, 0], (ys - dst_K[1, 2]) / dst_K[1, 1], np.ones_like(xs)], -1).reshape(-1, 3)
    if R_src_from_dst is not None:
        rays = rays @ R_src_from_dst.T
    uv, valid = fisheye624_project(rays, params)
    uv[~valid] = -1
    map_x = uv[:, 0].reshape(H, W).astype(np.float32); map_y = uv[:, 1].reshape(H, W).astype(np.float32)
    return map_x, map_y


def rectify(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def pinhole_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
