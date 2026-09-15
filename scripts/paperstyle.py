"""House style shared by every paper figure (Image and Vision Computing, elsarticle 5p two-column).

Widths: column 252 pt = 3.5 in, text 522 pt = 7.25 in. Figures are drawn at their final size so
that no scaling happens in LaTeX and all lettering is 6.5 to 7.5 pt on the page (Elsevier asks
for uniform lettering in Arial/Helvetica or Times). One colour code is used throughout:
direct regressor blue, factored model red, geometric head green, ground truth grey; in image
overlays the left hand is orange, the right hand blue, the object green and predictions red.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL, FULL = 3.5, 7.25
# chart colours (Paul Tol's colour-blind-safe palettes)
DIRECT, FACTORED, GEO, ORANGE, PURPLE, GREY, GT = "#4477AA", "#CC3311", "#228833", "#EE7733", "#AA3377", "#9E9E9E", "#666666"
DIRECT_LIGHT, FACTORED_LIGHT, GEO_LIGHT, GEO_PALE = "#9DBBD9", "#E8998A", "#8FCB8F", "#CDE7CD"
# overlay colours
LEFT, RIGHT, OBJ, PRED, GTVEC = "#E07B39", "#3E7CB1", "#3C9D4E", "#CC3311", "#5A5A5A"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def setup(size=7):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": size, "axes.labelsize": size, "axes.titlesize": size, "xtick.labelsize": size - 0.5, "ytick.labelsize": size - 0.5,
        "legend.fontsize": size - 0.5, "legend.frameon": False, "legend.handlelength": 1.2, "legend.handleheight": 0.7,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0, "lines.markersize": 3,
        "patch.linewidth": 0, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stixsans",
        "axes.titlelocation": "left", "axes.titlepad": 3, "axes.labelpad": 2, "axes.titleweight": "normal",
        "savefig.dpi": 300, "figure.dpi": 100,
    })


def title(ax, letter, text=""):
    """Panel heading: bold letter followed by a short title, left aligned above the axes."""
    ax.set_title((r"$\mathbf{(" + letter + ")}$ " if letter else "") + text, loc="left")


def label_panel(fig, ax, letter, dx=0.0, dy=0.004):
    """Bold panel letter at the top-left corner of an image panel (axes without title)."""
    b = ax.get_position()
    fig.text(b.x0 + dx, b.y1 + dy, r"$\mathbf{(" + letter + ")}$", ha="left", va="bottom", fontsize=7.5)


def bar_text(ax, x, y, s, above=True, color="#222222", size=5.6, dy=None, **kw):
    lo, hi = ax.get_ylim(); d = dy if dy is not None else 0.012 * (hi - lo)
    ax.text(x, y + (d if above else -d), s, ha="center", va="bottom" if above else "top", fontsize=size, color=color, **kw)


def n_labels(ax, xs, ns, y=-0.34, size=5.5):
    for x, n in zip(xs, ns):
        ax.text(x, y, f"n = {n/1000:.0f}k", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=size, color="#666666")


def save(fig, name, png_dpi=220):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf"); fig.savefig(FIG_DIR / f"{name}.png", dpi=png_dpi)
    print("wrote", FIG_DIR / name)


# ----------------------------------------------------------------------------------------------
# 3-D scene rendering without the matplotlib axes box: shaded object mesh, skeletons, vectors.
# ----------------------------------------------------------------------------------------------
BONES = [(5, 6), (6, 7), (7, 0), (5, 8), (8, 9), (9, 10), (10, 1), (5, 11), (11, 12), (12, 13), (13, 2), (5, 14), (14, 15), (15, 16), (16, 3), (5, 17), (17, 18), (18, 19), (19, 4)]


def scene_3d(ax, joints, fields_gt=None, fields_pred=None, verts=None, faces=None, elev=18, azim=-60, mesh_alpha=0.55, triad=True, vec_lw=0.6, max_faces=40000, zoom=1.35):
    """Draw joints (x, depth, -y), the posed object as a shaded surface and the field vectors.

    The matplotlib axes box, panes and ticks are removed; a small triad in the lower left gives the
    orientation (x right, z depth, y up). Ground-truth vectors are dark grey, predictions red."""
    pts = []
    if verts is not None and faces is not None:
        F = faces
        if len(F) > max_faces:  # decimate by random subset of faces keeps the silhouette at this size
            F = F[np.random.default_rng(0).choice(len(F), size=max_faces, replace=False)]
        ax.plot_trisurf(verts[:, 0], verts[:, 2], -verts[:, 1], triangles=F, color=OBJ, alpha=mesh_alpha, linewidth=0, antialiased=False, shade=True, rasterized=True, zorder=1)
        pts.append(verts)
    for k, col in enumerate((LEFT, RIGHT)):
        J = None if joints is None else joints[k]
        if J is None: continue
        pts.append(J)
        for i, j in BONES: ax.plot([J[i, 0], J[j, 0]], [J[i, 2], J[j, 2]], [-J[i, 1], -J[j, 1]], c=col, lw=1.1, zorder=5)
        ax.scatter(J[:, 0], J[:, 2], -J[:, 1], s=6, c=col, linewidths=0, zorder=6, depthshade=False)
        for fields, col2 in ((fields_gt, GTVEC), (fields_pred, PRED)):
            if fields is None or fields[k] is None: continue
            Fv = fields[k]
            for i in range(21): ax.plot([J[i, 0], J[i, 0] + Fv[i, 0]], [J[i, 2], J[i, 2] + Fv[i, 2]], [-J[i, 1], -J[i, 1] - Fv[i, 1]], c=col2, lw=vec_lw, zorder=4)
    allp = np.concatenate(pts); c = allp.mean(0); r = (allp.max(0) - allp.min(0)).max() / 2 * 1.02
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[2] - r, c[2] + r); ax.set_zlim(-c[1] - r, -c[1] + r)
    ax.set_box_aspect((1, 1, 1), zoom=zoom); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    if triad:
        o = np.array([c[0] - 0.95 * r, c[2] - 0.95 * r, -c[1] - 0.95 * r]); L = 0.35 * r
        for d, lab in (((1, 0, 0), "x"), ((0, 1, 0), "z"), ((0, 0, 1), "y")):
            d = np.array(d) * L
            ax.plot([o[0], o[0] + d[0]], [o[1], o[1] + d[1]], [o[2], o[2] + d[2]], c="#444444", lw=0.7, zorder=10)
            ax.text(o[0] + d[0] * 1.25, o[1] + d[1] * 1.25, o[2] + d[2] * 1.25, lab, fontsize=6, color="#444444", ha="center", va="center", zorder=11)
    return c, r


def triad_overlay(fig, ax, anchor=(0.86, 0.16), length_in=0.2, labels=("x", "z", "y"), color="#444444"):
    """Small orientation triad drawn in figure space at a corner of a 3-D axes, using the axes' own
    projection so that the arrows point along the projected x, depth and up directions."""
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import FancyArrowPatch
    fig.canvas.draw()
    Mp = ax.get_proj(); c = np.array([np.mean(ax.get_xlim()), np.mean(ax.get_ylim()), np.mean(ax.get_zlim())]); r = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
    to_fig = fig.transFigure.inverted(); disp = lambda p: np.array(ax.transData.transform(proj3d.proj_transform(*p, Mp)[:2]))
    b = ax.get_position(); o_fig = np.array([b.x0 + anchor[0] * b.width, b.y0 + anchor[1] * b.height])
    o_disp = fig.transFigure.transform(o_fig); L = length_in * fig.dpi
    for d, lab in zip(np.eye(3), labels):
        v = disp(c + d * r) - disp(c); v = v / np.linalg.norm(v) * L
        p1 = to_fig.transform(o_disp + v); p2 = to_fig.transform(o_disp + v * 1.3)
        fig.patches.append(FancyArrowPatch(o_fig, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=4, lw=0.6, color=color, shrinkA=0, shrinkB=0))
        fig.text(p2[0], p2[1], lab, fontsize=6, color=color, ha="center", va="center")
