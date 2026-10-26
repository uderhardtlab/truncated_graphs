"""Generate bosperrus pipeline figure — uses the bosperrus package for all fitting."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from bosperrus.pipeline import Flow
from bosperrus.fit import (ConstantFit, PiecewiseLinearFit,
                            ExponentialSaturationFit, MichaelisMentenFit)

# ── palette from fit_palette.json ─────────────────────────────────────────────
C_CONST = "#24592F"
C_PIE   = "#0033FF"
C_EXP   = "#AB0C67"
C_MM    = "#FFCC00"

FS = 6.5
FT = 8.0

# ── synthetic data ────────────────────────────────────────────────────────────
np.random.seed(42)
d_obs = np.random.exponential(0.35, 1000)
d_obs = d_obs[d_obs < 1.3][:500]
C_obs = 0.44 * (1 - np.exp(-4.5 * d_obs)) + 0.12 + np.random.normal(0, 0.055, len(d_obs))

# ── run bosperrus ─────────────────────────────────────────────────────────────
distances = pd.Series(d_obs, name="distance_to_border")
scores    = pd.DataFrame({"score": C_obs})

flow = Flow.from_distances_and_scores(distances, scores)
flow.flow()

# All four fits with real AIC (instantiate each directly, same as flow.flow() does)
const = ConstantFit(C_obs, d_obs);                const.fit()
pw    = PiecewiseLinearFit(C_obs, d_obs);         pw.fit_correct()
exp   = ExponentialSaturationFit(C_obs, d_obs);   exp.fit_correct()
mm    = MichaelisMentenFit(C_obs, d_obs);         mm.fit_correct()

print("AIC values from bosperrus fits:")
print(f"  Constant        : {const.AIC:.1f}")
print(f"  Piecewise linear: {pw.AIC:.1f}")
print(f"  Exp. saturation : {exp.AIC:.1f}")
print(f"  Michaelis-Menten: {mm.AIC:.1f}")
print(f"  Best fit        : {flow.best_fits['score'].__class__.__name__}")

# Fitted curve on a fine grid (params follow naming convention in fit.py)
dl = np.linspace(0, 1.32, 300)

c_const = const.params["constant_c"]
C_const_line = np.full_like(dl, c_const)

m_pw = pw.params["piecewise_linear_m"]
b_pw = pw.params["piecewise_linear_b"]
c_pw = pw.params["piecewise_linear_c"]
C_pw_line = np.where(dl <= b_pw, m_pw * dl + c_pw, m_pw * b_pw + c_pw)

a_exp = exp.params["exponential_saturation_a"]
b_exp = exp.params["exponential_saturation_b"]
c_exp = exp.params["exponential_saturation_c"]
C_exp_line = a_exp * (1 - np.exp(-b_exp * dl)) + c_exp
C_plateau  = a_exp + c_exp

a_mm = mm.params["michaelis_menten_a"]
b_mm = mm.params["michaelis_menten_b"]
c_mm = mm.params["michaelis_menten_c"]
C_mm_line = a_mm * dl / (b_mm + dl) + c_mm

# Corrected scores from flow
C_corrected = flow.observations["BOSPERRUS corrected score"].values

# ── figure canvas ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.3), facecolor="white")

P = dict(
    entry = [0.01,  0.09, 0.195, 0.84],
    dist  = [0.23,  0.09, 0.165, 0.84],
    fits  = [0.435, 0.09, 0.185, 0.84],
    aic   = [0.638, 0.09, 0.115, 0.84],
    corr  = [0.780, 0.09, 0.210, 0.84],
)

for ax_x in [0.213, 0.401, 0.625, 0.765]:
    fig.add_artist(FancyArrowPatch(
        (ax_x - 0.008, 0.515), (ax_x + 0.008, 0.515),
        transform=fig.transFigure,
        arrowstyle="->", lw=2.0, color="#555555", mutation_scale=14, clip_on=False))

# ─────────────────────────────────────────────────────────────────────────────
# 1 · ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────
ax_e = fig.add_axes(P["entry"])
ax_e.set_xlim(0, 1); ax_e.set_ylim(0, 1); ax_e.axis("off")
ax_e.text(0.5, 1.04, "Entry point", ha="center", va="bottom",
          fontsize=FT, fontweight="bold")

ENTRIES = [
    ("Node coordinates",   "#dce8f7"),
    ("Coords + edge list", "#dce8f7"),
    ("Coords + scores",    "#dce8f7"),
    ("Distances + scores", "#d4eede"),
]
n_e = len(ENTRIES); bh = 0.17; gap = (1.0 - n_e * bh) / (n_e + 1)

for i, (label, color) in enumerate(ENTRIES):
    y = 1.0 - (i + 1) * (gap + bh)
    ax_e.add_patch(FancyBboxPatch(
        (0.04, y), 0.80, bh, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="#bbbbbb", lw=0.9, transform=ax_e.transAxes))
    ax_e.text(0.44, y + bh / 2, label, ha="center", va="center",
              fontsize=FS, transform=ax_e.transAxes, color="#111111")

bx  = 0.88
y_t = 1.0 - gap
y_b = 1.0 - n_e * (gap + bh)
y_m = (y_t + y_b) / 2
ax_e.plot([bx, bx],        [y_b, y_t], "k-", lw=1.4, transform=ax_e.transAxes)
ax_e.plot([bx, bx + 0.05], [y_m, y_m], "k-", lw=1.4, transform=ax_e.transAxes)

# ─────────────────────────────────────────────────────────────────────────────
# 2 · DISTANCE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
ax_d = fig.add_axes(P["dist"])
ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1); ax_d.axis("off")
ax_d.text(0.5, 1.04, "Distance to…", ha="center", va="bottom",
          fontsize=FT, fontweight="bold")

DISTS = [
    ("Convex hull",  "#f0dde7"),
    ("Bounding box", "#d8e0f4"),
    ("Point set",    "#d4eede"),
    ("Binary mask",  "#f0eed8"),
]
rng = np.random.default_rng(42)

for i, (label, col) in enumerate(DISTS):
    y0 = 1.0 - (i + 1) * (gap + bh)
    ax_d.add_patch(FancyBboxPatch(
        (0.03, y0), 0.94, bh, boxstyle="round,pad=0.01",
        facecolor=col, edgecolor="#cccccc", lw=0.7, transform=ax_d.transAxes))
    ax_d.text(0.72, y0 + bh / 2, label, ha="center", va="center",
              fontsize=FS + 0.5, fontweight="bold",
              transform=ax_d.transAxes, color="#111111", multialignment="center")

    ic = ax_d.inset_axes([0.04, y0 + 0.005, 0.38, bh - 0.012])
    ic.axis("off"); ic.set_xlim(-0.05, 1.05); ic.set_ylim(-0.05, 1.05)
    pts = rng.random((8, 2))
    ic.scatter(pts[:, 0], pts[:, 1], s=10, color="#444444", zorder=3)

    if i == 0:      # convex hull
        hull = ConvexHull(pts)
        hp   = pts[np.append(hull.vertices, hull.vertices[0])]
        ic.plot(hp[:, 0], hp[:, 1], color=C_EXP, lw=1.4, zorder=2)
        c = pts.mean(0); tip = hp[len(hp) // 3]
        ic.annotate("", xy=tip, xytext=c,
                    arrowprops=dict(arrowstyle="->", lw=0.8,
                                    color="#888888", mutation_scale=6))
    elif i == 1:    # bounding box
        x0r, x1r = pts[:,0].min()-0.04, pts[:,0].max()+0.04
        y0r, y1r = pts[:,1].min()-0.04, pts[:,1].max()+0.04
        ic.add_patch(plt.Rectangle(
            (x0r, y0r), x1r-x0r, y1r-y0r,
            facecolor="none", edgecolor=C_PIE, lw=1.4))
        p = pts[pts[:,0].argmin()]
        ic.annotate("", xy=(x0r, p[1]), xytext=p,
                    arrowprops=dict(arrowstyle="->", lw=0.8,
                                    color="#888888", mutation_scale=6))
    elif i == 2:    # point set
        ref = rng.random((3, 2)) * 0.35 + 0.58
        ic.scatter(ref[:,0], ref[:,1], s=16, color=C_MM, marker="D", zorder=4)
        for p in pts[:4]:
            nr = ref[np.argmin(np.sum((ref - p)**2, axis=1))]
            ic.annotate("", xy=nr, xytext=p,
                        arrowprops=dict(arrowstyle="->", lw=0.6,
                                        color="#aaaaaa", mutation_scale=5))
    elif i == 3:    # binary mask
        mv = np.array([[.08,.1],[.08,.9],[.65,.9],[.65,.5],
                        [.92,.5],[.92,.1],[.08,.1]])
        ic.fill(mv[:,0], mv[:,1], alpha=0.28, color=C_CONST)
        ic.plot(mv[:,0], mv[:,1], color=C_CONST, lw=1.3)
        inside = Path(mv).contains_points(pts)
        ic.scatter(pts[inside,0],  pts[inside,1],  s=10, color=C_CONST, zorder=4)
        ic.scatter(pts[~inside,0], pts[~inside,1], s=10, color="#666666", zorder=4)

ax_d.annotate("", xy=(1.09, 0.5), xytext=(1.01, 0.5),
               xycoords="axes fraction", textcoords="axes fraction",
               arrowprops=dict(arrowstyle="->", lw=1.8,
                               color="#444444", mutation_scale=12))

# ─────────────────────────────────────────────────────────────────────────────
# 3 · ① MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────────
ax_f = fig.add_axes(P["fits"])
ax_f.scatter(d_obs, C_obs, s=2, color="#999999", alpha=0.35, zorder=1)
ax_f.plot(dl, C_const_line, lw=2.0, color=C_CONST, label="Constant",         zorder=3)
ax_f.plot(dl, C_pw_line,    lw=2.0, color=C_PIE,   label="Piecewise linear",  zorder=3)
ax_f.plot(dl, C_exp_line,   lw=2.4, color=C_EXP,   label="Exp. saturation",   zorder=4)
ax_f.plot(dl, C_mm_line,    lw=2.0, color=C_MM,     label="Michaelis-Menten",  zorder=3)

ax_f.set_xlabel("Distance", fontsize=FS)
ax_f.set_ylabel("Score",    fontsize=FS, labelpad=1)
ax_f.yaxis.set_label_coords(-0.13, 0.5)
ax_f.set_title("① Fit models", fontsize=FT, fontweight="bold", pad=4)
ax_f.tick_params(labelsize=6, length=3)
ax_f.spines[["top","right"]].set_visible(False)
ax_f.set_xlim(-0.05, 1.38)

ax_f.annotate("", xy=(1.10, 0.5), xytext=(1.01, 0.5),
               xycoords="axes fraction", textcoords="axes fraction",
               arrowprops=dict(arrowstyle="->", lw=1.8,
                               color="#444444", mutation_scale=12))

# ─────────────────────────────────────────────────────────────────────────────
# 4 · ② AIC MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────
ax_a = fig.add_axes(P["aic"])

m_labels = ["Constant", "PW linear", "Exp. sat.", "MM"]
aic_vals  = [const.AIC, pw.AIC, exp.AIC, mm.AIC]
bar_cols  = [C_CONST,   C_PIE,  C_EXP,   C_MM]

order  = np.argsort(aic_vals)          # ascending → winner first
ml_s   = [m_labels[k] for k in order]
av_s   = [aic_vals[k] for k in order]
bc_s   = [bar_cols[k] for k in order]
y_pos  = np.arange(len(order))[::-1]  # winner at top

bars = ax_a.barh(y_pos, av_s, color=bc_s, height=0.55, zorder=2)
bars[0].set_edgecolor("#222222"); bars[0].set_linewidth(2)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(ml_s, fontsize=6)
ax_a.set_xlabel("AIC", fontsize=FS)
ax_a.set_title("② Select best\nmodel", fontsize=FT, fontweight="bold", pad=4)
ax_a.tick_params(axis="x", labelsize=6, length=3)
ax_a.spines[["top","right"]].set_visible(False)

x_range = max(av_s) - min(av_s)
ax_a.text(av_s[0] + x_range * 0.03, y_pos[0],
          "★", fontsize=9, va="center", color=C_EXP, fontweight="bold")

ax_a.annotate("", xy=(1.12, 0.5), xytext=(1.02, 0.5),
               xycoords="axes fraction", textcoords="axes fraction",
               arrowprops=dict(arrowstyle="->", lw=1.8,
                               color="#444444", mutation_scale=12))

# ─────────────────────────────────────────────────────────────────────────────
# 5 · ③ CORRECT SCORES
# ─────────────────────────────────────────────────────────────────────────────
ax_c = fig.add_axes(P["corr"])

ax_c.scatter(d_obs, C_obs,       s=3, color="#aaaaaa", alpha=0.40, zorder=1, label="Raw")
ax_c.scatter(d_obs, C_corrected, s=3, color=C_EXP,    alpha=0.55, zorder=2, label="Corrected")
ax_c.axhline(C_plateau, color="#444444", lw=1.0, ls="--", zorder=3)
ax_c.text(1.32, C_plateau + 0.01, "plateau", fontsize=6, va="bottom", ha="right")

# Correction arrow at a peripheral example node
idx_ex = np.argmin(np.abs(d_obs - 0.065))
d_ex   = d_obs[idx_ex]
C_ex   = C_obs[idx_ex]
ax_c.annotate("", xy=(d_ex, C_plateau - 0.015), xytext=(d_ex, C_ex + 0.015),
               arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))
ax_c.text(d_ex + 0.07, (C_ex + C_plateau) / 2,
          r"$\Delta(d)$", fontsize=8.5, va="center", color="#333333")

ax_c.set_xlabel("Distance", fontsize=FS)
ax_c.set_ylabel("Score",    fontsize=FS, labelpad=1)
ax_c.set_title("③ Correct scores", fontsize=FT, fontweight="bold", pad=4)
ax_c.tick_params(labelsize=6, length=3)
ax_c.spines[["top","right"]].set_visible(False)
ax_c.set_xlim(-0.05, 1.38)

ax_c.legend(
    handles=[
        Line2D([],[],marker="o",color="w",markerfacecolor="#aaaaaa",ms=6,label="Raw"),
        Line2D([],[],marker="o",color="w",markerfacecolor=C_EXP,    ms=6,label="Corrected"),
    ],
    fontsize=6, loc="lower right", frameon=False, handletextpad=0.3)

# ── Global legend ─────────────────────────────────────────────────────────────
fig.legend(
    handles=[
        Line2D([],[],color=C_CONST,lw=2.0,label="Constant fit"),
        Line2D([],[],color=C_PIE,  lw=2.0,label="Piecewise linear fit"),
        Line2D([],[],color=C_EXP,  lw=2.4,label="Exp. saturation fit"),
        Line2D([],[],color=C_MM,   lw=2.0,label="Michaelis-Menten fit"),
    ],
    loc="lower center", ncol=4, fontsize=6.5, frameon=False,
    bbox_to_anchor=(0.6, -0.02))

# ── Save ──────────────────────────────────────────────────────────────────────
out = "bosperrus/figures/Fig_pipeline"
plt.savefig(out + ".svg", format="svg", bbox_inches="tight")
plt.savefig(out + ".pdf", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}.svg / .pdf")
