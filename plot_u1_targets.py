"""Time evolution against the fixed U(1) late-time targets.

    python3 plot_u1_targets.py results/u1_raw_L10_LA2_n100_D0.5_J20.0.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw = sys.argv[1] if len(sys.argv) > 1 else "results/u1_raw_L10_LA2_n100_D0.5_J20.0.csv"
out = sys.argv[2] if len(sys.argv) > 2 else raw.replace("u1_raw_", "u1_time_evolution_").replace(".csv", ".png")
title = sys.argv[3] if len(sys.argv) > 3 else r"XXZ $(1+\Delta)=1.5$, $J_2=0$"

df = pd.read_csv(raw)
m = df[(df.model == "XXZ") & (df.t > 0)]
fl = df[df.model == "U1_random_state"]
nreal = m.realization.nunique()
g = m.groupby("t")
mean, se = g.mean(numeric_only=True), g.std(numeric_only=True) / np.sqrt(nreal)
t = mean.index.values
floor = fl.mean(numeric_only=True)
late = m[m.t >= 50].groupby("realization").mean(numeric_only=True)

C = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), constrained_layout=True)


def band(ax, key, color, label, ls="-"):
    ax.plot(t, mean[key], color=color, lw=2, ls=ls, label=label)
    ax.fill_between(t, mean[key] - se[key], mean[key] + se[key], color=color, alpha=0.18, lw=0)


def style(ax, title, ylabel):
    ax.set_xscale("log")
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# (a) first moment against fixed targets
ax = axes[0, 0]
band(ax, "D1_inf", "#9a9890", r"vs $\mathbb{1}/d_A$")
band(ax, "D1_U1", C[2], r"vs $\rho_A^{U(1)}$ (charge only)")
band(ax, "D1_DE", C[3], r"vs $\rho_A^{\rm DE}$ (diagonal ensemble)")
ax.axhline(floor["D1_U1"], color=C[2], lw=1.1, ls="--", label=r"U(1)-random floor vs $\rho_A^{U(1)}$")
style(ax, r"(a) First moment $\Delta^{(1)}$", r"$\frac{1}{2}\|\rho_A(t)-\rho_{\rm target}\|_1$")
ax.legend(frameon=False, fontsize=8.3, loc="upper right")

# (b) vs GS_DE, (c) vs GS_U1, (d) vs Haar
panels = [(axes[0, 1], "gsDE", r"(b) vs $\mathrm{GS}_{\rm DE}=\sum_m p_m\,\mathrm{Scrooge}[\rho_m^{\rm DE}]$"),
          (axes[0, 2], "gsU1", r"(c) vs $\mathrm{GS}_{U(1)}=\sum_m p_m\,\mathrm{Scrooge}[\rho_m^{U(1)}]$"),
          (axes[1, 0], "haar", r"(d) vs Haar $k$-design (reference)")]
for ax, suf, ttl in panels:
    for k in (2, 3, 4):
        band(ax, f"D{k}_{suf}", C[k], f"$k={k}$")
        fkey = f"D{k}_gsU1" if suf == "gsDE" else f"D{k}_{suf}"
        ax.axhline(floor[fkey], color=C[k], lw=1.1, ls="--")
    style(ax, ttl, r"$\Delta^{(k)}$")
    lt = "dashed: U(1)-random floor" + (" (vs GS$_{U(1)}$)" if suf == "gsDE" else "")
    ax.legend(frameon=False, fontsize=8.3, loc="upper right", title=lt, title_fontsize=8)

# (e) frame potential vs the targets' own values
ax = axes[1, 1]
for k in (2, 3, 4):
    band(ax, f"F{k}", C[k], f"$k={k}$")
    ax.axhline(late[f"FgsDE{k}"].mean(), color=C[k], lw=1.1, ls="-.", alpha=0.9)
    ax.axhline(late[f"FgsU1{k}"].mean(), color=C[k], lw=1.1, ls=":", alpha=0.9)
ax.set_yscale("log")
style(ax, r"(e) Frame potential $F^{(k)}/F^{(k)}_{\rm Haar}$", r"$F^{(k)}/F^{(k)}_{\rm Haar}$")
ax.legend(frameon=False, fontsize=8.3, loc="upper right",
          title=r"dash-dot: GS$_{\rm DE}$, dotted: GS$_{U(1)}$", title_fontsize=8)

# (f) late-time summary
ax = axes[1, 2]
ks = np.array([2, 3, 4]); w = 0.2
series = [("vs GS$_{\\rm DE}$", lambda k: late[f"D{k}_gsDE"], "#eb6834"),
          ("vs GS$_{U(1)}$", lambda k: late[f"D{k}_gsU1"], "#2a78d6"),
          ("U(1)-random floor", lambda k: fl[f"D{k}_gsU1"], "#9a9890"),
          ("GS$_{\\rm DE}$ vs GS$_{U(1)}$", lambda k: late[f"T{k}_DEvsU1"], "#1baf7a")]
for i, (lab, f, col) in enumerate(series):
    vals = np.array([f(k).mean() for k in ks])
    errs = np.array([f(k).std() / np.sqrt(len(f(k))) for k in ks])
    ax.bar(ks + (i - 1.5) * w, vals, w * 0.92, yerr=errs, color=col, label=lab,
           error_kw=dict(ecolor=MUTED, lw=1))
ax.set_xticks(ks, [f"$k={k}$" for k in ks])
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.set_title(r"(f) Late time ($Jt\geq 50$)", loc="left", color=INK)
ax.set_ylabel(r"$\Delta^{(k)}$")
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.3, ncol=2, loc="upper left")

import re
_mm = re.search(r"_L(\d+)_LA(\d+)_", raw)
Ltxt = rf"$L={_mm.group(1)}$, $L_A={_mm.group(2)}$" if _mm else ""
fig.suptitle(rf"Projected ensemble vs fixed late-time U(1) targets, {title}:  {Ltxt}, "
             rf"$z$-basis, {nreal} random product states (mean $\pm$ s.e.)",
             x=0.01, ha="left", color=INK, fontsize=11.5)
fig.savefig(out, dpi=160)
print("saved", out)
