"""Time evolution of projected-ensemble distances for one model.

    python3 plot_time_evolution.py results/raw_L10_LA2_n100_z.csv XXZ_easyaxis
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw = sys.argv[1] if len(sys.argv) > 1 else "results/raw_L10_LA2_n100_z.csv"
model = sys.argv[2] if len(sys.argv) > 2 else "XXZ_easyaxis"
out = sys.argv[3] if len(sys.argv) > 3 else f"results/time_evolution_{model}.png"

df = pd.read_csv(raw)
m = df[(df.model == model) & (df.t > 0)]
haar = df[df.model == "Haar_global_state"]
Delta, J2 = m.Delta.iloc[0], m.J2.iloc[0]
nreal = m.realization.nunique()

g = m.groupby("t")
mean, se = g.mean(numeric_only=True), g.std(numeric_only=True) / np.sqrt(nreal)
t = mean.index.values
floor = haar.mean(numeric_only=True)

C = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)


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


# (a) first moment
ax = axes[0, 0]
band(ax, "D1", C[2], r"vs $\mathbb{1}/d_A$ (infinite temperature)")
band(ax, "D1_DE", C[3], r"vs diagonal ensemble $\rho_A^{\rm DE}$")
ax.axhline(mean["DE_inf"].iloc[-1], color=MUTED, lw=1.2, ls="--",
           label=r"$\frac{1}{2}\|\rho_A^{\rm DE}-\mathbb{1}/d_A\|_1$ (finite-$L$ offset)")
ax.axhline(floor["D1"], color=MUTED, lw=1.2, ls=":", label="Haar global state")
style(ax, r"(a) First moment  $\Delta^{(1)}$", r"$\frac{1}{2}\|\rho_A(t)-\rho_{\rm target}\|_1$")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# (b)-(d) higher moments against the three targets
targets = [("haar", "(b) vs Haar $k$-design"),
           ("scr", r"(c) vs Scrooge$[\rho_A(t)]$"),
           ("gs", r"(d) vs generalized Scrooge (cond. on $Q_B$)")]
for ax, (suf, title) in zip([axes[0, 1], axes[0, 2], axes[1, 0]], targets):
    for k in (2, 3, 4):
        band(ax, f"D{k}_{suf}", C[k], f"$k={k}$")
        ax.axhline(floor[f"D{k}_{suf}"], color=C[k], lw=1.1, ls="--", alpha=0.9)
    style(ax, title, r"$\Delta^{(k)}$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", title="dashed: Haar global state",
              title_fontsize=8)

# (e) frame potential ratio
ax = axes[1, 1]
for k in (2, 3, 4):
    band(ax, f"F{k}", C[k], f"$k={k}$")
    ax.axhline(floor[f"F{k}"], color=C[k], lw=1.1, ls="--", alpha=0.9)
ax.axhline(1, color=MUTED, lw=1)
ax.set_yscale("log")
style(ax, r"(e) Frame potential  $F^{(k)}/F^{(k)}_{\rm Haar}$", r"$F^{(k)}/F^{(k)}_{\rm Haar}$")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# (f) excess over the Haar-global-state floor, late times
ax = axes[1, 2]
late = m[m.t >= 50]
per_real = late.groupby("realization").mean(numeric_only=True)
ks = np.array([2, 3, 4])
w = 0.26
for i, (suf, lab, col) in enumerate([("haar", "Haar", "#2a78d6"), ("scr", "Scrooge", "#eb6834"),
                                      ("gs", "gen. Scrooge", "#1baf7a")]):
    vals = np.array([per_real[f"D{k}_{suf}"].mean() - floor[f"D{k}_{suf}"] for k in ks])
    errs = np.array([per_real[f"D{k}_{suf}"].std() / np.sqrt(nreal) for k in ks])
    ax.bar(ks + (i - 1) * w, vals, w * 0.92, yerr=errs, color=col, label=lab,
           error_kw=dict(ecolor=MUTED, lw=1))
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(ks, [f"$k={k}$" for k in ks])
ax.set_title(r"(f) Late-time excess over Haar-state floor ($Jt\geq 50$)", loc="left", color=INK)
ax.set_ylabel(r"$\overline{\Delta^{(k)}} - \Delta^{(k)}_{\rm Haar\,state}$")
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center")

import re
_mm = re.search(r"_L(\d+)_LA(\d+)_", raw)
Ltxt = rf"$L={_mm.group(1)}$, $L_A={_mm.group(2)}$" if _mm else ""
fig.suptitle(rf"Projected ensemble, XXZ $(1+\Delta)={1+Delta:g}$, $J_2={J2:g}$:  "
             rf"{Ltxt}, $z$-basis measurement, {nreal} random product states (mean $\pm$ s.e.)",
             x=0.01, ha="left", color=INK, fontsize=11.5)
fig.savefig(out, dpi=160)
print("saved", out)
