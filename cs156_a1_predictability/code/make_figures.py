"""Figures for the Results section.  Usage: python3 -I make_figures.py RESULTS_DIR OUT_DIR"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
A = json.load(open(os.path.join(RES, "results.json")))
K = json.load(open(os.path.join(RES, "kn_results.json")))

EARLY, RECENT = "#2a78d6", "#eb6834"          # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

def pp(bits): return 2 ** bits

def models(label):
    a, k = A[label], K[label]
    out = {}
    for per, key in (("early", "E->E"), ("recent", "R->R")):
        bi_a = a["table"][key]["mean_bits"]; bi_k = k["table"][key]["mean_bits"]
        uni = bi_k + k["bigram_gain"][per]                # unigram from the Kneser-Ney run (wider smoothing grid)
        out[per] = [pp(uni), pp(bi_a), pp(bi_k)]
    return out

# ---------- Figure 1: unigram vs add-alpha bigram vs Kneser-Ney, by period
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
names = ["Unigram", "Add-α\nbigram", "Kneser–Ney\nbigram"]
for ax, (label, title) in zip(axes, (("main", "All words"), ("function_words", "Function words only"))):
    m = models(label)
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, m["early"], w, color=EARLY, label="Early essays (2023–24)", zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, m["recent"], w, color=RECENT, label="Recent essays (2025–26)", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():,.0f}" if label == "main" else f"{r.get_height():.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("Perplexity (lower = more predictable)" if label == "main" else "")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_ylim(0, max(m["early"] + m["recent"]) * 1.12)
    ax.spines["left"].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig1_models.{ext}"), dpi=300)
plt.close(fig)

# ---------- Figure 2: per-essay bits per word, Kneser-Ney, own-period (held-out essays)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
rng = np.random.default_rng(156)
for ax, (label, title) in zip(axes, (("main", "All words"), ("function_words", "Function words only"))):
    s = K[label]; E = s["per_essay_bits"]["E->E"]; R = s["per_essay_bits"]["R->R"]
    for xi, (vals, col) in enumerate(((E, EARLY), (R, RECENT))):
        v = np.array(list(vals.values()))
        jit = (rng.random(len(v)) - 0.5) * 0.22
        ax.scatter(xi + jit, v, s=34, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax.hlines(v.mean(), xi - 0.28, xi + 0.28, color=INK, lw=1.8, zorder=4)
        ax.text(xi + 0.32, v.mean(), f"mean {v.mean():.2f}", va="center", fontsize=8, color=INK)
    d = s["own_period_bigram"]
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.text(0.0, -0.16, f"recent − early = {d['delta_bits']:+.2f} bits per word, exact p = {d['p']:.4f}".replace("-0", "−0"),
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Early (n={len(E)})", f"Recent (n={len(R)})"])
    ax.set_xlim(-0.5, 2.15)
    ax.set_ylabel("Bits per word (lower = more predictable)" if label == "main" else "")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0); ax.spines["left"].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig2_essays.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("done")
