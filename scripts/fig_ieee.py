"""Column-width figures for the IEEE IoT Journal version (paper_ieee/figs).

Reads the same result files as make_figures.py / make_figures_v2.py; no number is recomputed.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper_ieee", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42})
COLW = 3.45
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "seed[0-9].json")))]
V2 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "v2_seed[0-9].json")))]
LOW = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "low_seed[0-9].json")))]
A = np.load(os.path.join(RES, "seed1_arrays.npz"))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def m1(sec, key, field):
    v = np.array([r[sec][key][field] for r in runs if key in r[sec]])
    return v.mean(), v.std()


def m2(rs, sec, k, f):
    v = [r[sec][k][f] if sec else r[k][f] for r in rs]
    return np.mean(v), np.std(v)


# ---------------------------------------------------------------- network model
INK, SEA, LORA, SAT, MUTED = "#1f2a36", "#2f6f9f", "#c0392b", "#2e7d32", "#5f6b78"


def box(ax, x, y, w, h, txt, fc="#ffffff", ec=INK, fs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, p, q, c=INK, ls="-", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=c, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def network():
    fig, ax = plt.subplots(figsize=(COLW, 2.45))
    ax.set_xlim(0, 10); ax.set_ylim(-0.15, 7.1); ax.axis("off"); ax.grid(False)
    x = np.linspace(0, 8.6, 500)
    y = 1.5 + 0.13 * np.sin(2 * np.pi * x / 0.8)
    ax.fill_between(x, 0, y, color="#dbe9f4"); ax.plot(x, y, color=SEA, lw=0.7)
    ax.text(0.1, 0.2, "sea state $H_s,T_p$, M2 tide; heave $\\zeta$, tilt $\\theta$ from IMU", fontsize=6, color=SEA)
    ax.add_patch(Polygon([[8.2, 0], [10, 0], [10, 2.1], [8.6, 1.9]], fc="#cfc6b2", ec=MUTED, lw=0.6))
    ax.plot([9.2, 9.2], [2.0, 4.0], color=INK, lw=1.4)
    ax.add_patch(Polygon([[9.05, 4.0], [9.35, 4.0], [9.2, 4.45]], fc=LORA, ec=INK, lw=0.6))
    box(ax, 7.45, 4.6, 2.5, 0.8, "LoRa gateway\nmast $h_g(t)$", fs=6)
    box(ax, 7.45, 5.9, 2.5, 0.75, "Network server", fc="#f2f4f6", fs=6)
    arrow(ax, (8.7, 5.4), (8.7, 5.9))
    bx = [1.0, 3.0, 5.0, 7.0]
    for i, b in enumerate(bx):
        yy = 1.5 + 0.13 * np.sin(2 * np.pi * b / 0.8); t = np.radians([-9, 7, -5, 10][i])
        ax.add_patch(Circle((b, yy), 0.2, fc="#f4a259", ec=INK, lw=0.6, zorder=4))
        ax.plot([b, b + 0.8 * np.sin(t)], [yy + 0.1, yy + 0.1 + 0.8 * np.cos(t)], color=INK, lw=1.0, zorder=4)
        ax.text(b + 0.32, yy - 0.5, f"B{i+1}", ha="left", fontsize=6)
    for i, b in enumerate(bx[1:], 1):
        arrow(ax, (b + 0.15, 2.55), (9.0, 4.15), c=LORA, ls="-" if i >= 2 else "--", rad=-0.08)
    ax.text(6.05, 2.05, "LoRa\nSF7–12", color=LORA, fontsize=6, ha="center", va="center")
    arrow(ax, (8.9, 4.35), (3.4, 2.75), c=MUTED, ls=":", lw=0.8, rad=0.15)
    ax.text(2.2, 3.45, "beacon + receipts", color=MUTED, fontsize=6, rotation=16)
    ax.add_patch(FancyBboxPatch((1.25, 6.05), 0.7, 0.45, boxstyle="round,pad=0.02", fc="#cfe3d2", ec=INK, lw=0.6))
    ax.plot([0.7, 1.25], [6.27, 6.27], color=INK, lw=2); ax.plot([1.95, 2.5], [6.27, 6.27], color=INK, lw=2)
    ax.text(2.7, 6.55, "LEO pass, elevation $\\varepsilon(t)$", fontsize=6.3)
    arrow(ax, (1.0, 2.45), (1.45, 6.0), c=SAT)
    arrow(ax, (7.05, 2.5), (1.95, 6.0), c=SAT, ls="--", lw=0.7)
    ax.text(1.45, 4.6, "LR-FHSS\nDtS", color=SAT, fontsize=6.3)
    box(ax, 4.4, 5.55, 2.4, 0.7, "Ground station", fc="#f2f4f6", fs=6)
    arrow(ax, (2.55, 6.1), (4.4, 5.95), c=SAT)
    arrow(ax, (6.8, 5.95), (7.45, 6.2))
    save(fig, "fig_network")


# ---------------------------------------------------------------- main result bars (full width)
def results_bars():
    pols = [("ADR-10dB", "policy", "ADR, 10 dB", "#7f7f7f"), ("WARDEN", "policy", "WARDEN", "#9ec5e8"),
            ("WARDEN-PI", "policy", "WARDEN-PI", "#1f5f99"), ("0.05", "tier", "WARDEN-PI-U", "#f4a259")]
    scen = ["test", "ood", "site"]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.75))
    for j, (key, sec, lab, col) in enumerate(pols):
        for metric, a in [("risk", ax[0]), ("rdr", ax[1]), ("energy_per_reading_mJ", ax[2])]:
            mu = [m2(V2, sec, f"{key}@{s}", metric)[0] for s in scen]
            sd = [m2(V2, sec, f"{key}@{s}", metric)[1] for s in scen]
            a.bar(np.arange(3) + (j - 1.5) * 0.2, mu, 0.2, yerr=sd, color=col, label=lab, capsize=1.5,
                  error_kw=dict(lw=0.6))
    for a, t in zip(ax, ["attempt-failure risk", "reading delivery ratio", "energy per reading (mJ)"]):
        a.set_xticks(range(3)); a.set_xticklabels(["ID", "OOD-A storms", "OOD-B site"]); a.set_title(t)
    ax[0].axhline(0.05, color="#c0392b", ls="--", lw=0.8); ax[1].set_ylim(0.7, 0.95); ax[2].set_yscale("log")
    ax[2].yaxis.set_major_locator(FixedLocator([20, 30, 50, 100, 200])); ax[2].yaxis.set_major_formatter(ScalarFormatter())
    ax[2].yaxis.set_minor_formatter(NullFormatter())
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    save(fig, "fig_results")


# ---------------------------------------------------------------- trade-off (column)
def tradeoff():
    C = {"WARDEN": "#1f5f99", "Transformer": "#c0392b", "GRU": "#2e7d32", "HGBoost": "#8a6bbe"}
    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    fig, ax = plt.subplots(1, 2, figsize=(COLW, 1.75))
    for name, col in C.items():
        r = [m1("tradeoff", f"{name}+ACI|{a}", "risk")[0] for a in alphas]
        e = [m1("tradeoff", f"{name}+ACI|{a}", "energy_per_reading_mJ")[0] for a in alphas]
        d = [m1("tradeoff", f"{name}+ACI|{a}", "rdr")[0] for a in alphas]
        lab = "Boosting" if name == "HGBoost" else name
        ax[0].plot(r, e, "o-", ms=2, color=col, label=lab); ax[1].plot(e, d, "o-", ms=2, color=col, label=lab)
    for mg, mk in [("5", "s"), ("10", "D"), ("15", "^")]:
        ax[0].plot(m1("policy", f"ADR-{mg}dB", "risk")[0], m1("policy", f"ADR-{mg}dB", "energy_per_reading_mJ")[0], mk, color="k", ms=3)
        ax[1].plot(m1("policy", f"ADR-{mg}dB", "energy_per_reading_mJ")[0], m1("policy", f"ADR-{mg}dB", "rdr")[0], mk, color="k", ms=3,
                   label=f"ADR {mg} dB")
    ax[0].set_xlabel("realised risk"); ax[0].set_ylabel("energy/reading (mJ)")
    ax[1].set_xlabel("energy/reading (mJ)"); ax[1].set_ylabel("delivery ratio")
    ax[0].set_yscale("log"); ax[1].set_xscale("log")
    for a_ in (ax[0].yaxis, ax[1].xaxis):
        a_.set_major_locator(FixedLocator([20, 50, 100, 200])); a_.set_major_formatter(ScalarFormatter()); a_.set_minor_formatter(NullFormatter())
    ax[1].legend(fontsize=5.2, ncol=1, loc="lower right", handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_tradeoff")


# ---------------------------------------------------------------- low data (column)
def lowdata():
    days = [2, 4, 8]
    fig, ax = plt.subplots(1, 2, figsize=(COLW, 1.75))
    for name, col, lab in [("WARDEN", "#9ec5e8", "WARDEN"), ("WARDEN-PI", "#1f5f99", "WARDEN-PI"),
                           ("WARDEN-PI-U", "#f4a259", "WARDEN-PI-U"), ("HGBoost", "#8a6bbe", "Boosting"),
                           ("Physics prior only", "#2e7d32", "Prior only")]:
        r = [m2(LOW, None, f"{name}|{d}", "rdr") for d in days]
        e = [m2(LOW, None, f"{name}|{d}", "energy_per_reading_mJ") for d in days]
        ax[0].errorbar(days, [x[0] for x in r], yerr=[x[1] for x in r], marker="o", ms=2, capsize=1.5, color=col, label=lab, lw=0.9)
        ax[1].errorbar(days, [x[0] for x in e], yerr=[x[1] for x in e], marker="o", ms=2, capsize=1.5, color=col, label=lab, lw=0.9)
    ax[0].axhline(np.mean([r["policy"]["ADR-10dB@test"]["rdr"] for r in V2]), color="#555", ls=":", lw=0.9, label="ADR, 10 dB")
    for a in ax:
        a.set_xticks(days); a.set_xlabel("training days")
    ax[0].set_ylabel("delivery ratio"); ax[1].set_ylabel("energy/reading (mJ)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, 1.13),
               handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_lowdata")


# ---------------------------------------------------------------- online risk (column)
def online():
    def rolling(choice, Y, tt, w=144):
        order = np.argsort(tt, kind="stable"); c, Yo, ts = choice[order], Y[order], tt[order]
        err = np.where(c >= 0, 1 - Yo[np.arange(len(c)), np.clip(c, 0, None)], 0)
        days = np.unique(ts); per_t = np.array([err[ts == u].mean() for u in days])
        return days / 144, np.convolve(per_t, np.ones(w) / w, mode="valid"), w
    fig, ax = plt.subplots(figsize=(COLW, 1.45))
    for key, lab, col in [("site_choice_s", "Mondrian LTT (offline)", "#1f5f99"), ("site_choice", "with online update", "#f4a259")]:
        d, r, w = rolling(A[key], A["site_y"], A["site_t"]); ax.plot(d[w - 1:], r, color=col, lw=0.9, label=lab)
    ax.axhline(runs[0]["alpha"], color="#c0392b", ls="--", lw=0.8, label="target $\\alpha$")
    ax.set_xlabel("time (days)"); ax.set_ylabel("24-h rolling risk"); ax.legend(ncol=3, fontsize=5.5, loc="upper center",
                                                                             bbox_to_anchor=(0.5, 1.2), frameon=False)
    save(fig, "fig_online")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["network", "results_bars", "tradeoff", "lowdata", "online"]
    for f in todo:
        globals()[f]()
    print("ok", todo)
