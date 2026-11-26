"""Result figures and aggregated tables from results/seed*.json."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
C = {"WARDEN": "#1f5f99", "Transformer": "#d1495b", "GRU": "#3a7d44", "HGBoost": "#8a6bbe",
     "LogReg": "#e08e2b", "ADR": "#555555"}
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "seed[0-9].json")))]
A = np.load(os.path.join(RES, "seed1_arrays.npz"))
print("seeds:", [r["seed"] for r in runs])


def ms(sec, key, field):
    v = np.array([r[sec][key][field] for r in runs if key in r[sec]])
    return v.mean(), v.std()


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# Fig 3: testbed traces --------------------------------------------------------
n = 144 * 10
t = np.arange(n) / 144
fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True)
ax[0].plot(t, A["hs_series"][:n], color="#2f6f9f", lw=1); ax[0].set_ylabel("$H_s$ (m)")
ax[1].plot(t, A["snr_series"][:n], color="#555", lw=0.5); ax[1].set_ylabel("beacon SNR (dB)")
ax[1].axhline(-7.5, color="#d1495b", ls="--", lw=0.8); ax[1].axhline(-20, color="#d1495b", ls=":", lw=0.8)
ax[1].text(9.2, -6.5, "SF7", color="#d1495b", fontsize=7); ax[1].text(9.2, -19, "SF12", color="#d1495b", fontsize=7)
ax[2].plot(t, A["p_series"][:n, 0], lw=0.7, label="SF7", color="#d1495b")
ax[2].plot(t, A["p_series"][:n, 5], lw=0.7, label="SF12", color="#e08e2b")
ax[2].plot(t, np.where(A["elev_series"][:n] > 0, A["p_series"][:n, 6], np.nan), ".", ms=2, label="SAT", color="#3a7d44")
ax[2].set_ylabel("true $P$(success)"); ax[2].set_xlabel("time (days)"); ax[2].legend(ncol=3, fontsize=7, loc="lower left")
save(fig, "fig3_testbed_traces")

# Fig 4: reliability diagrams ---------------------------------------------------
y, vis = A["y_test"], A["vis_test"]
m = np.concatenate([np.ones((len(y), 6)), vis[:, None]], 1) > 0
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.3))
edges = np.linspace(0, 1, 13)
for key, name in [("p_test", "WARDEN"), ("p_tr", "Transformer"), ("p_gru", "GRU")]:
    p = A[key][m]; yy = y[m]
    idx = np.clip(np.digitize(p, edges) - 1, 0, 11)
    xs = [p[idx == b].mean() for b in range(12) if (idx == b).sum() > 30]
    ys = [yy[idx == b].mean() for b in range(12) if (idx == b).sum() > 30]
    ax.plot(xs, ys, "o-", ms=3, lw=1, color=C[name], label=name)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("predicted success probability"); ax.set_ylabel("observed success frequency"); ax.legend(fontsize=7)
save(fig, "fig4_reliability")

# Fig 5: risk-energy trade-off ---------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
for name in ["WARDEN", "Transformer", "GRU", "HGBoost", "LogReg"]:
    r = [ms("tradeoff", f"{name}+ACI|{a}", "risk")[0] for a in alphas]
    e = [ms("tradeoff", f"{name}+ACI|{a}", "energy_per_reading_mJ")[0] for a in alphas]
    d = [ms("tradeoff", f"{name}+ACI|{a}", "rdr")[0] for a in alphas]
    ax[0].plot(r, e, "o-", ms=3, color=C[name], label=name)
    ax[1].plot(e, d, "o-", ms=3, color=C[name], label=name)
for mg, mk in [("5", "s"), ("10", "D"), ("15", "^")]:
    if f"ADR-{mg}dB" in runs[0]["policy"]:
        ax[0].plot(ms("policy", f"ADR-{mg}dB", "risk")[0], ms("policy", f"ADR-{mg}dB", "energy_per_reading_mJ")[0], mk, color="k", ms=5)
        ax[1].plot(ms("policy", f"ADR-{mg}dB", "energy_per_reading_mJ")[0], ms("policy", f"ADR-{mg}dB", "rdr")[0], mk, color="k", ms=5, label=f"ADR {mg} dB")
ax[0].set_xlabel("realised attempt-failure risk"); ax[0].set_ylabel("energy per delivered reading (mJ)")
ax[1].set_xlabel("energy per delivered reading (mJ)"); ax[1].set_ylabel("reading delivery ratio")
ax[0].set_yscale("log"); ax[1].set_xscale("log")
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
for a_ in (ax[0].yaxis, ax[1].xaxis):
    a_.set_major_locator(FixedLocator([20, 30, 50, 100, 200])); a_.set_major_formatter(ScalarFormatter()); a_.set_minor_formatter(NullFormatter())
ax[1].legend(fontsize=6.5, ncol=2)
save(fig, "fig5_tradeoff")

# Fig 6: per-stratum risk ---------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
pols = [("WARDEN+G-LTT", "global LTT"), ("WARDEN+S-LTT", "Mondrian LTT"), ("WARDEN+S-LTT+ACI", "Mondrian LTT + ACI")]
cols = ["#9aa5b1", "#1f5f99", "#f4a259"]
for k, (sec, title) in enumerate([("policy", "in-distribution"), ("ood", "OOD-A: severe storms"), ("site", "OOD-B: site shift")]):
    for j, (p, lab) in enumerate(pols):
        mu = [ms(sec, p, f"risk_s{s}")[0] for s in range(3)]
        sd = [ms(sec, p, f"risk_s{s}")[1] for s in range(3)]
        ax[k].bar(np.arange(3) + (j - 1) * 0.26, mu, 0.26, yerr=sd, color=cols[j], label=lab, capsize=2)
    ax[k].axhline(runs[0]["alpha"], color="#d1495b", ls="--", lw=1)
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels(["calm", "moderate", "rough"]); ax[k].set_title(title, fontsize=8.5)
ax[0].set_ylabel("attempt-failure risk"); ax[0].legend(fontsize=6.5)
save(fig, "fig6_stratum_risk")

# Fig 7: online risk under site shift --------------------------------------------
def rolling(choice, Y, tt, w=144):
    order = np.argsort(tt, kind="stable")
    c, Yo = choice[order], Y[order]
    err = np.where(c >= 0, 1 - Yo[np.arange(len(c)), np.clip(c, 0, None)], 0)
    ts = tt[order]
    days = np.unique(ts)
    per_t = np.array([err[ts == u].mean() for u in days])
    k = np.ones(w) / w
    return days / 144, np.convolve(per_t, k, mode="valid"), w


fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.4))
for key, lab, col in [("site_choice_s", "Mondrian LTT (offline)", "#1f5f99"), ("site_choice", "Mondrian LTT + ACI (online)", "#f4a259")]:
    d, r, w = rolling(A[key], A["site_y"], A["site_t"])
    ax.plot(d[w - 1:], r, color=col, lw=1.2, label=lab)
ax.axhline(runs[0]["alpha"], color="#d1495b", ls="--", lw=1, label="target α")
ax.set_xlabel("time (days)"); ax.set_ylabel("24-h rolling risk"); ax.legend(fontsize=7)
save(fig, "fig7_online_site_shift")

# Fig 8: calibration-set size ------------------------------------------------------
ncal = sorted(runs[0]["ncal"], key=int)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
rm = [np.mean([r["ncal"][k]["risk_mean"] for r in runs]) for k in ncal]
rx = [np.max([r["ncal"][k]["risk_max"] for r in runs]) for k in ncal]
em = [np.mean([r["ncal"][k]["defer"] for r in runs]) for k in ncal]
x = [int(k) for k in ncal]
ax[0].plot(x, rm, "o-", label="mean over 20 draws", color="#1f5f99"); ax[0].plot(x, rx, "s--", label="worst draw", color="#555")
ax[0].axhline(runs[0]["alpha"], color="#d1495b", ls="--", lw=1); ax[0].set_xscale("log")
ax[0].set_xlabel("calibration samples $n$"); ax[0].set_ylabel("test risk"); ax[0].legend(fontsize=7)
ax[1].plot(x, em, "o-", color="#1f5f99"); ax[1].set_xscale("log")
for a_ in ax:
    a_.xaxis.set_major_locator(FixedLocator(x)); a_.xaxis.set_major_formatter(ScalarFormatter()); a_.xaxis.set_minor_formatter(NullFormatter())
ax[1].set_xlabel("calibration samples $n$"); ax[1].set_ylabel("deferral rate")
save(fig, "fig8_calibration_size")

# Fig 9: cross-attention weights over history lag by sea state -------------------
att = np.load(os.path.join(RES, "attn_seed1.npy"))
st = A["strat_test"][:len(att)]
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))
for s, lab, col in [(0, "calm", "#9ec5e8"), (1, "moderate", "#1f5f99"), (2, "rough", "#d1495b")]:
    if (st == s).sum():
        ax.plot(np.arange(-att.shape[1] + 1, 1), att[st == s].mean(0), "o-", ms=3, color=col, label=lab)
ax.set_xlabel("history lag (epochs of 10 min)"); ax.set_ylabel("mean cross-attention weight"); ax.legend(fontsize=7)
save(fig, "fig9_attention")

# ---------------------------------------------------------------------------
# Tables (mean ± std over seeds)
# ---------------------------------------------------------------------------
def f(sec, key, field, pct=False, dec=3):
    mu, sd = ms(sec, key, field)
    if pct:
        return f"{100*mu:.2f} ± {100*sd:.2f}"
    return f"{mu:.{dec}f} ± {sd:.{dec}f}"


lines = []
lines.append("## Table A: prediction quality on test (in-distribution / OOD-A / OOD-B)")
for name in ["LogReg", "HGBoost", "MLP", "GRU", "Transformer", "WARDEN-single", "WARDEN-noIMU", "WARDEN-noCross", "WARDEN-noMono", "WARDEN"]:
    row = [name]
    for suf in ["", "@OOD", "@SITE"]:
        row += [f("pred", name + suf, "auroc"), f("pred", name + suf, "brier", dec=4), f("pred", name + suf, "ece", dec=4)]
    lines.append(" | ".join(row))
for sec in ["policy", "ood", "site", "isorisk", "robust"]:
    lines.append(f"\n## Table: {sec}")
    keys = list(runs[0][sec].keys())
    for k in keys:
        if k == "alpha_iso":
            lines.append(f"alpha_iso = {np.mean([r[sec][k] for r in runs]):.4f} ± {np.std([r[sec][k] for r in runs]):.4f}")
            continue
        flds = ["risk", "risk_s0", "risk_s1", "risk_s2", "defer", "rdr", "latency_min", "energy_per_reading_mJ"]
        lines.append(k + " | " + " | ".join(f(sec, k, x, dec=(1 if x in ("latency_min", "energy_per_reading_mJ") else 3)) for x in flds))
lines.append("\n## ncal")
for k in ncal:
    lines.append(f"{k} | risk_mean {np.mean([r['ncal'][k]['risk_mean'] for r in runs]):.4f} | worst {np.max([r['ncal'][k]['risk_max'] for r in runs]):.4f} | viol {np.mean([r['ncal'][k]['viol'] for r in runs]):.3f} | E {np.mean([r['ncal'][k]['e_mean'] for r in runs]):.1f}")
open(os.path.join(RES, "tables.md"), "w").write("\n".join(lines))
print("\n".join(lines))
