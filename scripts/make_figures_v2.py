"""Figures and tables for the WARDEN-PI / WARDEN-U extensions (results/v2_seed*.json, low_seed*.json)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
V2 = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "v2_seed[0-9].json")))]
LOW = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "low_seed[0-9].json")))]
print("v2 seeds", [r["seed"] for r in V2], "low seeds", [r["seed"] for r in LOW])
ms = lambda runs, sec, k, f: (np.mean([r[sec][k][f] if sec else r[k][f] for r in runs]), np.std([r[sec][k][f] if sec else r[k][f] for r in runs]))

pols = [("ADR-10dB", "policy", "ADR, 10 dB", "#555555"), ("WARDEN", "policy", "WARDEN", "#9ec5e8"),
        ("WARDEN-PI", "policy", "WARDEN-PI", "#1f5f99"), ("0.05", "tier", "WARDEN-PI-U", "#f4a259")]
scen = [("test", "in-distribution"), ("ood", "OOD-A storms"), ("site", "OOD-B site shift")]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
for j, (key, sec, lab, col) in enumerate(pols):
    for m, (metric, a) in enumerate([("risk", ax[0]), ("rdr", ax[1]), ("energy_per_reading_mJ", ax[2])]):
        mu = [ms(V2, sec, f"{key}@{s}", metric)[0] for s, _ in scen]
        sd = [ms(V2, sec, f"{key}@{s}", metric)[1] for s, _ in scen]
        a.bar(np.arange(3) + (j - 1.5) * 0.2, mu, 0.2, yerr=sd, color=col, label=lab, capsize=2)
for a, t in zip(ax, ["attempt-failure risk", "reading delivery ratio", "energy per reading (mJ)"]):
    a.set_xticks(range(3)); a.set_xticklabels(["ID", "OOD-A", "OOD-B"]); a.set_title(t, fontsize=8.5)
ax[0].axhline(0.05, color="#d1495b", ls="--", lw=1); ax[1].set_ylim(0.7, 0.95); ax[2].set_yscale("log")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "fig10_pi_u.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig10_pi_u.pdf")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
days = [2, 4, 8]
for name, col in [("WARDEN", "#9ec5e8"), ("WARDEN-PI", "#1f5f99"), ("WARDEN-PI-U", "#f4a259"), ("HGBoost", "#8a6bbe"),
                  ("Physics prior only", "#3a7d44")]:
    r = [ms(LOW, None, f"{name}|{d}", "rdr") for d in days]; e = [ms(LOW, None, f"{name}|{d}", "energy_per_reading_mJ") for d in days]
    ax[0].errorbar(days, [x[0] for x in r], yerr=[x[1] for x in r], marker="o", ms=3, capsize=2, color=col, label=name)
    ax[1].errorbar(days, [x[0] for x in e], yerr=[x[1] for x in e], marker="o", ms=3, capsize=2, color=col, label=name)
ax[0].axhline(np.mean([r["policy"]["ADR-10dB@test"]["rdr"] for r in V2]), color="#555", ls=":", lw=1, label="ADR, 10 dB")
for a in ax:
    a.set_xticks(days); a.set_xlabel("days of labelled commissioning data")
ax[0].set_ylabel("reading delivery ratio"); ax[1].set_ylabel("energy per reading (mJ)"); ax[0].legend(fontsize=6.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig11_lowdata.png"), dpi=300); fig.savefig(os.path.join(FIG, "fig11_lowdata.pdf")); plt.close(fig)

lines = ["## Low-data (mean ± std)"]
for d in days:
    for name in ["WARDEN", "WARDEN-PI", "WARDEN-PI-U", "HGBoost", "HGBoost+prior", "Physics prior only"]:
        k = f"{name}|{d}"
        lines.append(f"{k} | risk {ms(LOW,None,k,'risk')[0]:.3f} | RDR {ms(LOW,None,k,'rdr')[0]:.3f} ± {ms(LOW,None,k,'rdr')[1]:.3f} | E {ms(LOW,None,k,'energy_per_reading_mJ')[0]:.1f} ± {ms(LOW,None,k,'energy_per_reading_mJ')[1]:.1f}"
                     + (f" | Brier {ms(LOW,None,k,'pred_brier')[0]:.4f}" if "pred_brier" in LOW[0][k] else ""))
open(os.path.join(RES, "tables_lowdata.md"), "w").write("\n".join(lines)); print("\n".join(lines))
