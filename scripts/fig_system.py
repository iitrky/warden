"""System-model figures (Fig. 1 network view, Fig. 2 algorithm pipeline)."""
import os, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

INK, SEA, LORA, SAT, MUTED = "#1f2a36", "#2f6f9f", "#d1495b", "#3a7d44", "#6b7785"


def box(ax, x, y, w, h, txt, fc="#ffffff", ec=INK, fs=8.5, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, p, q, c=INK, ls="-", lw=1.3, rad=0.0, head=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=c, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(0, 12.4); ax.set_ylim(-0.9, 7.4); ax.axis("off")
    # sea surface
    x = np.linspace(0, 12, 600)
    y = 1.6 + 0.12 * np.sin(2 * np.pi * x / 0.9) + 0.06 * np.sin(2 * np.pi * x / 0.37)
    ax.fill_between(x, 0, y, color="#dbe9f4")
    ax.plot(x, y, color=SEA, lw=1.0)
    ax.text(0.15, 0.3, "Open sea: significant wave height $H_s$, peak period $T_p$, tide $\\eta_{tide}(t)$",
            fontsize=8, color=SEA)
    # coast + gateway mast
    ax.add_patch(Polygon([[10.2, 0], [12, 0], [12, 2.3], [10.6, 2.1], [10.2, 1.6]], fc="#cfc6b2", ec=MUTED))
    ax.plot([11.2, 11.2], [2.2, 4.3], color=INK, lw=2)
    ax.add_patch(Polygon([[11.05, 4.3], [11.35, 4.3], [11.2, 4.75]], fc=LORA, ec=INK))
    box(ax, 10.0, 4.85, 2.3, 0.62, "LoRa gateway\n(mast height $h_g$)", fs=7.5)
    box(ax, 10.0, 5.75, 2.3, 0.62, "Network server\n+ cloud analytics", fs=7.5, fc="#f2f4f6")
    arrow(ax, (11.15, 5.47), (11.15, 5.75), lw=1.0)
    # buoys
    bx = [1.4, 3.6, 5.8, 8.0]
    for i, b in enumerate(bx):
        yy = 1.6 + 0.12 * np.sin(2 * np.pi * b / 0.9)
        tilt = [-8, 6, -4, 9][i]
        t = np.radians(tilt)
        ax.add_patch(Circle((b, yy), 0.17, fc="#f4a259", ec=INK, lw=0.8, zorder=4))
        ax.plot([b, b + 0.75 * np.sin(t)], [yy + 0.1, yy + 0.1 + 0.75 * np.cos(t)], color=INK, lw=1.4, zorder=4)
        ax.text(b, yy - 0.42, f"Buoy {i+1}", ha="center", fontsize=7.5)
    ax.text(2.5, 0.62, "heave $\\zeta(t)$, tilt $\\theta(t)$\nfrom on-board IMU", ha="center", fontsize=7, color=MUTED)
    # LoRa links
    for i, b in enumerate(bx[1:], 1):
        ls = "-" if i >= 2 else "--"
        arrow(ax, (b + 0.1, 2.55), (11.0, 4.4), c=LORA, ls=ls, lw=1.2, rad=-0.08)
    ax.text(8.6, 2.0, "LoRa uplink, SF7–SF12\n(two-ray sea reflection,\nwave blockage)", color=LORA, fontsize=7.2,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    arrow(ax, (10.9, 4.6), (4.0, 2.75), c=MUTED, ls=":", lw=1.0, rad=0.12)
    ax.text(5.6, 4.62, "beacon + receipt bitmap (downlink)", color=MUTED, fontsize=7, rotation=10)
    # satellite
    ax.add_patch(FancyBboxPatch((2.2, 6.2), 0.7, 0.45, boxstyle="round,pad=0.02", fc="#cfe3d2", ec=INK))
    ax.plot([1.6, 2.2], [6.42, 6.42], color=INK, lw=3); ax.plot([2.9, 3.5], [6.42, 6.42], color=INK, lw=3)
    ax.text(2.55, 6.95, "LEO satellite (550 km), elevation $\\epsilon(t)$", ha="center", fontsize=7.5)
    ax.annotate("", xy=(6.2, 6.8), xytext=(3.7, 6.5), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.text(4.9, 6.85, "orbit", fontsize=7, color=MUTED)
    arrow(ax, (1.45, 2.5), (2.35, 6.15), c=SAT, lw=1.3)
    arrow(ax, (8.05, 2.5), (2.95, 6.15), c=SAT, ls="--", lw=1.0)
    ax.text(0.05, 4.4, "LR-FHSS\ndirect-to-\nsatellite\nuplink", color=SAT, fontsize=7.5)
    box(ax, 6.5, 5.95, 2.6, 0.62, "Satellite ground station", fs=7, fc="#f2f4f6")
    arrow(ax, (3.55, 6.25), (6.5, 6.25), c=SAT, lw=1.0)
    arrow(ax, (9.1, 6.25), (10.0, 6.1), lw=1.0)
    # decision legend
    box(ax, 1.6, -0.85, 9.2, 0.62,
        "Per epoch each buoy picks  {SF7,…,SF12, SAT, DEFER}  to minimise energy\n"
        "subject to a certified packet-loss risk  $\\leq \\alpha$  (WARDEN)", fs=7.5, fc="#fff7ec", ec="#f4a259")
    fig.savefig(os.path.join(OUT, "fig1_system_network.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1_system_network.pdf"), bbox_inches="tight")
    plt.close(fig)


DOT = r'''
digraph G {
  graph [rankdir=TB, fontname="DejaVu Sans", fontsize=12, nodesep=0.3, ranksep=0.3, pad=0.1, newrank=true];
  node  [shape=box, style="rounded,filled", fillcolor="white", color="#1f2a36", fontname="DejaVu Sans", fontsize=11];
  edge  [color="#1f2a36", arrowsize=0.6, fontname="DejaVu Sans"];

  subgraph cluster_in { label="Inputs at epoch t (on buoy)"; color="#9aa5b1"; style="rounded";
    s1 [label="Beacon SNR history\nx(t-W+1..t)"];
    s2 [label="IMU sea-state\nHs, Tp, tilt rms, heave"];
    s3 [label="Ephemeris\nelevation, next pass"];
    s4 [label="Tide table, distance,\nmast and antenna heights"]; }

  pp [label="Physics-informed wave prior\nGauss-Hermite over heave\n(Section 4.4)", fillcolor="#e7f1fb"];

  subgraph cluster_net { label="WARDEN-Net predictor (Section 4.2)"; color="#9aa5b1"; style="rounded";
    e1 [label="Embedding +\npositional code"];
    e2 [label="Temporal multi-head\nself-attention + FFN"];
    c1 [label="Context encoder\nc(t)"];
    e3 [label="Sea-state-conditioned\ncross-attention\n(query = c(t))"];
    e4 [label="Gated fusion"];
    h1 [label="SF-monotone\nordinal head\np7 ≤ … ≤ p12"];
    h2 [label="Satellite head\npSAT"];
    rh [label="Physics-residual head\nz + softplus(β)·logit(π)", fillcolor="#e7f1fb"]; }

  subgraph cluster_unc { label="Uncertainty (Section 4.3)"; color="#9aa5b1"; style="rounded";
    u1 [label="Deep ensemble (M)\n+ temperature scaling"];
    u2 [label="Lower-confidence score\ns = μ − κσ"]; }

  subgraph cluster_rc { label="Risk-controlled selection (Sections 4.3, 4.5)"; color="#9aa5b1"; style="rounded";
    r1 [label="Mondrian Learn-then-Test\nλ_b per sea-state stratum\n(offline, calibration split)", fillcolor="#fff7ec"];
    r2 [label="Selective rule:\ncheapest option with s ≥ λ(u,b)\nelse DEFER", fillcolor="#fff7ec"];
    r3 [label="Adaptive conformal\nupdate of α_(u,b)(t)\n(online, receipt bitmap)", fillcolor="#fff7ec"];
    s5 [label="Buffer age\n(oldest reading)"];
    r4 [label="Urgency flag u(t)\n(Section 4.5)", fillcolor="#fff7ec"]; }

  out [label="Transmit SFk / LR-FHSS\nor buffer reading", fillcolor="#f2f4f6"];

  s1 -> e1 -> e2 -> e3; s2 -> e1; s2 -> c1; s3 -> c1; c1 -> e3; e3 -> e4; c1 -> e4;
  s2 -> pp; s3 -> pp; s4 -> pp; pp -> c1; pp -> rh; s5 -> r4; r4 -> r2;
  e4 -> h1; e4 -> h2; h1 -> rh; h2 -> rh; rh -> u1; u1 -> u2 -> r2; r1 -> r2; r2 -> out;
  out -> r3 [style=dashed, label=" ACK / loss ", fontsize=8]; r3 -> r2 [style=dashed];
}
'''


def fig2():
    p = os.path.join(OUT, "fig2_warden_pipeline.dot")
    open(p, "w").write(DOT)
    for ext in ["png", "pdf"]:
        args = ["dot", f"-T{ext}", p, "-o", os.path.join(OUT, f"fig2_warden_pipeline.{ext}")]
        if ext == "png":
            args.insert(1, "-Gdpi=300")
        subprocess.run(args, check=True)


if __name__ == "__main__":
    fig1(); fig2(); print("ok")
