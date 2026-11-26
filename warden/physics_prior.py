"""Physics-informed wave prior (WARDEN-PI).

Closed-form, sensor-driven estimate of each action's success probability. It
uses only quantities a buoy knows or measures: distance, antenna and mast
heights (with tide-table correction), IMU sea state and satellite ephemeris.
The heave is integrated out with Gauss-Hermite quadrature over
zeta ~ N(0, (Hs/4)^2); fading, shadowing and ducting, which the buoy cannot
compute, are folded into a residual spread sigma_r, and the network learns
the remainder from the beacon history.
"""
import numpy as np
from scipy.stats import norm
from . import channel as ch

GH_NODES, GH_W = np.polynomial.hermite_e.hermegauss(10)
GH_W = GH_W / GH_W.sum()


def lora_prior(d_km, h0, hgw, hs_hat, tilt_rms_hat, sigma_r=4.5, block_db=10.5, gamma=-0.95, beta=1.2e-3):
    zeta = (hs_hat / 4.0)[:, None] * GH_NODES[None]                       # (n, K)
    tilt = (tilt_rms_hat * np.sqrt(2 / np.pi))[:, None]                    # mean of |N(0, rms^2)|
    snr = ch.lora_snr_mean(d_km * 1e3, h0 + zeta, np.asarray(hgw)[:, None], tilt,
                           sigma_h=(hs_hat / 4.0)[:, None], gamma=gamma, beta=beta)  # (n, K)
    p_block = 1 / (1 + np.exp(-(hs_hat / 2 - h0) / 0.25)) * np.clip(d_km / 30, 0.2, 1.0)
    P = np.zeros((len(hs_hat), 6))
    for k, g in enumerate(ch.SNR_TH):
        clear = norm.cdf((snr - g) / sigma_r) @ GH_W
        blocked = norm.cdf((snr - block_db - g) / sigma_r) @ GH_W
        P[:, k] = (1 - p_block) * clear + p_block * blocked
    exp_snr = snr @ GH_W
    return P, exp_snr


def sat_prior(elev, tilt_rms_hat, sigma_s=3.0, coll=0.08, sens=-137.0):
    vis = elev > 0
    tilt = tilt_rms_hat * np.sqrt(2 / np.pi)
    m = ch.sat_margin_mean(np.maximum(elev, 1.0), tilt, sens=sens)
    p = norm.cdf(m / sigma_s) * (1 - ch.doppler_fail_prob(np.maximum(elev, 10.0))) * (1 - coll)
    return np.where(vis, p, 0.0)


# Deliberately misspecified physics (robustness study): weaker sea reflection,
# optimistic diffraction slope, +3 m mast-height error, narrower residual spread,
# underestimated blockage and a 2 dB optimistic satellite receiver.
MISSPEC = dict(gamma=-0.80, beta=0.8e-3, hgw_bias=3.0, sigma_r=3.0, block_db=6.0, sens=-139.0)


def prior_logits(d_km, h0, hgw, hs_hat, tilt_rms_hat, elev, misspec=False):
    if misspec:
        m = MISSPEC
        P, exp_snr = lora_prior(d_km, h0, np.asarray(hgw) + m["hgw_bias"], hs_hat, tilt_rms_hat,
                                sigma_r=m["sigma_r"], block_db=m["block_db"], gamma=m["gamma"], beta=m["beta"])
        ps = sat_prior(elev, tilt_rms_hat, sens=m["sens"])
    else:
        P, exp_snr = lora_prior(d_km, h0, hgw, hs_hat, tilt_rms_hat)
        ps = sat_prior(elev, tilt_rms_hat)
    P = np.clip(np.concatenate([P, ps[:, None]], 1), 1e-4, 1 - 1e-4)
    return np.log(P / (1 - P)), exp_snr
