"""Generate the synthetic ocean-monitoring testbed and model inputs."""
import numpy as np
from . import channel as ch
from .physics_prior import prior_logits

DT_MIN = 10.0          # reporting epoch (minutes)
WINDOW = 16            # history length (epochs) fed to sequence models
N_MC = 32              # Monte-Carlo draws used to compute ground-truth success probabilities

SEQ_NAMES = ["beacon_snr", "beacon_lost", "hs_imu", "tp_imu", "tilt_imu", "heave_imu", "elev"]
CTX_NAMES = ["dist", "h_ant", "hs_imu", "tp_imu", "tilt_imu", "elev_now", "visible",
             "tod_sin", "tod_cos", "next_pass"]


def _ar1(n, rho, sigma, rng):
    x = np.zeros(n)
    s = sigma * np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + s * rng.standard_normal()
    return x


def simulate_deployment(seed=0, n_buoys=20, days=60, hs_mean=1.2, storm_rate=0.08,
                        storm_peak=(2.5, 3.5), dist_km=(5, 32), h_gw=15.0,
                        imu_noise=0.08, beacon_noise=1.0, sat_coll=0.08, lora_load=0.02,
                        tilt_amp=2.0, tide_amp=1.5, prior_misspec=False):
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / DT_MIN)
    hs_reg, tp_reg = ch.sea_state_series(n, DT_MIN / 60, rng, hs_mean, storm_rate, storm_peak)
    out = []
    for b in range(n_buoys):
        d_km = rng.uniform(*dist_km)
        h0 = rng.uniform(0.8, 1.5)
        hs = np.clip(hs_reg * (1 + 0.08 * rng.standard_normal()) *
                     np.exp(_ar1(n, 0.99, 0.05, rng)), 0.1, 10)
        tp = tp_reg * (1 + 0.03 * rng.standard_normal())
        tod = (np.arange(n) * DT_MIN / 60.0) % 24
        # semidiurnal tide (M2, 12.42 h) changes the gateway height above the sea surface
        hgw = h_gw + tide_amp * np.sin(2 * np.pi * np.arange(n) * DT_MIN / 60.0 / 12.42 + rng.uniform(0, 2 * np.pi))
        # slowly varying propagation states
        shadow = _ar1(n, 0.97, 3.0, rng)                         # dB
        duct_drive = _ar1(n, 0.995, 1.0, rng) + 0.6 * np.cos(2 * np.pi * (tod - 15) / 24) - 0.6 * (hs - 1)
        duct = 6.0 / (1 + np.exp(-2.0 * (duct_drive - 0.8)))      # dB of extra gain (0..6)
        elev = ch.leo_passes(n, DT_MIN, rng)
        # time to next pass (epochs, capped)
        nxt = np.full(n, 30.0)
        cnt = 30.0
        for t in range(n - 1, -1, -1):
            cnt = 0.0 if elev[t] > 0 else min(cnt + 1, 30.0)
            nxt[t] = cnt
        # IMU-derived sea-state estimates (what the buoy can actually measure)
        hs_imu = hs * (1 + imu_noise * rng.standard_normal(n))
        tp_imu = tp * (1 + imu_noise * rng.standard_normal(n))
        _, _, tilt_rms = ch.buoy_motion(hs, tp, rng, tilt_amp)
        tilt_imu = tilt_rms * (1 + imu_noise * rng.standard_normal(n))
        k_db = np.clip(10 - 2.0 * hs, 1.0, 10.0)                  # Rician K falls with roughness

        # ---- beacon at the start of each epoch (observed) ----
        heave_b, tilt_b, _ = ch.buoy_motion(hs, tp, rng, tilt_amp)
        snr_b = (ch.lora_snr_mean(d_km * 1e3, h0 + heave_b, hgw, tilt_b, sigma_h=hs / 4)
                 + shadow + duct - ch.wave_blockage_db(hs, h0, d_km, rng)
                 + ch.rician_db(k_db, rng, n) + beacon_noise * rng.standard_normal(n))
        lost = snr_b < ch.SNR_TH[2] - 1.0                           # beacon sent at SF9
        snr_obs = np.where(lost, -25.0, snr_b)
        heave_imu = heave_b + 0.05 * rng.standard_normal(n)

        # ---- uplink at mid-epoch: Monte-Carlo over the unknown wave phase / fading ----
        hsM, tpM = np.repeat(hs[None], N_MC, 0), np.repeat(tp[None], N_MC, 0)
        heave_u, tilt_u, _ = ch.buoy_motion(hsM, tpM, rng, tilt_amp)
        snr_u = (ch.lora_snr_mean(d_km * 1e3, h0 + heave_u, hgw[None], tilt_u, sigma_h=hsM / 4)
                 + shadow[None] + duct[None] - ch.wave_blockage_db(hsM, h0, d_km, rng)
                 + ch.rician_db(np.repeat(k_db[None], N_MC, 0), rng, (N_MC, n)))
        toa = np.array([ch.lora_toa(s) for s in ch.SF_LIST])
        p_coll = 1 - np.exp(-2 * lora_load * toa)
        succ = np.zeros((N_MC, n, ch.N_OPT), dtype=bool)
        for k in range(6):
            ok = snr_u + 0.5 * rng.standard_normal(snr_u.shape) >= ch.SNR_TH[k]
            succ[:, :, k] = ok & (rng.random(snr_u.shape) > p_coll[k])
        vis = elev > 0
        marg = (ch.sat_margin_mean(np.repeat(elev[None], N_MC, 0), tilt_u)
                + ch.rician_db(np.repeat(np.clip(8 - 0.8 * hs, 2, 8)[None], N_MC, 0), rng, (N_MC, n)))
        sat_ok = (marg >= 0) & (rng.random((N_MC, n)) > ch.doppler_fail_prob(np.repeat(elev[None], N_MC, 0))) \
            & (rng.random((N_MC, n)) > sat_coll)
        succ[:, :, 6] = sat_ok & vis[None]
        p_true = succ.mean(0)
        y = succ[0]

        seq = np.stack([snr_obs / 10.0, lost.astype(float), hs_imu, tp_imu / 10.0, tilt_imu / 10.0,
                        heave_imu, elev / 90.0], -1)
        ctx = np.stack([np.full(n, d_km / 40.0), np.full(n, h0), hs_imu, tp_imu / 10.0, tilt_imu / 10.0,
                        elev / 90.0, vis.astype(float), np.sin(2 * np.pi * tod / 24),
                        np.cos(2 * np.pi * tod / 24), nxt / 30.0], -1)
        # physics-informed prior (deterministic; no random draws, so the channel realisation is unchanged)
        plog, exp_snr = prior_logits(d_km, h0, hgw, np.maximum(hs_imu, 0.05), np.maximum(tilt_imu, 0.0), elev,
                                     misspec=prior_misspec)
        out.append(dict(seq=seq, ctx=ctx, y=y, p=p_true, vis=vis, hs_true=hs, hs_imu=hs_imu,
                        d_km=d_km, h0=h0, elev=elev, snr_b=snr_b, prior=plog, exp_snr=exp_snr))
    return out


def stratum(hs_imu):
    """Sea-state stratum used for Mondrian calibration: 0 calm, 1 moderate, 2 rough."""
    return np.digitize(hs_imu, [1.0, 2.0])


def build_arrays(dep, t0_day, t1_day):
    """Windows for epochs in [t0_day, t1_day). Returns dict of stacked arrays."""
    per_day = int(24 * 60 / DT_MIN)
    a, b = max(int(t0_day * per_day), WINDOW), int(t1_day * per_day)
    S, X, Y, P, V, H, B, T, PR = [], [], [], [], [], [], [], [], []
    for bi, d in enumerate(dep):
        idx = np.arange(a, min(b, len(d["y"])))
        win = idx[:, None] + np.arange(-WINDOW + 1, 1)[None]
        S.append(d["seq"][win]); X.append(d["ctx"][idx]); Y.append(d["y"][idx])
        P.append(d["p"][idx]); V.append(d["vis"][idx]); H.append(d["hs_imu"][idx])
        B.append(np.full(idx.size, bi)); T.append(idx)
        PR.append(np.concatenate([d["prior"][idx], d["exp_snr"][idx, None] / 10.0], 1))
    cat = lambda z: np.concatenate(z, 0)
    return dict(seq=cat(S).astype(np.float32), ctx=cat(X).astype(np.float32), y=cat(Y).astype(np.float32),
                p=cat(P).astype(np.float32), vis=cat(V), hs=cat(H), buoy=cat(B), t=cat(T),
                strat=stratum(cat(H)), prior=cat(PR).astype(np.float32))
