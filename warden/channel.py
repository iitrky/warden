"""Physics-based simulation testbed for hybrid LoRa / LEO-satellite ocean IoT.

Every quantity here is synthetic. The models are deliberately simple, documented
approximations (two-ray over sea, radio-horizon diffraction penalty, wave-driven
buoy heave and tilt, Rician fading, LEO pass geometry). They are meant to give a
reproducible, non-stationary benchmark, not to replace sea-trial measurements.
"""
import numpy as np

C = 3e8
R_EARTH = 6371e3
K_FACTOR = 4.0 / 3.0
G = 9.81

# LoRa demodulation SNR thresholds (dB) for SF7..SF12 (Semtech datasheet values)
SF_LIST = np.array([7, 8, 9, 10, 11, 12])
SNR_TH = np.array([-7.5, -10.0, -12.5, -15.0, -17.5, -20.0])


def lora_toa(sf, payload=20, bw=125e3, cr=1, preamble=8, header=True):
    """LoRa time-on-air in seconds (standard Semtech formula)."""
    t_sym = (2.0 ** sf) / bw
    de = 1 if (sf >= 11 and bw == 125e3) else 0
    ih = 0 if header else 1
    num = 8 * payload - 4 * sf + 28 + 16 - 20 * ih
    n_payload = 8 + max(np.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * t_sym + n_payload * t_sym


# Electrical power drawn by the radio while transmitting (W), SX126x-class values
P_ELEC_14DBM = 0.15   # ~45 mA at 3.3 V
P_ELEC_22DBM = 0.39   # ~118 mA at 3.3 V
LRFHSS_TOA = 2.0      # s, DR8-class LR-FHSS frame (20 B) incl. header replicas

ENERGY = np.array([P_ELEC_14DBM * lora_toa(sf) for sf in SF_LIST] +
                  [P_ELEC_22DBM * LRFHSS_TOA])          # J per attempt
OPTION_NAMES = [f"SF{s}" for s in SF_LIST] + ["SAT"]
N_OPT = len(OPTION_NAMES)


# ----------------------------------------------------------------------------
# Sea state
# ----------------------------------------------------------------------------
def sea_state_series(n, dt_h, rng, hs_mean=1.2, storm_rate_per_day=0.08,
                     storm_peak=(2.5, 3.5), ar=0.995):
    """Significant wave height Hs (m) and peak period Tp (s) time series.

    log-Hs follows an AR(1) process; storms are added as Poisson events with a
    smooth rise/decay envelope.
    """
    lh = np.zeros(n)
    sig = 0.06
    lh[0] = np.log(hs_mean)
    for t in range(1, n):
        lh[t] = np.log(hs_mean) + ar * (lh[t - 1] - np.log(hs_mean)) + sig * rng.standard_normal()
    hs = np.exp(lh)
    t_days = np.arange(n) * dt_h / 24.0
    n_storms = rng.poisson(storm_rate_per_day * t_days[-1])
    for _ in range(n_storms):
        c = rng.uniform(0, t_days[-1])
        w = rng.uniform(0.4, 1.2)          # days
        peak = rng.uniform(*storm_peak)
        env = np.exp(-0.5 * ((t_days - c) / (w / 2.355)) ** 2)
        hs = np.maximum(hs, hs + (peak - hs_mean) * env)
    hs = np.clip(hs, 0.15, 9.0)
    tp = 4.6 * np.sqrt(hs) + 0.6 * rng.standard_normal(n)
    tp = np.clip(tp, 2.5, 16.0)
    return hs, tp


def buoy_motion(hs, tp, rng, amp=2.0):
    """Instantaneous heave (m) and tilt (deg) of a small spar/disc buoy."""
    heave = (hs / 4.0) * rng.standard_normal(hs.shape)
    wavelength = G * tp ** 2 / (2 * np.pi)
    slope_rms = np.pi * hs / wavelength / np.sqrt(2)
    tilt_rms = np.degrees(amp * slope_rms)
    tilt = np.abs(tilt_rms * rng.standard_normal(hs.shape))
    return heave, tilt, tilt_rms


# ----------------------------------------------------------------------------
# Terrestrial LoRa link (buoy -> platform/shore gateway)
# ----------------------------------------------------------------------------
def lora_snr_mean(d_m, h_buoy, h_gw, tilt_deg, f=868e6, pt=14.0, g_gw=6.0,
                  nf=6.0, bw=125e3, gamma=-0.95, sigma_h=0.0, beta=1.2e-3):
    """Mean (large-scale) LoRa SNR in dB over sea.

    Two-ray with sea reflection (|Gamma| close to 1 at grazing incidence),
    Ament roughness factor, a radio-horizon diffraction penalty, and a
    monopole tilt loss at the buoy.
    """
    lam = C / f
    h_b = np.maximum(h_buoy, 0.15)
    fspl = 20 * np.log10(4 * np.pi * d_m / lam)
    dphi = 4 * np.pi * h_gw * h_b / (lam * d_m)
    psi = (h_gw + h_b) / d_m
    rough = np.exp(-8 * (np.pi * sigma_h * np.sin(psi) / lam) ** 2)
    two_ray = np.abs(1 + gamma * rough * np.exp(-1j * dphi))
    pl = fspl - 20 * np.log10(np.maximum(two_ray, 1e-4))
    d_h = np.sqrt(2 * K_FACTOR * R_EARTH) * (np.sqrt(h_gw) + np.sqrt(h_b))
    pl = pl + np.where(d_m > d_h, beta * (d_m - d_h), 0.0)   # beta = 1.2 dB/km beyond horizon
    g_b = 2.15 + 10 * np.log10(np.maximum(np.cos(np.radians(tilt_deg)) ** 2, 0.05))
    noise = -174 + 10 * np.log10(bw) + nf
    return pt + g_b + g_gw - pl - noise


def wave_blockage_db(hs, h0, d_km, rng):
    """Extra loss when wave crests screen a low antenna (Huo et al. style effect)."""
    p_block = 1 / (1 + np.exp(-(hs / 2 - h0) / 0.25)) * np.clip(d_km / 30, 0.2, 1.0)
    blocked = rng.random(hs.shape) < p_block
    return blocked * rng.uniform(6, 15, hs.shape)


def rician_db(k_db, rng, shape):
    k = 10 ** (k_db / 10)
    s = np.sqrt(k / (k + 1))
    sig = np.sqrt(1 / (2 * (k + 1)))
    h = (s + sig * rng.standard_normal(shape)) + 1j * sig * rng.standard_normal(shape)
    return 20 * np.log10(np.abs(h))


# ----------------------------------------------------------------------------
# LEO satellite (direct-to-satellite LR-FHSS)
# ----------------------------------------------------------------------------
def leo_passes(n, dt_min, rng, mean_gap_min=95.0, alt=550e3):
    """Elevation (deg) seen by a buoy for each epoch; 0 when no satellite."""
    elev = np.zeros(n)
    t = rng.exponential(mean_gap_min)
    horizon = n * dt_min
    while t < horizon:
        e_max = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(12)), 1.0)))
        dur = 4.0 + 8.0 * (e_max / 90.0)             # minutes above 10 deg
        dur *= 1.6                                    # epochs are coarse; widen window
        idx = np.arange(int(t // dt_min), int((t + dur) // dt_min) + 1)
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.size:
            ph = (idx * dt_min - t) / dur
            e = 10 + (e_max - 10) * np.sin(np.pi * np.clip(ph, 0, 1))
            elev[idx] = np.maximum(elev[idx], e)
        t += dur + rng.exponential(mean_gap_min)
    return elev


def slant_range(elev_deg, alt=550e3):
    e = np.radians(elev_deg)
    return np.sqrt((R_EARTH + alt) ** 2 - (R_EARTH * np.cos(e)) ** 2) - R_EARTH * np.sin(e)


def sat_margin_mean(elev_deg, tilt_deg, f=868e6, pt=22.0, g_sat=2.0, sens=-137.0):
    lam = C / f
    r = slant_range(np.maximum(elev_deg, 1.0))
    fspl = 20 * np.log10(4 * np.pi * r / lam)
    e_eff = np.clip(elev_deg - tilt_deg, 0, 90)
    g_b = 3.0 - 4.5e-4 * (90 - e_eff) ** 2          # hemispherical QHA-like pattern
    rx = pt + g_b + g_sat - fspl - 2.0               # 2 dB polarisation/atmosphere
    return rx - sens


def doppler_fail_prob(elev_deg):
    """Packet failure probability due to Doppler rate at low elevation."""
    return 0.25 * np.exp(-(elev_deg - 10) / 8.0)
