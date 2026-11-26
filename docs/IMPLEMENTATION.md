# Implementation details

This document maps every part of the paper to the code, lists all hyper-parameters, and explains the data flow. Equation numbers refer to the manuscript (`paper/manuscript.pdf`).

## 1. Data flow

```
channel.py ──► dataset.simulate_deployment() ──► dataset.build_arrays() ──► models.fit() ──► calib.*
 (physics)        per buoy: sea state, motion,       windows (W=16), context,     WARDEN-Net,        temperature scaling,
                  LoRa/LEO links, labels, prior      labels, prior, strata         baselines          Mondrian LTT, ACI, WARDEN-U
                         ▲
                physics_prior.py (closed-form prior, Eqs. 28-29)
```

Each experiment script (`scripts/run_*.py`) generates its own deployments from a seed, so results are reproducible bit-for-bit on the same NumPy version.

## 2. Simulator — `warden/channel.py`, `warden/dataset.py`

| Paper | Code | Notes |
|---|---|---|
| Sea state $H_s, T_p$ (Sec. 3.2) | `channel.sea_state_series` | log-AR(1), coefficient 0.995 per epoch; Poisson storms with a Gaussian envelope |
| Heave, Eq. (1) | `channel.buoy_motion` | $\zeta\sim\mathcal N(0,(H_s/4)^2)$ |
| Tilt, Eq. (2) | `channel.buoy_motion` | hull factor `amp` = $\kappa_b$ (2.0; 3.0 in OOD-B) |
| Tide $h_g(t)$ (Sec. 3.3) | `dataset.simulate_deployment` (`hgw`) | M2 tide, 12.42 h, amplitude 1.5 m, random phase |
| Two-ray phase, Eq. (3) | `channel.lora_snr_mean` | `dphi` |
| Ament roughness, Eq. (4) | `channel.lora_snr_mean` | `rough` |
| Path loss with horizon, Eq. (5) | `channel.lora_snr_mean` | `beta` = 1.2 dB/km beyond $d_h$, $k=4/3$ |
| LoRa SNR, Eq. (6) | `channel.lora_snr_mean` + `dataset` | shadowing AR(1) ($\rho=0.97$, 3 dB), ducting 0–6 dB, blockage `channel.wave_blockage_db`, Rician `channel.rician_db` |
| Success and collisions, Eq. (7) | `dataset.simulate_deployment` | thresholds `channel.SNR_TH`; ALOHA collisions $1-e^{-2GT_k}$ |
| LEO passes and slant range, Eq. (8) | `channel.leo_passes`, `channel.slant_range` | 550 km, mean revisit 95 min |
| LR-FHSS margin, Eq. (9) | `channel.sat_margin_mean`, `channel.doppler_fail_prob` | 22 dBm, −137 dBm sensitivity |
| Energy and time on air, Eqs. (10)–(11) | `channel.lora_toa`, `channel.ENERGY` | checked against Semtech values in `tests/` |
| Ground-truth probabilities | `dataset.simulate_deployment` | 32 Monte-Carlo draws (`N_MC`) per epoch |
| Strata $b(x)$ | `dataset.stratum` | $\hat H_s$ < 1 m, 1–2 m, ≥ 2 m |
| Windows and features (Sec. 4.1) | `dataset.build_arrays` | `SEQ_NAMES` (7 features × 16 steps), `CTX_NAMES` (10) |

## 3. Physics-informed prior — `warden/physics_prior.py`

| Paper | Code |
|---|---|
| Eq. (28), wave-averaged LoRa prior | `lora_prior` (10-point Gauss–Hermite, `sigma_r` = 4.5 dB, blockage 10.5 dB) |
| Eq. (29), satellite prior | `sat_prior` |
| Prior logits and expected SNR (8 features) | `prior_logits` |
| Misspecified prior (Sec. 7.4) | `MISSPEC`, `prior_logits(..., misspec=True)` |

The prior uses only quantities a buoy knows: distance, antenna and mast heights, tide-table height, IMU sea state, and ephemeris. It draws no random numbers, so adding it leaves the simulated channel unchanged.

## 4. Models — `warden/autograd.py`, `warden/models.py`

`autograd.py` is a small reverse-mode autodiff engine on NumPy: elementwise ops, matmul, softmax, layer norm, cumsum, masked BCE, and Adam. Every model is gradient-checked in `tests/test_autograd.py`.

| Paper | Code |
|---|---|
| Temporal encoder, Eq. (15) | `WardenNet.__call__` (`we`, `pos`, `sa`, FFN, two LayerNorms) |
| Multi-head attention | `MHA` (2 heads, $d=32$) |
| Sea-state-conditioned cross attention, Eq. (16) | `WardenNet.ca`, query = context encoding `cz` |
| Gated fusion, Eq. (17) | `wg`, `bg` |
| SF-monotone ordinal head, Eq. (18) | base logit + `softplus(...).cumsum()` |
| Masked BCE, Eq. (19) | `autograd.bce_logits` + mask from visibility |
| Physics-residual head, Eq. (30) | `WardenNet(prior=True)`: `beta_sf`, `beta_sat` |
| Baselines | `TransformerNet`, `GRUNet`; scikit-learn LogReg / HGB / MLP in the scripts |
| Training | `models.fit`: Adam, lr 2e-3 decayed ×0.7 per epoch, weight decay 1e-4, batch 512, ≤ 8 epochs, early stopping (patience 2) |

Parameter counts: WARDEN-Net 17 229 (17 487 with the residual head); Transformer 10 791; GRU 5 447.

## 5. Uncertainty and certification — `warden/calib.py`

| Paper | Code |
|---|---|
| Temperature scaling (Sec. 4.3) | `fit_temperature` (one temperature per action, bounded search 0.2–5) |
| Ensemble mean/std, Eq. (20) | experiment scripts (`M = 3` members) |
| Penalised score, Eq. (21) | `score(...)` in scripts, $\kappa = 1$ |
| Selective rule, Eq. (14) | `choose` (cheapest action with $s\ge\lambda$; −1 = defer) |
| Loss, Eq. (12), and risk curve, Eq. (22) | `outcome`, `loss_curve` over `LAMBDA_GRID` (141 values, 0.999 → 0.30) |
| Binomial p-value and fixed-sequence LTT, Eqs. (23)–(24) | `ltt_lambda` |
| Mondrian (per-stratum) calibration, Prop. 1 | `calibrate(..., stratified=True)` |
| Adaptive conformal update, Eqs. (26)–(27) | `online_aci` (γ = 0.005) |
| Urgency-aware cells, Eqs. (31)–(33) | `online_two_tier` (guard g = 3, $A_{\max}$ = 18 epochs) |
| Store-and-forward metrics (RDR, latency, energy per reading) | `buffer_metrics` |
| ADR baseline, oracle | `adr_baseline`, `oracle` |
| AUROC, Brier, ECE | `auroc`, `pred_metrics`, `ece` (15 bins) |

## 6. Experiments and outputs

| Script | Produces | Paper |
|---|---|---|
| `scripts/run_experiments.py --seed s` | `results/seed{s}.json` (+ `seed1_arrays.npz`, `attn_seed{s}.npy`) | Tables 4, 5, 8, 9; Figs. 3–5, 8–11 |
| `scripts/run_v2.py --seed s` | `results/v2_seed{s}.json` | Tables 6, 7, 10; Fig. 6 |
| `scripts/run_lowdata.py --seed s` | `results/low_seed{s}.json` | Fig. 7, Sec. 6.6 |
| `scripts/make_figures.py`, `make_figures_v2.py`, `show_v2.py` | `figures/*`, `results/tables*.md` | all tables and figures |
| `scripts/fig_system.py` | Figs. 1–2 | system model |

Data splits (days): train 0–36 (every second window), validation 36–42, calibration 42–48, test 48–60. OOD-A uses seed 100+s with a severe-storm climate. OOD-B uses seed 200+s with a 10 m mast, hull factor 3, load 0.05 and DtS collision probability 0.15.

## 7. Compute

On 2 CPU cores, one seed takes about 15 min for `run_experiments.py`, 15 min for `run_v2.py` and 10 min for `run_lowdata.py`, with a peak of about 3 GB RAM. No GPU is needed.

## 8. Deploying on a real buoy (outline)

1. Log the beacon SNR, IMU wave statistics and pass schedule. During commissioning, send periodic probe frames on all spreading factors to obtain labels.
2. Train WARDEN-PI offline, fit temperatures, and run `calibrate` on a held-out block of days.
3. Export the network weights (NumPy arrays), the per-stratum risk curves and `LAMBDA_GRID` to the gateway or the buoy MCU.
4. At run time, apply Algorithm 2: compute the prior, run the network, look up λ from the current level α(u,b), transmit or defer, and update α from the receipt bitmap.
