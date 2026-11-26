
# 5 Baselines and experimental setup

## 5.1 Sandbox and test bed

All experiments run in a software sandbox test bed that implements the models of Section 3 in Python (NumPy, SciPy, scikit-learn) with a purpose-written automatic-differentiation engine, so that every number in this paper can be regenerated on a laptop without a GPU. The test bed is synthetic: it is a physics-guided emulator, not a sea trial, and its parameters (Table 3) were chosen from published offshore and DtS studies [5, 8, 15–19]. Each deployment contains $N=20$ buoys with distances drawn uniformly in 5–32 km and antenna heights in 0.8–1.5 m, observed for 60 days at 10-min epochs (8 640 epochs per buoy). The regional sea state follows a log-AR(1) process with Poisson storm events; each buoy adds a local perturbation. For every epoch the ground-truth success probability of every action is computed from 32 Monte Carlo draws of wave phase, blockage and fading, and one draw provides the observed labels.

\needspace{16\baselineskip}

**Table 3** Test-bed parameters

| Parameter | Value |
|------------------------------|----------------------------------------------------------------------|
| Carrier, bandwidth, noise figure | 868 MHz, 125 kHz, 6 dB |
| LoRa $P_t$, gateway gain, mast height | 14 dBm, 6 dBi, 15 m (10 m in OOD-B) |
| Tide | M2, amplitude 1.5 m |
| Diffraction slope beyond horizon $\beta$ | 1.2 dB/km |
| Shadowing | AR(1), $\rho=0.97$ per epoch, $\sigma=3$ dB |
| Ducting gain | 0–6 dB, logistic in an AR(1) drive with diurnal and sea-state terms |
| Rician K (LoRa / satellite) | $\max(1,10-2H_s)$ dB / $\max(2,8-0.8H_s)$ dB |
| SF thresholds $\gamma_7..\gamma_{12}$ | −7.5, −10, −12.5, −15, −17.5, −20 dB |
| Background load $G$ | 0.02 frames/s (0.05 in OOD-B) |
| LEO altitude, mean revisit | 550 km, 95 min, max elevation 12°–90° |
| LR-FHSS power, sensitivity, frame | 22 dBm, −137 dBm, 2.0 s |
| DtS collision probability | 0.08 (0.15 in OOD-B) |
| Radio power draw (14 / 22 dBm) | 0.15 W / 0.39 W |
| Payload, epoch, buffer age limit | 20 bytes, 10 min, 18 epochs (3 h) |
| Hull response $\kappa_b$, IMU noise | 2.0 (3.0 in OOD-B), 8 % |
| Sea state, in-distribution | mean $H_s$ 1.2 m, 0.08 storms/day, storm peak 2.5–3.5 m |
| Sea state, OOD-A | mean $H_s$ 1.8 m, 0.15 storms/day, storm peak 4–7 m |

**Splits.** Splits are chronological to respect temporal dependence: days 0–36 for training (every second epoch used, 51 680 windows), 36–42 for validation and temperature fitting (17 280), 42–48 for calibration (17 280) and 48–60 for testing (34 560). **OOD-A** is a separate deployment with new buoys and severe storms. **OOD-B** is a site and hardware shift: lower mast, livelier hull, heavier interference and a shorter distance range. Two robustness sets reuse the in-distribution channel realisation with 30 % IMU noise or 3 dB beacon noise. Every experiment is repeated with three independent seeds (1, 2, 3), which change the deployment, the network initialisation and the calibration draw; tables report mean ± standard deviation.

## 5.2 Baselines

*Rule-based policies.* (i) **ADR-$m$ dB**: an adaptive-data-rate style rule that picks the smallest SF whose threshold plus a margin $m\in\{5,10,15\}$ dB lies below an exponentially weighted beacon SNR, otherwise the satellite if visible, otherwise SF12; (ii) **Always-SF12**; (iii) **Sat-first**: satellite whenever visible, otherwise SF12; (iv) **Oracle**: the cheapest action that actually succeeds (unattainable lower bound).

*Predictors* (all followed by the same temperature scaling and selection rule): logistic regression, histogram gradient boosting and a two-layer MLP on 42 hand-crafted features (last four window steps, context, SNR mean, minimum, spread and beacon-loss rate); a GRU sequence model; and a Transformer encoder with mean pooling and concatenated context, representative of attention channel predictors [20–22]. All neural models use width $d=32$ and Adam with learning rate $2\times10^{-3}$, batch 512, up to 8 epochs and early stopping on validation loss.

*Physics-informed baselines.* (v) **Physics prior only**: the closed-form prior (28)–(29) with temperature scaling and the same certification, i.e. no learning at all; (vi) **gradient boosting + prior**: histogram gradient boosting given the prior logits as extra features, which tests whether the gain of WARDEN-PI comes from the prior or from the attention residual design. *Prior misspecification.* To test robustness to a wrong physics model, WARDEN-PI is also trained and calibrated with a deliberately biased prior (reflection coefficient −0.80 instead of −0.95, diffraction slope 0.8 instead of 1.2 dB/km, a +3 m mast-height error, a 3 dB instead of 4.5 dB residual spread, 6 dB instead of 10.5 dB blockage loss, and a 2 dB optimistic satellite receiver).

*Selection rules* (with the WARDEN ensemble): a naive threshold of 0.5, the nominal threshold $1-\alpha$, global (unstratified) LTT, Mondrian LTT, Mondrian LTT with the online update (26), and the urgency-aware two-tier rule (31)–(32) with $\alpha_U\in\{0.05,0.10,0.20,0.30\}$ and guard $g=3$.

## 5.3 Metrics

Prediction quality is measured by the mean AUROC over actions, the Brier score and the expected calibration error (ECE, 15 bins). Decision quality is measured by the realised attempt-failure risk overall and per stratum, the deferral rate, the reading delivery ratio (RDR, the fraction of readings that reach the gateway within the 3-h buffer limit), the mean latency of delivered readings and the energy per delivered reading. Unless stated otherwise $\alpha=0.05$ and $\delta=0.1$.

