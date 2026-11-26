
# 4 Proposed WARDEN algorithm

WARDEN (Wave-Aware Risk-controlled DEcision eNgine) has three stages, shown in Fig. 2: input representation with a physics-informed wave prior, the attention-enhanced predictor, and uncertainty-aware, urgency-aware risk-controlled selection. Sections 4.1–4.3 describe the base engine, and Sections 4.4–4.5 the two extensions WARDEN-PI and WARDEN-U; the full system is denoted WARDEN-PI-U.

![**Fig. 2** WARDEN-PI-U pipeline: inputs, physics-informed wave prior, WARDEN-Net predictor with physics-residual head, uncertainty quantification, and urgency-aware risk-controlled selection with offline Mondrian Learn-then-Test calibration and online adaptive conformal updates.](../figures/fig2_warden_pipeline.png){width=92%}

## 4.1 Preprocessing, segmentation and input representation

**Segmentation.** The stream of each buoy is cut into sliding windows of $W=16$ epochs (2 h 40 min), long enough to span the decorrelation time of shadowing and about one fifth of a tidal cycle. Windows end at the beacon of epoch $t$, so no information from the decision interval itself leaks into the input.

**Sequence features.** Each window step $\tau$ carries a seven-dimensional vector
$\mathbf{u}_\tau=[\mathrm{SNR}^{\mathrm{bcn}}_\tau/10,\;m_\tau,\;\hat H_{s,\tau},\;\hat T_{p,\tau}/10,\;\hat\theta_{\mathrm{rms},\tau}/10,\;\hat\zeta_\tau,\;\varepsilon_\tau/90]$, where $m_\tau=1$ flags a lost beacon (its SNR is then set to −25 dB). **Context features.** The current context $\mathbf{c}_t\in\mathbb{R}^{10}$ contains the known distance, antenna height, current IMU sea state, current elevation and visibility, time-of-day encoded as $(\sin,\cos)$, and the number of epochs to the next pass. **Normalisation.** All features are standardised with statistics from the training split only. **Stratum.** The stratum $b(x)$ is computed from $\hat H_s$ in the context vector.

## 4.2 Proposed attention-enhanced architecture (WARDEN-Net)

**Temporal encoder.** The window $\mathbf{U}\in\mathbb{R}^{W\times 7}$ is embedded and combined with a learned positional code $\mathbf{P}$, and one post-norm Transformer block with $H=2$ heads is applied:

$$\mathbf{Z}_0=\mathbf{U}\mathbf{W}_e+\mathbf{P},\quad \mathbf{Z}_1=\mathrm{LN}\left(\mathbf{Z}_0+\mathrm{MHA}(\mathbf{Z}_0,\mathbf{Z}_0)\right),\quad \mathbf{Z}=\mathrm{LN}\left(\mathbf{Z}_1+\mathrm{FFN}(\mathbf{Z}_1)\right), \tag{15}$$

with $\mathrm{MHA}(\mathbf{Q},\mathbf{K})=[\mathrm{softmax}(\mathbf{Q}\mathbf{W}^q_h(\mathbf{K}\mathbf{W}^k_h)^{\top}/\sqrt{d/H})\,\mathbf{K}\mathbf{W}^v_h]_{h=1}^{H}\mathbf{W}^o$.

**Sea-state-conditioned cross attention.** The context is encoded as $\mathbf{e}=\mathrm{GELU}(\mathbf{c}\mathbf{W}_{c1})\mathbf{W}_{c2}$ and used as the *query* over the encoded history:

$$\mathbf{z}=\mathrm{MHA}(\mathbf{e},\mathbf{Z})\in\mathbb{R}^{d}. \tag{16}$$

Under calm water the query can emphasise the latest beacons, which track slow shadowing and ducting; in rough water it can spread weight over the window to average out wave-induced fluctuations. Section 7 shows that the learned attention indeed differs by stratum.

**Gated fusion.** A context-dependent gate decides how much history to trust:

$$\mathbf{v}=\mathrm{GELU}\!\left(\left[\mathbf{z}\odot\sigma(\mathbf{e}\mathbf{W}_g+\mathbf{b}_g)+\mathbf{e}\right]\mathbf{W}_h\right). \tag{17}$$

**SF-monotone ordinal head.** Following the nesting noted after (7), the SF logits are built as a base logit plus cumulative non-negative increments, and a separate head models the satellite:

$$z_{7}=\mathbf{v}^{\top}\mathbf{w}_0,\qquad z_{k}=z_{7}+\sum_{j=8}^{k}\mathrm{softplus}(\mathbf{v}^{\top}\mathbf{w}_j),\;k=8,\dots,12,\qquad z_{\mathrm{S}}=\mathbf{v}^{\top}\mathbf{w}_{\mathrm{S}}. \tag{18}$$

Hence $p_7\le p_8\le\dots\le p_{12}$ by construction, which prevents physically impossible rankings that would otherwise confuse the cost-ordered rule (14).

**Training loss.** All seven outputs are trained jointly with a masked binary cross-entropy in which the satellite term is active only during passes ($v_t=1$):

$$\mathcal{L}=\frac{1}{\sum_i\sum_a\omega_{ia}}\sum_i\sum_{a}\omega_{ia}\left[\log(1+e^{z_{ia}})-Y_{ia}z_{ia}\right],\qquad \omega_{ia}=\begin{cases}v_{i},&a=\mathrm{SAT}\\1,&\text{otherwise.}\end{cases} \tag{19}$$

In the testbed all actions are labelled in every epoch because the simulator evaluates counterfactual outcomes; in a deployment the same labels are obtained from periodic probe frames during a commissioning period (Section 8).

## 4.3 Uncertainty estimation, calibration and risk-controlled selective prediction

**Ensemble and temperature scaling.** $M=3$ WARDEN-Nets with different initialisations are trained. Each member's logits are divided by an action-specific temperature fitted on the validation split, and the calibrated probabilities are averaged:

$$\mu_a(x)=\frac{1}{M}\sum_{m=1}^{M}\sigma\!\left(z^{(m)}_a(x)/T^{(m)}_a\right),\qquad \sigma_a(x)=\mathrm{std}_m\,\sigma\!\left(z^{(m)}_a(x)/T^{(m)}_a\right). \tag{20}$$

**Penalised score.** Epistemic disagreement is converted into caution with a lower-confidence score

$$s_a(x)=\mathrm{clip}\left(\mu_a(x)-\kappa\,\sigma_a(x),\,0,\,1\right),\qquad \kappa=1. \tag{21}$$

**Mondrian Learn-then-Test.** Let $\mathcal{D}^{\mathrm{cal}}_b$ be the $n_b$ calibration epochs of stratum $b$ and $\Lambda=\{\lambda_1>\lambda_2>\dots>\lambda_J\}$ a grid from 0.999 down to 0.30 ($J=141$). For each $\lambda_j$ the empirical risk of policy (14) is

$$\hat R_b(\lambda_j)=\frac{1}{n_b}\sum_{i\in\mathcal{D}^{\mathrm{cal}}_b}\ell(\pi_{\lambda_j};x_i,Y_i). \tag{22}$$

Because the loss is binary, the null hypothesis $H_{b,j}:R_b(\lambda_j)>\alpha$ is tested with the exact binomial p-value

$$p_{b,j}=\Pr\left[\mathrm{Bin}(n_b,\alpha)\le n_b\hat R_b(\lambda_j)\right]. \tag{23}$$

Hypotheses are tested in the fixed order $j=1,2,\dots$ (most conservative first), and the threshold is the last rejected one:

$$\hat\lambda_b=\lambda_{\hat j},\qquad \hat j=\max\{j:\;p_{b,1}\le\delta,\dots,p_{b,j}\le\delta\}, \tag{24}$$

with $\hat\lambda_b=1.01$ (always defer) when $p_{b,1}>\delta$.

**Proposition 1 (per-stratum risk certificate).** If the calibration and test epochs within stratum $b$ are exchangeable and the predictor is fixed before calibration, then

$$\Pr\left[R_b(\pi_{\hat\lambda_b})\le\alpha\right]\ge 1-\delta\quad\text{for every stratum } b. \tag{25}$$

*Proof sketch.* Fixed-sequence testing of $H_{b,1},H_{b,2},\dots$ at level $\delta$ controls the family-wise error rate at $\delta$ without any monotonicity assumption, because a false rejection can occur only at the first true null in the sequence [34]. The p-values (23) are valid under exchangeability since $n_b\hat R_b(\lambda_j)$ is binomial with success probability $R_b(\lambda_j)$. Stratification is a Mondrian partition: conditioning on $b$ preserves exchangeability within each cell, so the argument holds separately for each $b$. $\square$

**Online adaptive conformal update.** Exchangeability is broken by storms and site changes. During operation each stratum keeps a working level $\alpha_{b,t}$, initialised at $\alpha$, and maps it to the most aggressive threshold whose calibration risk curve stays below it in the same fixed order; the receipt bitmap then supplies the realised loss $\mathrm{err}_t$ of the previous attempt and

$$\alpha_{b,t+1}=\mathrm{clip}\left(\alpha_{b,t}+\gamma(\alpha-\mathrm{err}_t),\,10^{-4},\,0.5\right),\qquad \gamma=0.005. \tag{26}$$

Following Gibbs and Candès [35], without clipping the update guarantees, for any sequence of data,

$$\left|\frac{1}{T_b}\sum_{t=1}^{T_b}\mathrm{err}_t-\alpha\right|\le\frac{\max(\alpha_{b,1},1-\alpha_{b,1})+\gamma}{\gamma\,T_b}, \tag{27}$$

so the long-run failure frequency in each stratum converges to $\alpha$ at rate $O(1/T_b)$. Clipping at 0.5 is inactive in all our runs except the first epochs after an abrupt shift. Algorithms 1 and 2 summarise the offline and online stages; Algorithm 2 already includes the extensions of Sections 4.4 and 4.5.

**Algorithm 1** WARDEN offline training and certification

```
Input : training, validation and calibration splits; α, δ, κ, grid Λ
1  standardise features with training statistics
2  for m = 1..M: train WARDEN-Net_m with loss (19); fit temperatures T_a^(m) on validation
3  for each calibration epoch i: compute μ, σ by (20) and s by (21)
4  for each stratum b:
5      for j = 1..J (λ_1 > … > λ_J):
6          compute R̂_b(λ_j) by (22) and p_{b,j} by (23)
7          if p_{b,j} > δ: break
8      λ̂_b ← last λ_j with p_{b,j} ≤ δ (1.01 if none)
9      store the calibration risk curve R̂_b(·)
Output: ensemble, temperatures, {λ̂_b}, {R̂_b(·)}
```

**Algorithm 2** WARDEN-PI-U online operation on buoy n

```
1   α_{u,b} ← α (u = 0) or α_U (u = 1) for all cells
2   every epoch t:
3       receive beacon; read SNR and receipt bit err_{t-1}
4       update the level of the previous epoch's cell (u,b) by (32)
5       read IMU sea state, tide table, ephemeris; build window and context
6       compute physics prior π by (28)-(29)
7       b ← stratum(Ĥ_s);  u ← urgency flag by (31)
8       λ ← most aggressive λ_j with R̂_b(λ_1..λ_j) ≤ α_{u,b}
9       compute s_a for available actions by (20), (21), (30)
10      a* ← cheapest action with s_a ≥ λ, otherwise ∅           # rule (14)
11      if a* = ∅: buffer reading (drop readings older than A_max)
12      else: send buffered readings with action a*
```

## 4.4 Physics-informed wave prior and residual head (WARDEN-PI)

The buoy knows its distance $d_n$, antenna height $h_0$ and mast height $\bar h_g$, can read the tide $h_g(t)$ from tide tables, measures $\hat H_s$ and $\hat\theta_{\mathrm{rms}}$ with its IMU, and knows the satellite elevation from ephemeris. These quantities are enough to evaluate the deterministic part of (3)–(9) in closed form. Let $\overline{\mathrm{SNR}}(\zeta)$ denote the large-scale SNR (6) without the unknown shadowing, ducting and fading terms, evaluated at heave $\zeta$, mean tilt $\bar\theta=\hat\theta_{\mathrm{rms}}\sqrt{2/\pi}$ and roughness $\sigma_h=\hat H_s/4$. Integrating the heave distribution (1) with $Q$-point Gauss–Hermite quadrature (nodes $\xi_q$, normalised weights $w_q$, $Q=10$) gives the wave-averaged success prior of SF$k$:

$$\pi_k(x)=(1-p_B)\sum_{q=1}^{Q}w_q\,\Phi\!\left(\frac{\overline{\mathrm{SNR}}(\xi_q\hat H_s/4)-\gamma_k}{\sigma_r}\right)+p_B\sum_{q=1}^{Q}w_q\,\Phi\!\left(\frac{\overline{\mathrm{SNR}}(\xi_q\hat H_s/4)-L_B-\gamma_k}{\sigma_r}\right), \tag{28}$$

where $\Phi$ is the standard normal distribution function, $\sigma_r=4.5$ dB lumps the residual variability the buoy cannot compute, $L_B$ is the mean blockage loss and $p_B$ the blockage probability implied by $\hat H_s$ and $h_0$ [8]. For the satellite,

$$\pi_{\mathrm{S}}(x)=\mathbb{1}\{\varepsilon>0\}\,\Phi\!\left(\frac{M_S(\varepsilon,\bar\theta)}{\sigma_s}\right)\left(1-p_D(\varepsilon)\right)\left(1-p_c\right), \tag{29}$$

with $M_S$ from (9) without fading and $p_c$ the nominal collision probability. Because $\gamma_7>\dots>\gamma_{12}$, the prior satisfies $\pi_7\le\dots\le\pi_{12}$ automatically.

The prior enters WARDEN-Net twice. Its logits $\boldsymbol{\ell}=\mathrm{logit}(\boldsymbol\pi)$ and the expected SNR are appended to the context vector $\mathbf{c}_t$, so that the cross-attention query of (16) is informed by physics. In addition, the output logits of (18) are replaced by a *physics-residual head*

$$\tilde z_k=z_k+\mathrm{softplus}(\beta_{\mathrm{SF}})\,\ell_k,\quad k=7,\dots,12,\qquad \tilde z_{\mathrm{S}}=z_{\mathrm{S}}+\mathrm{softplus}(\beta_{\mathrm{S}})\,\ell_{\mathrm{S}}, \tag{30}$$

with learnable scalars $\beta_{\mathrm{SF}},\beta_{\mathrm{S}}$. A single shared $\beta_{\mathrm{SF}}\ge0$ preserves the SF ordering of (18), and the network only has to learn what physics leaves unexplained: shadowing, ducting, interference and modelling error. If the prior is wrong, the network can shrink $\beta$, and the certification layer still enforces the risk target (Section 7.4). Evaluating (28)–(29) costs $7Q$ link-budget evaluations, about $10^3$ operations per epoch.

## 4.5 Urgency-aware Mondrian certification (WARDEN-U)

A reading whose age approaches the buffer limit $A_{\max}$ is lost if the buoy keeps deferring. WARDEN-U therefore refines the Mondrian partition with an urgency flag

$$u_t=\mathbb{1}\{t-t^{\mathrm{old}}_{n}\ge A_{\max}-g\}, \tag{31}$$

where $t^{\mathrm{old}}_n$ is the arrival epoch of the oldest buffered reading and $g$ a guard interval ($g=3$ epochs). Each cell $(u,b)$ keeps its own working level $\alpha_{u,b,t}$ and its own target, $\alpha$ for ordinary epochs and $\alpha_U$ for urgent ones, updated as in (26):

$$\alpha_{u,b,t+1}=\mathrm{clip}\left(\alpha_{u,b,t}+\gamma\,(\alpha^{(u)}-\mathrm{err}_t),\,10^{-4},\,0.6\right),\qquad \alpha^{(0)}=\alpha,\;\alpha^{(1)}=\alpha_U. \tag{32}$$

By (27), applied to each cell, the long-run failure frequency of ordinary traffic converges to $\alpha$ and that of urgent traffic to $\alpha_U$. The operator thus states two numbers, the reliability of routine transmissions and the extra risk accepted to rescue data that would otherwise expire, and the overall failure rate is the urgency-weighted mixture

$$R=(1-\rho_U)\,R^{(0)}+\rho_U\,R^{(1)},\qquad \rho_U=\Pr[u_t=1]. \tag{33}$$

Because urgent epochs are rare, (33) stays close to $\alpha$ even when $\alpha_U$ is larger. Separating the cells also has a subtler benefit, visible even with $\alpha_U=\alpha$: the update no longer averages the very different failure statistics of fresh and nearly expired buffers, so each cell's threshold adapts to its own conditions.

**Complexity.** WARDEN-Net has 17 229 parameters (17 487 with the physics-residual head). One forward pass costs $O(W^2d+Wd^2)$ operations, of the order of $10^5$ multiply-accumulate operations for $W=16$, $d=32$, so a three-member ensemble needs well under 1 MFLOP per decision, within reach of a Cortex-M4-class microcontroller every 10 min. Online calibration costs one table look-up and one addition per epoch.

