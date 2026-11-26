"""Calibration, uncertainty metrics, risk-controlled selective link policy."""
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize_scalar
from . import channel as ch

LAMBDA_GRID = np.round(np.linspace(0.999, 0.30, 141), 4)   # conservative -> aggressive


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(z, y, m):
    """Per-option temperature scaling on a validation split."""
    temps = np.ones(z.shape[1])
    for k in range(z.shape[1]):
        idx = m[:, k] > 0
        zk, yk = z[idx, k], y[idx, k]

        def nll(t):
            zt = zk / t
            return np.mean(np.logaddexp(0, zt) - yk * zt)
        temps[k] = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded").x
    return temps


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return e


def auroc(p, y):
    o = np.argsort(p)
    r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pred_metrics(P, d):
    m = np.concatenate([np.ones((len(P), 6)), d["vis"][:, None]], 1) > 0
    p, y = P[m], d["y"][m]
    au = np.nanmean([auroc(P[m[:, k], k], d["y"][m[:, k], k]) for k in range(7)])
    nll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    return dict(auroc=au, brier=np.mean((p - y) ** 2), ece=ece(p, y), nll=nll)


# ----------------------------------------------------------------------------
# Selective policy
# ----------------------------------------------------------------------------
def choose(S, vis, lam):
    """Cheapest option whose score clears lam; -1 = defer (abstain)."""
    S = S.copy()
    S[:, 6] = np.where(vis, S[:, 6], -1.0)
    lam = np.broadcast_to(np.asarray(lam, float), (len(S),))
    ok = S >= lam[:, None]
    first = np.where(ok.any(1), ok.argmax(1), -1)
    return first


def outcome(choice, y):
    tx = choice >= 0
    succ = np.zeros(len(choice), bool)
    succ[tx] = y[np.where(tx)[0], choice[tx]] > 0.5
    loss = tx & ~succ
    energy = np.where(tx, ch.ENERGY[np.clip(choice, 0, None)], 0.0)
    return loss.astype(float), succ, energy


def loss_curve(S, d):
    """Empirical loss for every lambda on the grid: shape (n, n_lambda)."""
    L = np.zeros((len(S), len(LAMBDA_GRID)))
    for j, lam in enumerate(LAMBDA_GRID):
        L[:, j] = outcome(choose(S, d["vis"], lam), d["y"])[0]
    return L


def ltt_lambda(L, alpha, delta):
    """Learn-then-Test with fixed-sequence testing and exact binomial p-values.

    Tests H0_j: R(lambda_j) > alpha in order from the most conservative lambda;
    returns the last lambda for which H0 is rejected at level delta. Valid
    without monotonicity of the loss in lambda.
    """
    n = L.shape[0]
    best = 1.01
    for j in range(L.shape[1]):
        k = int(round(L[:, j].sum()))
        pval = binom.cdf(k, n, alpha)
        if pval <= delta:
            best = LAMBDA_GRID[j]
        else:
            break
    return best


def calibrate(S_cal, d_cal, alpha, delta, stratified=True):
    L = loss_curve(S_cal, d_cal)
    if not stratified:
        return {s: ltt_lambda(L, alpha, delta) for s in range(3)}, L
    return {s: ltt_lambda(L[d_cal["strat"] == s], alpha, delta) if (d_cal["strat"] == s).sum() > 50
            else ltt_lambda(L, alpha, delta) for s in range(3)}, L


def evaluate(S, d, lam_by_strat):
    lam = np.vectorize(lambda s: lam_by_strat[s])(d["strat"])
    c = choose(S, d["vis"], lam)
    return summarize(c, d)


def summarize(c, d):
    loss, succ, energy = outcome(c, d["y"])
    res = dict(risk=loss.mean(), delivery=succ.mean(), defer=(c < 0).mean(),
               sat_share=(c == 6).mean(), energy_mJ=1e3 * energy.mean(),
               energy_per_deliv_mJ=1e3 * energy.sum() / max(succ.sum(), 1))
    for s in range(3):
        m = d["strat"] == s
        res[f"risk_s{s}"] = loss[m].mean() if m.any() else np.nan
    if "buoy" in d:
        res.update(buffer_metrics(c, succ, d))
    res["choice"] = c
    return res


def buffer_metrics(c, succ, d, max_age=18, dt_min=10.0):
    """Store-and-forward accounting of individual sensor readings.

    A new reading is buffered every epoch. A successful transmission delivers
    every buffered reading (compact delta-encoded payload); a failed unconfirmed
    transmission loses them; readings older than max_age epochs are discarded.
    """
    delivered, lost_air, stale, lat, energy = 0, 0, 0, [], 0.0
    E = np.where(c >= 0, ch.ENERGY[np.clip(c, 0, None)], 0.0)
    for b in np.unique(d["buoy"]):
        idx = np.where(d["buoy"] == b)[0]
        idx = idx[np.argsort(d["t"][idx])]
        buf = []
        for i in idx:
            t = d["t"][i]
            buf.append(t)
            energy += E[i]
            if c[i] >= 0:
                if succ[i]:
                    delivered += len(buf); lat += [t - x for x in buf]
                else:
                    lost_air += len(buf)
                buf = []
            else:
                old = [x for x in buf if t - x >= max_age]
                stale += len(old)
                buf = [x for x in buf if t - x < max_age]
    total = delivered + lost_air + stale + 0.0
    return dict(rdr=delivered / max(total, 1), read_lost_air=lost_air / max(total, 1),
                read_stale=stale / max(total, 1), latency_min=float(np.mean(lat)) * dt_min if lat else 0.0,
                energy_per_reading_mJ=1e3 * energy / max(delivered, 1))


def online_aci(S, d, L_cal, strat_cal, alpha, gamma=0.005, stratified=True):
    """Adaptive conformal update of the per-stratum target level (Gibbs & Candes style).

    The calibration loss curve maps a working level a_t to the most aggressive
    lambda whose calibration risk stays <= a_t (fixed-sequence order). After each
    epoch the observed loss (ACK-based) updates a_t <- a_t + gamma (alpha - err_t).
    """
    curves = {}
    for s in range(3):
        m = (strat_cal == s) if stratified else np.ones(len(strat_cal), bool)
        if m.sum() < 50:
            m = np.ones(len(strat_cal), bool)
        curves[s] = L_cal[m].mean(0)

    def lam_for(s, a):
        r = curves[s]
        ok = r <= a
        if not ok[0]:
            return 1.01
        j = np.argmin(ok) - 1 if not ok.all() else len(r) - 1
        return LAMBDA_GRID[j]
    order = np.lexsort((d["buoy"], d["t"]))
    a = {s: alpha for s in range(3)}
    c = np.full(len(S), -1)
    Sm = S.copy(); Sm[:, 6] = np.where(d["vis"], Sm[:, 6], -1.0)
    for i in order:
        s = d["strat"][i] if stratified else 0
        lam = lam_for(d["strat"][i] if stratified else 0, a[s])
        ok = Sm[i] >= lam
        k = int(ok.argmax()) if ok.any() else -1
        c[i] = k
        err = 0.0 if k < 0 else float(d["y"][i, k] < 0.5)
        a[s] = float(np.clip(a[s] + gamma * (alpha - err), 1e-4, 0.5))
    return summarize(c, d)


def oracle(d):
    """Cheapest option that actually succeeds (unattainable lower bound)."""
    Y = d["y"].copy(); Y[:, 6] = Y[:, 6] * d["vis"]
    c = np.where(Y.any(1), Y.argmax(1), -1)
    return summarize(c, d)


def adr_baseline(d, margin_db=10.0):
    """LoRaWAN-ADR-like rule on an EWMA of beacon SNR with a fixed margin.

    Picks the lowest SF whose threshold plus margin is below the smoothed SNR;
    if none qualifies, uses the satellite when visible, otherwise SF12.
    """
    snr = d["seq"][:, :, 0] * 10.0          # dB (lost beacons = -25)
    w = 0.7 ** np.arange(snr.shape[1])[::-1]
    ew = (snr * w).sum(1) / w.sum()
    ok = (ew[:, None] - margin_db) >= ch.SNR_TH[None]
    c = np.where(ok.any(1), ok.argmax(1), np.where(d["vis"], 6, 5))
    return summarize(c, d)


def online_two_tier(S, d, L_cal, strat_cal, alpha, alpha_u, gamma=0.005, max_age=18, guard=3):
    """Urgency-aware two-tier certification (WARDEN-U).

    Epochs whose oldest buffered reading is within `guard` epochs of expiry
    form an 'urgent' Mondrian cell with its own target alpha_u; all other
    epochs keep target alpha. Each (tier, sea-state) cell runs its own
    adaptive conformal level, so the long-run failure rate of each tier
    converges to its own target.
    """
    curves = {}
    for s in range(3):
        m = strat_cal == s
        curves[s] = L_cal[m].mean(0) if m.sum() >= 50 else L_cal.mean(0)

    def lam_for(s, a):
        r = curves[s]
        ok = r <= a
        if not ok[0]:
            return 1.01
        j = np.argmin(ok) - 1 if not ok.all() else len(r) - 1
        return LAMBDA_GRID[j]
    order = np.lexsort((d["buoy"], d["t"]))
    lv = {(u, s): (alpha_u if u else alpha) for u in (0, 1) for s in range(3)}
    tgt = {0: alpha, 1: alpha_u}
    oldest = {}
    c = np.full(len(S), -1)
    urgent = np.zeros(len(S), bool)
    Sm = S.copy(); Sm[:, 6] = np.where(d["vis"], Sm[:, 6], -1.0)
    for i in order:
        b, t, s = d["buoy"][i], d["t"][i], d["strat"][i]
        o = oldest.get(b)
        if o is None or t - o >= max_age:
            o = t if o is None else max(o, t - max_age + 1)
        u = int(t - o >= max_age - guard)
        urgent[i] = bool(u)
        lam = lam_for(s, lv[(u, s)])
        ok = Sm[i] >= lam
        k = int(ok.argmax()) if ok.any() else -1
        c[i] = k
        err = 0.0 if k < 0 else float(d["y"][i, k] < 0.5)
        lv[(u, s)] = float(np.clip(lv[(u, s)] + gamma * (tgt[u] - err), 1e-4, 0.6))
        oldest[b] = (t + 1) if k >= 0 else o      # transmission empties the buffer (success or loss)
    res = summarize(c, d)
    loss = outcome(c, d["y"])[0]
    res["risk_normal"] = float(loss[~urgent].mean())
    res["risk_urgent"] = float(loss[urgent].mean()) if urgent.any() else 0.0
    res["urgent_share"] = float(urgent.mean())
    return res
