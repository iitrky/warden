"""Statistical sanity checks of the risk-control layer and the simulator."""
import numpy as np
from warden import calib as cb, channel as ch, dataset as ds


def _synthetic(n, rng):
    p = rng.uniform(0.3, 1.0, (n, 7))
    p.sort(axis=1)
    y = (rng.random((n, 7)) < p).astype(np.float32)
    vis = rng.random(n) < 0.3
    hs = rng.uniform(0.2, 3.0, n)
    return dict(y=y, vis=vis, strat=ds.stratum(hs)), p


def test_ltt_controls_risk():
    """Mondrian LTT on an oracle score: test risk <= alpha in >= (1-delta) of repetitions."""
    rng = np.random.default_rng(0)
    alpha, delta, viol, n = 0.05, 0.1, 0, 0
    for _ in range(40):
        cal, pc = _synthetic(3000, rng)
        te, pt = _synthetic(50000, rng)
        lam, _ = cb.calibrate(pc, cal, alpha, delta)
        lamv = np.vectorize(lambda s: lam[s])(te["strat"])
        c = cb.choose(pt, te["vis"], lamv)
        # exact conditional risk from the true success probabilities (removes test-set noise)
        exp_loss = np.where(c >= 0, 1 - pt[np.arange(len(c)), np.clip(c, 0, None)], 0.0)
        for s in range(3):
            viol += exp_loss[te["strat"] == s].mean() > alpha; n += 1
    assert viol / n <= delta + 0.05     # per-stratum guarantee P[R_b > alpha] <= delta


def test_online_aci_tracks_target():
    rng = np.random.default_rng(1)
    cal, pc = _synthetic(3000, rng)
    te, pt = _synthetic(20000, rng)
    te["buoy"] = np.arange(20000) % 20
    te["t"] = np.arange(20000) // 20
    _, L = cb.calibrate(pc, cal, 0.05, 0.1)
    r = cb.online_aci(pt, te, L, cal["strat"], 0.05)
    assert abs(r["risk"] - 0.05) < 0.01


def test_lora_time_on_air_matches_semtech():
    # 20-byte payload, BW 125 kHz, CR 4/5, explicit header (Semtech calculator values, ms)
    ref = {7: 56.6, 9: 185.3, 12: 1318.9}
    for sf, ms in ref.items():
        assert abs(1e3 * ch.lora_toa(sf) - ms) < 1.0


def test_simulator_is_reproducible_and_prior_is_monotone():
    a = ds.simulate_deployment(seed=5, n_buoys=2, days=2)
    b = ds.simulate_deployment(seed=5, n_buoys=2, days=2)
    assert np.array_equal(a[0]["y"], b[0]["y"])
    assert np.all(np.diff(a[0]["prior"][:, :6], axis=1) >= -1e-6)
