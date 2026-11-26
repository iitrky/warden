"""Finite-difference gradient checks for the NumPy autodiff engine and all models."""
import numpy as np
import pytest
import warden.autograd as ag
from warden.autograd import bce_logits
from warden.models import WardenNet, GRUNet, TransformerNet


@pytest.mark.parametrize("ctor", [
    lambda: WardenNet(7, 10, 16),
    lambda: WardenNet(7, 10, 16, cross=False, monotone=False),
    lambda: WardenNet(7, 10, 16, prior=True),
    lambda: GRUNet(7, 10, 16),
    lambda: TransformerNet(7, 10, 16),
])
def test_gradients(ctor):
    old = ag.DTYPE
    ag.DTYPE = np.float64
    try:
        rng = np.random.default_rng(0)
        B, W = 3, 16
        seq, ctx, pr = rng.standard_normal((B, W, 7)), rng.standard_normal((B, 10)), rng.standard_normal((B, 8))
        y, m = (rng.random((B, 7)) > 0.5).astype(float), np.ones((B, 7))
        M = ctor()
        for p in M.params():
            p.data = p.data.astype(np.float64)
        loss = lambda: bce_logits(M(seq, ctx, pr), y, m)
        L = loss(); L.backward()
        worst = 0.0
        for p in M.params():
            if p.grad is None:
                continue
            g = p.grad.copy()
            idx = tuple(rng.integers(0, s) for s in p.data.shape)
            o = p.data[idx]
            p.data[idx] = o + 1e-6; lp = loss().data
            p.data[idx] = o - 1e-6; lm = loss().data
            p.data[idx] = o
            num = (lp - lm) / 2e-6
            if abs(num) + abs(g[idx]) > 1e-7:
                worst = max(worst, abs(num - g[idx]) / (abs(num) + abs(g[idx])))
        assert worst < 1e-4
    finally:
        ag.DTYPE = old


def test_monotone_head_orders_spreading_factors():
    rng = np.random.default_rng(1)
    M = WardenNet(7, 10, 16, seed=3)
    z = M(rng.standard_normal((50, 16, 7)).astype(np.float32), rng.standard_normal((50, 10)).astype(np.float32)).data
    assert np.all(np.diff(z[:, :6], axis=1) >= -1e-6)
