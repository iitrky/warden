"""Predictors: proposed WARDEN dual-attention network and neural baselines."""
import numpy as np
from .autograd import T, Param, softmax, layernorm, concat, bce_logits, Adam

N_OPT = 7


def _w(rng, i, o, s=None):
    s = s or np.sqrt(2.0 / (i + o))
    return Param(rng.standard_normal((i, o)).astype(np.float32) * s)


def _b(o, v=0.0):
    return Param(np.full((o,), v, np.float32))


class Module:
    def params(self):
        return [v for v in self.__dict__.values() if isinstance(v, Param)] + \
            [p for v in self.__dict__.values() if isinstance(v, Module) for p in v.params()]


class MHA(Module):
    def __init__(self, rng, d, h):
        self.h, self.dh = h, d // h
        self.wq, self.wk, self.wv, self.wo = (_w(rng, d, d) for _ in range(4))
        self.last_attn = None

    def __call__(self, q_in, kv_in):
        B, Lq, d = q_in.shape
        Lk = kv_in.shape[1]
        sp = lambda x, L: x.reshape(B, L, self.h, self.dh).transpose(0, 2, 1, 3)
        q, k, v = sp(q_in @ self.wq, Lq), sp(kv_in @ self.wk, Lk), sp(kv_in @ self.wv, Lk)
        a = softmax((q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.dh)), -1)
        self.last_attn = a.data
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, Lq, d)
        return o @ self.wo


class WardenNet(Module):
    """Wave-informed dual-attention link-success predictor.

    * temporal self-attention over the beacon/IMU history,
    * sea-state-conditioned cross-attention (query built from the current
      IMU sea-state and satellite-ephemeris context),
    * gated fusion, and
    * an SF-monotone ordinal head: p(SF7) <= p(SF8) <= ... <= p(SF12).
    """

    def __init__(self, d_seq, d_ctx, win, d=32, heads=2, seed=0, cross=True, monotone=True, prior=False):
        rng = np.random.default_rng(seed)
        self.cross, self.monotone, self.use_prior = cross, monotone, prior
        if prior:
            d_ctx = d_ctx + 8                                  # prior logits (7) + expected SNR
            self.beta_sf, self.beta_sat = Param(np.array([0.5], np.float32)), Param(np.array([0.5], np.float32))
        self.we, self.be = _w(rng, d_seq, d), _b(d)
        self.pos = Param(rng.standard_normal((win, d)).astype(np.float32) * 0.1)
        self.sa = MHA(rng, d, heads)
        self.g1, self.b1 = Param(np.ones(d, np.float32)), _b(d)
        self.f1, self.fb1, self.f2, self.fb2 = _w(rng, d, 2 * d), _b(2 * d), _w(rng, 2 * d, d), _b(d)
        self.g2, self.b2 = Param(np.ones(d, np.float32)), _b(d)
        self.c1, self.cb1, self.c2, self.cb2 = _w(rng, d_ctx, d), _b(d), _w(rng, d, d), _b(d)
        self.ca = MHA(rng, d, heads)
        self.wg, self.bg = _w(rng, d, d), _b(d)
        self.h1, self.hb1 = _w(rng, d, d), _b(d)
        self.w_base, self.b_base = _w(rng, d, 1), _b(1, 1.0)
        self.w_inc, self.b_inc = _w(rng, d, 5), _b(5, -1.0)
        self.w_sf, self.b_sf = _w(rng, d, 6), _b(6)
        self.w_sat, self.b_sat = _w(rng, d, 1), _b(1)

    def __call__(self, seq, ctx, prior=None):
        if self.use_prior:
            ctx = np.concatenate([ctx, prior], 1)
        s, c = T(seq), T(ctx)
        h = s @ self.we + self.be + self.pos
        h = layernorm(h + self.sa(h, h), self.g1, self.b1)
        h = layernorm(h + (h @ self.f1 + self.fb1).gelu() @ self.f2 + self.fb2, self.g2, self.b2)
        cz = (c @ self.c1 + self.cb1).gelu() @ self.c2 + self.cb2
        if self.cross:
            B, d = cz.shape
            z = self.ca(cz.reshape(B, 1, d), h).reshape(B, d)
        else:
            z = h.mean(1)
        u = z * (cz @ self.wg + self.bg).sigmoid() + cz
        u = (u @ self.h1 + self.hb1).gelu()
        if self.monotone:
            base = u @ self.w_base + self.b_base
            inc = (u @ self.w_inc + self.b_inc).softplus().cumsum(-1)
            sf = concat([base, base + inc], -1)
        else:
            sf = u @ self.w_sf + self.b_sf
        sat = u @ self.w_sat + self.b_sat
        if self.use_prior:
            # physics-residual head: network corrects the closed-form wave prior (eq. 20)
            sf = sf + self.beta_sf.softplus() * T(prior[:, :6])
            sat = sat + self.beta_sat.softplus() * T(prior[:, 6:7])
        return concat([sf, sat], -1)


class TransformerNet(Module):
    """Plain attention baseline: self-attention + mean pooling, context concatenated."""

    def __init__(self, d_seq, d_ctx, win, d=32, heads=2, seed=0):
        rng = np.random.default_rng(seed)
        self.we, self.be = _w(rng, d_seq, d), _b(d)
        self.pos = Param(rng.standard_normal((win, d)).astype(np.float32) * 0.1)
        self.sa = MHA(rng, d, heads)
        self.g1, self.b1 = Param(np.ones(d, np.float32)), _b(d)
        self.f1, self.fb1, self.f2, self.fb2 = _w(rng, d, 2 * d), _b(2 * d), _w(rng, 2 * d, d), _b(d)
        self.g2, self.b2 = Param(np.ones(d, np.float32)), _b(d)
        self.h1, self.hb1 = _w(rng, d + d_ctx, d), _b(d)
        self.wo, self.bo = _w(rng, d, N_OPT), _b(N_OPT)

    def __call__(self, seq, ctx, prior=None):
        s = T(seq)
        h = s @ self.we + self.be + self.pos
        h = layernorm(h + self.sa(h, h), self.g1, self.b1)
        h = layernorm(h + (h @ self.f1 + self.fb1).gelu() @ self.f2 + self.fb2, self.g2, self.b2)
        u = (concat([h.mean(1), T(ctx)], -1) @ self.h1 + self.hb1).gelu()
        return u @ self.wo + self.bo


class GRUNet(Module):
    def __init__(self, d_seq, d_ctx, win, d=32, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.wx, self.wh, self.bb = _w(rng, d_seq, 3 * d), _w(rng, d, 3 * d), _b(3 * d)
        self.h1, self.hb1 = _w(rng, d + d_ctx, d), _b(d)
        self.wo, self.bo = _w(rng, d, N_OPT), _b(N_OPT)

    def __call__(self, seq, ctx, prior=None):
        B, W, _ = seq.shape
        d = self.d
        h = T(np.zeros((B, d), np.float32))
        xs = T(seq) @ self.wx + self.bb
        for t in range(W):
            xt = xs[:, t, :]
            hh = h @ self.wh
            r = (xt[:, :d] + hh[:, :d]).sigmoid()
            zg = (xt[:, d:2 * d] + hh[:, d:2 * d]).sigmoid()
            n = (xt[:, 2 * d:] + r * hh[:, 2 * d:]).tanh()
            h = (1 - zg) * n + zg * h
        u = (concat([h, T(ctx)], -1) @ self.h1 + self.hb1).gelu()
        return u @ self.wo + self.bo


# ----------------------------------------------------------------------------
def fit(model, tr, va, epochs=8, bs=256, lr=2e-3, seed=0, verbose=False, patience=2):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    n = len(tr["y"])
    mask_tr = np.concatenate([np.ones((n, 6)), tr["vis"][:, None]], 1).astype(np.float32)
    best, best_state, bad = np.inf, None, 0
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            pr = tr["prior"][j] if "prior" in tr else None
            loss = bce_logits(model(tr["seq"][j], tr["ctx"][j], pr), tr["y"][j], mask_tr[j])
            loss.backward()
            opt.step()
        vl = eval_loss(model, va)
        hist.append(vl)
        if verbose:
            print(f"  epoch {ep} val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = [p.data.copy() for p in model.params()]
        else:
            bad += 1
            if bad >= patience:
                break
        opt.lr *= 0.7
    for p, s in zip(model.params(), best_state):
        p.data = s
    return hist


def predict_logits(model, d, bs=2048):
    out = []
    for i in range(0, len(d["y"]), bs):
        pr = d["prior"][i:i + bs] if "prior" in d else None
        out.append(model(d["seq"][i:i + bs], d["ctx"][i:i + bs], pr).data)
    return np.concatenate(out, 0)


def eval_loss(model, d):
    z = predict_logits(model, d)
    m = np.concatenate([np.ones((len(z), 6)), d["vis"][:, None]], 1)
    l = np.logaddexp(0, z) - z * d["y"]
    return float((l * m).sum() / m.sum())
