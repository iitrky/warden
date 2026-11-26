"""A minimal reverse-mode automatic differentiation engine on NumPy.

Written so the whole pipeline runs with NumPy only (no deep-learning framework
required). Supports the operations needed by attention and GRU models.
"""
import numpy as np

DTYPE = np.float32


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class T:
    __slots__ = ("data", "grad", "_back", "_prev", "req")

    def __init__(self, data, prev=(), back=None, req=False):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = None
        self._prev = prev
        self._back = back
        self.req = req or any(p.req for p in prev)

    shape = property(lambda s: s.data.shape)

    # ---- graph utilities -------------------------------------------------
    def backward(self, g=None):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen or not v.req:
                return
            seen.add(id(v))
            for p in v._prev:
                build(p)
            topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data) if g is None else g
        for v in reversed(topo):
            if v._back is not None and v.grad is not None:
                v._back(v.grad)

    def _acc(self, g):
        if not self.req:
            return
        self.grad = g if self.grad is None else self.grad + g

    # ---- elementwise -----------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.data + o.data, (self, o))
        out._back = lambda g: (self._acc(_unbroadcast(g, self.shape)), o._acc(_unbroadcast(g, o.shape)))
        return out
    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-(o if isinstance(o, T) else T(o)))

    def __rsub__(self, o):
        return T(o) - self

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.data * o.data, (self, o))
        out._back = lambda g: (self._acc(_unbroadcast(g * o.data, self.shape)),
                               o._acc(_unbroadcast(g * self.data, o.shape)))
        return out
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, T) else T(o)
        return self * o.pow(-1.0)

    def pow(self, p):
        out = T(self.data ** p, (self,))
        out._back = lambda g: self._acc(g * p * self.data ** (p - 1))
        return out

    def exp(self):
        e = np.exp(self.data)
        out = T(e, (self,))
        out._back = lambda g: self._acc(g * e)
        return out

    def log(self):
        out = T(np.log(self.data), (self,))
        out._back = lambda g: self._acc(g / self.data)
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = T(t, (self,))
        out._back = lambda g: self._acc(g * (1 - t * t))
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = T(s, (self,))
        out._back = lambda g: self._acc(g * s * (1 - s))
        return out

    def relu(self):
        m = self.data > 0
        out = T(self.data * m, (self,))
        out._back = lambda g: self._acc(g * m)
        return out

    def gelu(self):
        x = self.data
        c = np.sqrt(2 / np.pi)
        u = c * (x + 0.044715 * x ** 3)
        th = np.tanh(u)
        out = T(0.5 * x * (1 + th), (self,))
        du = c * (1 + 3 * 0.044715 * x ** 2)
        out._back = lambda g: self._acc(g * (0.5 * (1 + th) + 0.5 * x * (1 - th ** 2) * du))
        return out

    def softplus(self):
        x = self.data
        out = T(np.logaddexp(0, x), (self,))
        out._back = lambda g: self._acc(g / (1 + np.exp(-x)))
        return out

    # ---- reductions / shape ---------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).copy())
        out._back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def reshape(self, *s):
        out = T(self.data.reshape(*s), (self,))
        out._back = lambda g: self._acc(g.reshape(self.shape))
        return out

    def transpose(self, *ax):
        out = T(self.data.transpose(*ax), (self,))
        inv = np.argsort(ax)
        out._back = lambda g: self._acc(g.transpose(*inv))
        return out

    def __getitem__(self, idx):
        out = T(self.data[idx], (self,))

        def back(g):
            z = np.zeros_like(self.data)
            np.add.at(z, idx, g)
            self._acc(z)
        out._back = back
        return out

    def __matmul__(self, o):
        out = T(self.data @ o.data, (self, o))

        def back(g):
            a, b = self.data, o.data
            ga = g @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ g
            self._acc(_unbroadcast(ga, a.shape))
            o._acc(_unbroadcast(gb, b.shape))
        out._back = back
        return out

    def cumsum(self, axis=-1):
        out = T(np.cumsum(self.data, axis), (self,))
        out._back = lambda g: self._acc(np.flip(np.cumsum(np.flip(g, axis), axis), axis))
        return out


def concat(ts, axis=-1):
    out = T(np.concatenate([t.data for t in ts], axis), tuple(ts))
    sizes = np.cumsum([t.shape[axis] for t in ts])[:-1]

    def back(g):
        for t, gg in zip(ts, np.split(g, sizes, axis)):
            t._acc(gg)
    out._back = back
    return out


def softmax(x, axis=-1):
    m = x.data.max(axis, keepdims=True)
    e = (x - m).exp()
    return e / e.sum(axis, keepdims=True)


def layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    return xc * (var + eps).pow(-0.5) * gamma + beta


def bce_logits(z, y, mask):
    """Masked binary cross-entropy on logits (numerically stable)."""
    loss = z.softplus() - z * y
    return (loss * mask).sum() * (1.0 / max(mask.sum(), 1.0))


class Param(T):
    def __init__(self, data):
        super().__init__(data, req=True)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-4):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)
            p.grad = None
