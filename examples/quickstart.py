"""Five-minute WARDEN demo: simulate, train one WARDEN-PI network, certify, evaluate.

python examples/quickstart.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from warden import dataset as ds, calib as cb
from warden.models import WardenNet, fit, predict_logits

dep = ds.simulate_deployment(seed=7, n_buoys=8, days=30)            # 8 buoys, 30 days
D = {k: ds.build_arrays(dep, *v) for k, v in
     dict(train=(0, 18), val=(18, 21), cal=(21, 24), test=(24, 30)).items()}
s, c = D["train"]["seq"].reshape(-1, 7), D["train"]["ctx"]
mu_s, sd_s, mu_c, sd_c = s.mean(0), s.std(0) + 1e-6, c.mean(0), c.std(0) + 1e-6
V = {k: dict(seq=(d["seq"] - mu_s) / sd_s, ctx=(d["ctx"] - mu_c) / sd_c, y=d["y"], vis=d["vis"], prior=d["prior"])
     for k, d in D.items()}

net = WardenNet(7, 10, ds.WINDOW, prior=True)                         # WARDEN-PI (single member)
fit(net, V["train"], V["val"], epochs=5, bs=512, verbose=True)
mask = np.concatenate([np.ones((len(D["val"]["y"]), 6)), D["val"]["vis"][:, None]], 1)
T = cb.fit_temperature(predict_logits(net, V["val"]), D["val"]["y"], mask)
score = {k: cb.sigmoid(predict_logits(net, V[k]) / T) for k in ["cal", "test"]}

alpha = 0.05
lam, L_cal = cb.calibrate(score["cal"], D["cal"], alpha, delta=0.10)           # Mondrian LTT
offline = cb.evaluate(score["test"], D["test"], lam)
online = cb.online_two_tier(score["test"], D["test"], L_cal, D["cal"]["strat"], alpha, alpha)  # WARDEN-U
adr = cb.adr_baseline(D["test"])
for name, r in [("ADR 10 dB", adr), ("WARDEN-PI offline LTT", offline), ("WARDEN-PI-U online", online)]:
    print(f"{name:24s} risk {r['risk']:.3f}  RDR {r['rdr']:.3f}  energy/reading {r['energy_per_reading_mJ']:.1f} mJ")
