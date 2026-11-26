"""Version-2 experiments: physics-informed prior (WARDEN-PI), prior misspecification,
and urgency-aware two-tier certification (WARDEN-U).  usage: python scripts/run_v2.py --seed 1
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from warden import dataset as ds, calib as cb
from warden.models import WardenNet, fit, predict_logits
from sklearn.ensemble import HistGradientBoostingClassifier

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
SPLITS = dict(train=(0, 36), val=(36, 42), cal=(42, 48), test=(48, 60))
make = lambda dep: {k: ds.build_arrays(dep, *v) for k, v in SPLITS.items()}


def standardize(D, stats=None):
    if stats is None:
        s = D["train"]["seq"].reshape(-1, D["train"]["seq"].shape[-1]); c = D["train"]["ctx"]
        stats = (s.mean(0), s.std(0) + 1e-6, c.mean(0), c.std(0) + 1e-6)
    for d in D.values():
        d["seq_n"] = ((d["seq"] - stats[0]) / stats[1]).astype(np.float32)
        d["ctx_n"] = ((d["ctx"] - stats[2]) / stats[3]).astype(np.float32)
    return stats


view = lambda d, sub=1, prior=False: dict(seq=d["seq_n"][::sub], ctx=d["ctx_n"][::sub], y=d["y"][::sub], vis=d["vis"][::sub],
                                          **({"prior": d["prior"][::sub]} if prior else {}))
mask_of = lambda d: np.concatenate([np.ones((len(d["y"]), 6)), d["vis"][:, None]], 1)
logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
clean = lambda r: {k: float(v) for k, v in r.items() if k != "choice"}

D = make(ds.simulate_deployment(seed=args.seed)); stats = standardize(D)
D_mis = make(ds.simulate_deployment(seed=args.seed, prior_misspec=True)); standardize(D_mis, stats)
D_ood = make(ds.simulate_deployment(seed=100 + args.seed, hs_mean=1.8, storm_rate=0.15, storm_peak=(4.0, 7.0))); standardize(D_ood, stats)
D_site = make(ds.simulate_deployment(seed=200 + args.seed, h_gw=10.0, tilt_amp=3.0, lora_load=0.05,
                                     sat_coll=0.15, dist_km=(5, 26))); standardize(D_site, stats)
EVAL = {"test": D["test"], "ood": D_ood["test"], "site": D_site["test"]}
log("data ready")
sub, ep = (6, 2) if args.quick else (2, 8)
W, dseq, dctx = ds.WINDOW, 7, 10
SC = {}


def neural(name, prior, DD, members=3, seed0=0):
    probs = {k: [] for k in ["cal", "test", "ood", "site"]}
    for m in range(members):
        model = WardenNet(dseq, dctx, W, seed=seed0 + m, prior=prior)
        fit(model, view(DD["train"], sub, prior), view(DD["val"], 2, prior), epochs=ep, bs=512, lr=2e-3, seed=m)
        temps = cb.fit_temperature(predict_logits(model, view(DD["val"], 1, prior)), DD["val"]["y"], mask_of(DD["val"]))
        srcs = {"cal": DD["cal"], "test": DD["test"], "ood": D_ood["test"], "site": D_site["test"]}
        for k, dd in srcs.items():
            probs[k].append(cb.sigmoid(predict_logits(model, view(dd, 1, prior)) / temps))
    SC[name] = {k: (np.mean(v, 0), np.std(v, 0)) for k, v in probs.items()}
    log("trained", name)


neural("WARDEN", False, D)
neural("WARDEN-PI", True, D, seed0=100)
neural("WARDEN-PI (misspecified prior)", True, D_mis, seed0=200)


def tab(d, prior):
    s, c = d["seq_n"], d["ctx_n"]; snr = s[:, :, 0]
    f = [s[:, -4:, :].reshape(len(s), -1), c, snr.mean(1, keepdims=True), snr.min(1, keepdims=True),
         snr.std(1, keepdims=True), s[:, :, 1].mean(1, keepdims=True)]
    return np.concatenate(f + ([d["prior"]] if prior else []), 1)


for name, prior in [("HGBoost", False), ("HGBoost+prior", True)]:
    tr = D["train"]; X = tab(tr, prior)[::sub]; Y = tr["y"][::sub]; V = tr["vis"][::sub]
    mods = [HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
            .fit(X[np.ones(len(Y), bool) if k < 6 else V], Y[np.ones(len(Y), bool) if k < 6 else V, k].astype(int)) for k in range(7)]
    pf = lambda d: np.stack([m.predict_proba(tab(d, prior))[:, 1] for m in mods], 1)
    temps = cb.fit_temperature(logit(pf(D["val"])), D["val"]["y"], mask_of(D["val"]))
    SC[name] = {k: (cb.sigmoid(logit(pf(dd)) / temps), 0) for k, dd in {"cal": D["cal"], **EVAL}.items()}
    log("trained", name)
# physics prior alone (no learning), temperature-scaled
temps = cb.fit_temperature(D["val"]["prior"][:, :7], D["val"]["y"], mask_of(D["val"]))
SC["Physics prior only"] = {k: (cb.sigmoid(dd["prior"][:, :7] / temps), 0) for k, dd in {"cal": D["cal"], **EVAL}.items()}

R = dict(seed=args.seed, pred={}, policy={}, tier={}, tradeoff={})
score = lambda n, k, kap=1.0: np.clip(SC[n][k][0] - kap * SC[n][k][1], 0, 1)
for n in SC:
    for k in EVAL:
        R["pred"][f"{n}@{k}"] = {a: float(b) for a, b in cb.pred_metrics(SC[n][k][0], EVAL[k]).items()}
    lam, Lc = cb.calibrate(score(n, "cal"), D["cal"], 0.05, 0.10)
    for k in EVAL:
        R["policy"][f"{n}@{k}"] = clean(cb.online_aci(score(n, k), EVAL[k], Lc, D["cal"]["strat"], 0.05))
        R["policy"][f"{n}@{k}|offline"] = clean(cb.evaluate(score(n, k), EVAL[k], lam))
for k in EVAL:
    R["policy"][f"ADR-10dB@{k}"] = clean(cb.adr_baseline(EVAL[k]))
log("policies done")
lamPI, LPI = cb.calibrate(score("WARDEN-PI", "cal"), D["cal"], 0.05, 0.10)
for au in [0.05, 0.10, 0.20, 0.30]:
    for k in EVAL:
        R["tier"][f"{au}@{k}"] = clean(cb.online_two_tier(score("WARDEN-PI", k), EVAL[k], LPI, D["cal"]["strat"], 0.05, au))
for a in [0.02, 0.05, 0.10, 0.15]:
    for n in ["WARDEN", "WARDEN-PI", "HGBoost+prior"]:
        _, Li = cb.calibrate(score(n, "cal"), D["cal"], a, 0.10)
        R["tradeoff"][f"{n}|{a}"] = clean(cb.online_aci(score(n, "test"), D["test"], Li, D["cal"]["strat"], a))
    for au in [0.20, 0.30]:
        _, Li = cb.calibrate(score("WARDEN-PI", "cal"), D["cal"], a, 0.10)
        R["tradeoff"][f"WARDEN-PI-U{au}|{a}"] = clean(cb.online_two_tier(score("WARDEN-PI", "test"), D["test"], Li, D["cal"]["strat"], a, au))
json.dump(R, open(os.path.join(OUT, f"v2_seed{args.seed}.json"), "w"), indent=1)
log("saved")
