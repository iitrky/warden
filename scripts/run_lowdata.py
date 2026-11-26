"""Data-efficiency study: commissioning with only a few days of labelled data.
Train on the first `days` days, validate on the next day, calibrate on the following two days,
test on days 48-60.  usage: python scripts/run_lowdata.py --seed 1
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from warden import dataset as ds, calib as cb
from warden.models import WardenNet, fit, predict_logits
from sklearn.ensemble import HistGradientBoostingClassifier

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); args = ap.parse_args()
OUT = os.path.join(os.path.dirname(__file__), "..", "results"); T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
dep = ds.simulate_deployment(seed=args.seed)
mask_of = lambda d: np.concatenate([np.ones((len(d["y"]), 6)), d["vis"][:, None]], 1)
logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
clean = lambda r: {k: float(v) for k, v in r.items() if k != "choice"}
R = {"seed": args.seed}
for days in [2, 4, 8]:
    D = {"train": ds.build_arrays(dep, 0, days), "val": ds.build_arrays(dep, days, days + 1),
         "cal": ds.build_arrays(dep, days + 1, days + 3), "test": ds.build_arrays(dep, 48, 60)}
    s = D["train"]["seq"].reshape(-1, 7); c = D["train"]["ctx"]
    st = (s.mean(0), s.std(0) + 1e-6, c.mean(0), c.std(0) + 1e-6)
    for d in D.values():
        d["seq_n"] = ((d["seq"] - st[0]) / st[1]).astype(np.float32); d["ctx_n"] = ((d["ctx"] - st[2]) / st[3]).astype(np.float32)
    view = lambda d, pr: dict(seq=d["seq_n"], ctx=d["ctx_n"], y=d["y"], vis=d["vis"], **({"prior": d["prior"]} if pr else {}))
    SC = {}
    for name, pr in [("WARDEN", False), ("WARDEN-PI", True)]:
        P = {"cal": [], "test": []}
        for m in range(3):
            mdl = WardenNet(7, 10, ds.WINDOW, seed=300 + m, prior=pr)
            fit(mdl, view(D["train"], pr), view(D["val"], pr), epochs=20, bs=256, lr=2e-3, seed=m, patience=3)
            T = cb.fit_temperature(predict_logits(mdl, view(D["val"], pr)), D["val"]["y"], mask_of(D["val"]))
            for k in P:
                P[k].append(cb.sigmoid(predict_logits(mdl, view(D[k], pr)) / T))
        SC[name] = {k: np.clip(np.mean(v, 0) - np.std(v, 0), 0, 1) for k, v in P.items()}
    def tab(d, pr):
        sq = d["seq_n"]; snr = sq[:, :, 0]
        f = [sq[:, -4:, :].reshape(len(sq), -1), d["ctx_n"], snr.mean(1, keepdims=True), snr.min(1, keepdims=True), snr.std(1, keepdims=True)]
        return np.concatenate(f + ([d["prior"]] if pr else []), 1)
    for name, pr in [("HGBoost", False), ("HGBoost+prior", True)]:
        X, Y, V = tab(D["train"], pr), D["train"]["y"], D["train"]["vis"]
        mods = [HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(
            X[np.ones(len(Y), bool) if k < 6 else V], Y[np.ones(len(Y), bool) if k < 6 else V, k].astype(int)) for k in range(7)]
        pf = lambda d: np.stack([mm.predict_proba(tab(d, pr))[:, 1] for mm in mods], 1)
        T = cb.fit_temperature(logit(pf(D["val"])), D["val"]["y"], mask_of(D["val"]))
        SC[name] = {k: cb.sigmoid(logit(pf(D[k])) / T) for k in ["cal", "test"]}
    T = cb.fit_temperature(D["val"]["prior"][:, :7], D["val"]["y"], mask_of(D["val"]))
    SC["Physics prior only"] = {k: cb.sigmoid(D[k]["prior"][:, :7] / T) for k in ["cal", "test"]}
    for name in SC:
        _, Lc = cb.calibrate(SC[name]["cal"], D["cal"], 0.05, 0.10)
        r = clean(cb.online_aci(SC[name]["test"], D["test"], Lc, D["cal"]["strat"], 0.05))
        r.update({"pred_" + a: float(b) for a, b in cb.pred_metrics(SC[name]["test"], D["test"]).items()})
        R[f"{name}|{days}"] = r
        if name == "WARDEN-PI":
            R[f"WARDEN-PI-U|{days}"] = clean(cb.online_two_tier(SC[name]["test"], D["test"], Lc, D["cal"]["strat"], 0.05, 0.05))
    log("done days", days, len(D["train"]["y"]))
json.dump(R, open(os.path.join(OUT, f"low_seed{args.seed}.json"), "w"), indent=1)
log("saved")
