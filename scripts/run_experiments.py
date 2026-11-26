"""Full experimental pipeline for one random seed.

usage: python scripts/run_experiments.py --seed 1
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from warden import dataset as ds, calib as cb, channel as ch
from warden.models import WardenNet, GRUNet, TransformerNet, fit, predict_logits
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--delta", type=float, default=0.10)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

IMU_SEQ, IMU_CTX = [2, 3, 4, 5], [2, 3, 4]
SPLITS = dict(train=(0, 36), val=(36, 42), cal=(42, 48), test=(48, 60))


def make(dep):
    return {k: ds.build_arrays(dep, *v) for k, v in SPLITS.items()}


def standardize(D, stats=None):
    if stats is None:
        tr = D["train"]
        stats = (tr["seq"].reshape(-1, tr["seq"].shape[-1]).mean(0), tr["seq"].reshape(-1, tr["seq"].shape[-1]).std(0) + 1e-6,
                 tr["ctx"].mean(0), tr["ctx"].std(0) + 1e-6)
    ms, ss, mc, sc = stats
    for d in D.values():
        d["seq_n"] = ((d["seq"] - ms) / ss).astype(np.float32)
        d["ctx_n"] = ((d["ctx"] - mc) / sc).astype(np.float32)
    return stats


def view(d, no_imu=False, sub=1):
    s, c = d["seq_n"].copy(), d["ctx_n"].copy()
    if no_imu:
        s[:, :, IMU_SEQ] = 0; c[:, IMU_CTX] = 0
    v = dict(seq=s[::sub], ctx=c[::sub], y=d["y"][::sub], vis=d["vis"][::sub])
    return v


def tab_feats(d):
    s, c = d["seq_n"], d["ctx_n"]
    snr = s[:, :, 0]
    return np.concatenate([s[:, -4:, :].reshape(len(s), -1), c, snr.mean(1, keepdims=True),
                           snr.min(1, keepdims=True), snr.std(1, keepdims=True),
                           s[:, :, 1].mean(1, keepdims=True)], 1)


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def mask_of(d):
    return np.concatenate([np.ones((len(d["y"]), 6)), d["vis"][:, None]], 1)


# --------------------------------------------------------------------------
dep = ds.simulate_deployment(seed=args.seed, days=60)
D = make(dep)
stats = standardize(D)
# OOD: unseen buoys + severe storms; robustness: noisier IMU / beacon (same channel realisation)
D_ood = make(ds.simulate_deployment(seed=100 + args.seed, days=60, hs_mean=1.8, storm_rate=0.15,
                                    storm_peak=(4.0, 7.0), dist_km=(5, 32)))
standardize(D_ood, stats)
# OOD-B: site/hardware shift (lower gateway mast, livelier buoy hull, heavier interference)
D_site = make(ds.simulate_deployment(seed=200 + args.seed, days=60, h_gw=10.0, tilt_amp=3.0,
                                     lora_load=0.05, sat_coll=0.15, dist_km=(5, 26)))
standardize(D_site, stats)
D_imu = make(ds.simulate_deployment(seed=args.seed, days=60, imu_noise=0.30)); standardize(D_imu, stats)
D_bcn = make(ds.simulate_deployment(seed=args.seed, days=60, beacon_noise=3.0)); standardize(D_bcn, stats)
log("data ready", {k: len(v["y"]) for k, v in D.items()}, "ood test", len(D_ood["test"]["y"]))

sub = 2 if not args.quick else 6
ep = args.epochs if not args.quick else 2
W, dseq, dctx = ds.WINDOW, D["train"]["seq"].shape[-1], D["train"]["ctx"].shape[-1]

# --------------------------------------------------------------------------
# Train predictors -> calibrated scores on val/cal/test/ood
# --------------------------------------------------------------------------
SCORES = {}      # name -> {split: (mean prob, std prob)}


def neural(name, ctor, no_imu=False, members=1):
    probs = {k: [] for k in ["val", "cal", "test", "ood", "site", "imu", "bcn"]}
    for m in range(members):
        model = ctor(m)
        fit(model, view(D["train"], no_imu, sub), view(D["val"], no_imu, 2), epochs=ep, bs=512,
            lr=2e-3, seed=m)
        zv = predict_logits(model, view(D["val"], no_imu))
        temps = cb.fit_temperature(zv, D["val"]["y"], mask_of(D["val"]))
        for k, dd in [("val", D["val"]), ("cal", D["cal"]), ("test", D["test"]), ("ood", D_ood["test"]),
                      ("site", D_site["test"]), ("imu", D_imu["test"]), ("bcn", D_bcn["test"])]:
            probs[k].append(cb.sigmoid(predict_logits(model, view(dd, no_imu)) / temps))
        if name == "WARDEN" and m == 0:
            # attention weights of cross-attention for a figure
            _ = model(view(D["test"])["seq"][:2000], view(D["test"])["ctx"][:2000])
            np.save(os.path.join(OUT, f"attn_seed{args.seed}.npy"), model.ca.last_attn.mean(1)[:, 0, :])
    SCORES[name] = {k: (np.mean(v, 0), np.std(v, 0)) for k, v in probs.items()}
    log("trained", name)


neural("WARDEN", lambda m: WardenNet(dseq, dctx, W, seed=m), members=3)
neural("WARDEN-noIMU", lambda m: WardenNet(dseq, dctx, W, seed=10 + m), no_imu=True)
neural("WARDEN-noCross", lambda m: WardenNet(dseq, dctx, W, seed=20 + m, cross=False))
neural("WARDEN-noMono", lambda m: WardenNet(dseq, dctx, W, seed=30 + m, monotone=False))
neural("Transformer", lambda m: TransformerNet(dseq, dctx, W, seed=40 + m))
neural("GRU", lambda m: GRUNet(dseq, dctx, W, seed=50 + m))
# single WARDEN member (no ensemble) for ablation: re-use member variance-free mean of a separate run
neural("WARDEN-single", lambda m: WardenNet(dseq, dctx, W, seed=60 + m))

Xtr = tab_feats(D["train"])[::sub]; Ytr = D["train"]["y"][::sub]; Vtr = D["train"]["vis"][::sub]
for name, mk in [("LogReg", lambda: LogisticRegression(max_iter=300, C=1.0)),
                 ("HGBoost", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)),
                 ("MLP", None)]:
    if name == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=40, early_stopping=True, random_state=0)
        clf.fit(Xtr, Ytr.astype(int))
        pf = lambda X: np.stack([p[:, 1] for p in clf.predict_proba(X)], 1) if isinstance(clf.predict_proba(X), list) else clf.predict_proba(X)
    else:
        mods = []
        for k in range(7):
            idx = np.ones(len(Ytr), bool) if k < 6 else Vtr
            mods.append(mk().fit(Xtr[idx], Ytr[idx, k].astype(int)))
        pf = lambda X, mods=mods: np.stack([m.predict_proba(X)[:, 1] for m in mods], 1)
    zv = logit(pf(tab_feats(D["val"])))
    temps = cb.fit_temperature(zv, D["val"]["y"], mask_of(D["val"]))
    SCORES[name] = {}
    for k, dd in [("val", D["val"]), ("cal", D["cal"]), ("test", D["test"]), ("ood", D_ood["test"]),
                  ("site", D_site["test"]), ("imu", D_imu["test"]), ("bcn", D_bcn["test"])]:
        p = cb.sigmoid(logit(pf(tab_feats(dd))) / temps)
        SCORES[name][k] = (p, np.zeros_like(p))
    log("trained", name)

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
R = dict(seed=args.seed, alpha=args.alpha, delta=args.delta, pred={}, policy={}, ood={}, site={},
         robust={}, tradeoff={}, ncal={}, kappa={})
a, dl = args.alpha, args.delta


def score(name, split, kappa=1.0):
    mu, sd = SCORES[name][split]
    return np.clip(mu - kappa * sd, 0, 1)


def clean(r):
    return {k: float(v) for k, v in r.items() if k != "choice"}


for name in SCORES:
    R["pred"][name] = {k: float(v) for k, v in cb.pred_metrics(SCORES[name]["test"][0], D["test"]).items()}
    R["pred"][name + "@OOD"] = {k: float(v) for k, v in cb.pred_metrics(SCORES[name]["ood"][0], D_ood["test"]).items()}
    R["pred"][name + "@SITE"] = {k: float(v) for k, v in cb.pred_metrics(SCORES[name]["site"][0], D_site["test"]).items()}
log("pred metrics done")

# baselines without learning
R["policy"]["ADR-10dB"] = clean(cb.adr_baseline(D["test"]))
R["policy"]["Always-SF12"] = clean(cb.summarize(np.full(len(D["test"]["y"]), 5), D["test"]))
R["policy"]["Sat-first"] = clean(cb.summarize(np.where(D["test"]["vis"], 6, 5), D["test"]))
R["policy"]["Oracle"] = clean(cb.oracle(D["test"]))
R["ood"]["ADR-10dB"] = clean(cb.adr_baseline(D_ood["test"]))
R["ood"]["Oracle"] = clean(cb.oracle(D_ood["test"]))
R["site"]["ADR-10dB"] = clean(cb.adr_baseline(D_site["test"]))
R["site"]["Oracle"] = clean(cb.oracle(D_site["test"]))

# predictor x stratified-LTT
for name in SCORES:
    kap = 1.0 if name == "WARDEN" else 0.0
    lam, Lc = cb.calibrate(score(name, "cal", kap), D["cal"], a, dl, stratified=True)
    R["policy"][name + "+S-LTT"] = clean(cb.evaluate(score(name, "test", kap), D["test"], lam))
    R["policy"][name + "+S-LTT"]["lambda"] = {str(k): float(v) for k, v in lam.items()}
    R["ood"][name + "+S-LTT"] = clean(cb.evaluate(score(name, "ood", kap), D_ood["test"], lam))
    R["site"][name + "+S-LTT"] = clean(cb.evaluate(score(name, "site", kap), D_site["test"], lam))
log("predictor policies done")

# calibration-rule ablation on WARDEN
S_cal, S_te, S_ood = score("WARDEN", "cal"), score("WARDEN", "test"), score("WARDEN", "ood")
S_site = score("WARDEN", "site")
lam_g, Lg = cb.calibrate(S_cal, D["cal"], a, dl, stratified=False)
lam_s, Ls = cb.calibrate(S_cal, D["cal"], a, dl, stratified=True)
naive = {s: 1 - a for s in range(3)}
half = {s: 0.5 for s in range(3)}
for tag, lam in [("WARDEN+thr0.5", half), ("WARDEN+thr(1-a)", naive), ("WARDEN+G-LTT", lam_g)]:
    R["policy"][tag] = clean(cb.evaluate(S_te, D["test"], lam))
    R["ood"][tag] = clean(cb.evaluate(S_ood, D_ood["test"], lam))
    R["site"][tag] = clean(cb.evaluate(S_site, D_site["test"], lam))
full = cb.online_aci(S_te, D["test"], Ls, D["cal"]["strat"], a)
R["policy"]["WARDEN+S-LTT+ACI"] = clean(full)
ood_full = cb.online_aci(S_ood, D_ood["test"], Ls, D["cal"]["strat"], a)
R["ood"]["WARDEN+S-LTT+ACI"] = clean(ood_full)
site_full = cb.online_aci(S_site, D_site["test"], Ls, D["cal"]["strat"], a)
R["site"]["WARDEN+S-LTT+ACI"] = clean(site_full)
R["site"]["WARDEN+G-LTT+ACI"] = clean(cb.online_aci(S_site, D_site["test"], Lg, D["cal"]["strat"], a, stratified=False))
R["ood"]["WARDEN+G-LTT+ACI"] = clean(cb.online_aci(S_ood, D_ood["test"], Lg, D["cal"]["strat"], a, stratified=False))
R["policy"]["WARDEN(k=0)+S-LTT"] = clean(cb.evaluate(score("WARDEN", "test", 0.0), D["test"],
                                                    cb.calibrate(score("WARDEN", "cal", 0.0), D["cal"], a, dl)[0]))
log("calibration ablations done")

# robustness (models trained on clean data)
for tag, key, DD in [("IMU noise 30%", "imu", D_imu), ("Beacon noise 3 dB", "bcn", D_bcn)]:
    R["robust"][tag] = clean(cb.online_aci(score("WARDEN", key), DD["test"], Ls, D["cal"]["strat"], a))
    R["robust"][tag + " (no ACI)"] = clean(cb.evaluate(score("WARDEN", key), DD["test"], lam_s))
    R["robust"][tag + " ADR"] = clean(cb.adr_baseline(DD["test"]))

# ADR family and iso-risk comparison (alpha set to ADR's risk measured on the calibration split)
for mg in [5.0, 15.0]:
    R["policy"][f"ADR-{int(mg)}dB"] = clean(cb.adr_baseline(D["test"], mg))
a_iso = float(cb.adr_baseline(D["cal"])["risk"])
R["isorisk"] = {"alpha_iso": a_iso, "ADR-10dB": R["policy"]["ADR-10dB"]}
for name in ["WARDEN", "Transformer", "GRU", "LogReg", "HGBoost", "MLP", "WARDEN-noIMU", "WARDEN-noCross", "WARDEN-noMono"]:
    kap = 1.0 if name == "WARDEN" else 0.0
    lam, Li = cb.calibrate(score(name, "cal", kap), D["cal"], a_iso, dl)
    R["isorisk"][name + "+S-LTT+ACI"] = clean(cb.online_aci(score(name, "test", kap), D["test"], Li, D["cal"]["strat"], a_iso))

# risk-energy trade-off
for aa in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    for name in ["WARDEN", "Transformer", "GRU", "HGBoost", "LogReg"]:
        kap = 1.0 if name == "WARDEN" else 0.0
        lam, Li = cb.calibrate(score(name, "cal", kap), D["cal"], aa, dl)
        R["tradeoff"][f"{name}|{aa}"] = clean(cb.evaluate(score(name, "test", kap), D["test"], lam))
        R["tradeoff"][f"{name}+ACI|{aa}"] = clean(cb.online_aci(score(name, "test", kap), D["test"], Li, D["cal"]["strat"], aa))

# calibration-set size (repeat random sub-samples)
rng = np.random.default_rng(args.seed)
for ncal in [250, 500, 1000, 2000, 4000]:
    rs = []
    for r in range(20):
        idx = rng.choice(len(D["cal"]["y"]), ncal, replace=False)
        dc = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(D["cal"]["y"]) else v) for k, v in D["cal"].items()}
        lam, _ = cb.calibrate(S_cal[idx], dc, a, dl)
        rr = clean(cb.evaluate(S_te, D["test"], lam))
        rs.append([rr["risk"], rr["energy_per_deliv_mJ"], rr["defer"]])
    rs = np.array(rs)
    R["ncal"][str(ncal)] = dict(risk_mean=float(rs[:, 0].mean()), risk_max=float(rs[:, 0].max()),
                                viol=float((rs[:, 0] > a).mean()), e_mean=float(rs[:, 1].mean()),
                                defer=float(rs[:, 2].mean()))

json.dump(R, open(os.path.join(OUT, f"seed{args.seed}.json"), "w"), indent=1)
if args.seed == 1:
    np.savez_compressed(os.path.join(OUT, "seed1_arrays.npz"),
                        p_test=SCORES["WARDEN"]["test"][0], sd_test=SCORES["WARDEN"]["test"][1],
                        p_gru=SCORES["GRU"]["test"][0], p_tr=SCORES["Transformer"]["test"][0],
                        y_test=D["test"]["y"], vis_test=D["test"]["vis"], strat_test=D["test"]["strat"],
                        ood_choice=ood_full["choice"], ood_y=D_ood["test"]["y"], ood_t=D_ood["test"]["t"],
                        ood_buoy=D_ood["test"]["buoy"], ood_hs=D_ood["test"]["hs"],
                        ood_choice_g=cb.evaluate(S_ood, D_ood["test"], lam_g)["choice"],
                        ood_choice_s=cb.evaluate(S_ood, D_ood["test"], lam_s)["choice"],
                        site_choice=site_full["choice"], site_choice_s=cb.evaluate(S_site, D_site["test"], lam_s)["choice"],
                        site_y=D_site["test"]["y"], site_t=D_site["test"]["t"], site_buoy=D_site["test"]["buoy"],
                        hs_series=dep[0]["hs_true"], snr_series=dep[0]["snr_b"], elev_series=dep[0]["elev"],
                        p_series=dep[0]["p"])
log("saved; total", time.time() - T0)
