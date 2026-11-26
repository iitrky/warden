import json, glob, sys, numpy as np
runs=[json.load(open(f)) for f in sorted(glob.glob('results/v2_seed[0-9].json'))]
print('seeds',[r['seed'] for r in runs])
def m(sec,k,f): v=np.array([r[sec][k][f] for r in runs]); return v.mean(), v.std()
print('== pred')
for k in runs[0]['pred']:
    print(f"{k:42s}", ' '.join(f"{f} {m('pred',k,f)[0]:.4f}±{m('pred',k,f)[1]:.4f}" for f in ['auroc','brier','ece']))
print('== policy')
for k in runs[0]['policy']:
    print(f"{k:46s}", ' '.join(f"{f} {m('policy',k,f)[0]:.3f}" for f in ['risk','risk_s0','risk_s1','risk_s2','defer','rdr','latency_min','energy_per_reading_mJ']), 'sdE %.1f sdR %.3f'%(m('policy',k,'energy_per_reading_mJ')[1],m('policy',k,'rdr')[1]))
print('== tier')
for k in runs[0]['tier']:
    print(f"{k:14s}", ' '.join(f"{f} {m('tier',k,f)[0]:.3f}" for f in ['risk','risk_normal','risk_urgent','urgent_share','defer','rdr','latency_min','energy_per_reading_mJ']), 'sdRDR %.3f'%m('tier',k,'rdr')[1])
print('== tradeoff')
for k in runs[0]['tradeoff']:
    print(f"{k:28s}", ' '.join(f"{f} {m('tradeoff',k,f)[0]:.3f}" for f in ['risk','rdr','energy_per_reading_mJ','latency_min']))
