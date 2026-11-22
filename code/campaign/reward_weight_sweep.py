import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'environment')); sys.path.insert(0, str(HERE.parent / 'components'))
import fuzzy_system as fz
from train_campaign_numpy import train_distributed, evaluate, DistPolicy, SLICES
out = HERE / 'results_adapted4' / 'extras' / 'reward_weight_sensitivity.csv'
rows = pd.read_csv(out).to_dict('records') if out.exists() else []
done = {r['variant'] for r in rows}
base = {k: v.copy() for k, v in fz.WEIGHTS.items()}
for tag, idx, f in [('lat-20', 0, 0.8), ('lat+20', 0, 1.2), ('rel-20', 1, 0.8), ('rel+20', 1, 1.2)]:
    if tag in done: continue
    for k in fz.WEIGHTS:
        w = base[k].copy(); w[idx] *= f; w = w / w.sum(); w.setflags(write=False); fz.WEIGHTS[k] = w
    agents, _, _ = train_distributed(0, 200000); agg, _ = evaluate(DistPolicy(agents))
    for s in SLICES: rows.append(dict(variant=tag, slice=s, **agg[s]))
    pd.DataFrame(rows).to_csv(out, index=False); print('[weights]', tag, ' '.join(f"{s}={agg[s]['sla_viol']:.2f}%" for s in SLICES), flush=True)
    for k in fz.WEIGHTS: fz.WEIGHTS[k] = base[k]
