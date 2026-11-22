import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'environment')); sys.path.insert(0, str(HERE.parent / 'components'))
import trace_resource_env as tre
from train_campaign_numpy import train_centralized, evaluate, CentPolicy, DistPolicy, SLICES
out = HERE / 'results_adapted4' / 'extras' / 'priority_weight_sensitivity.csv'
rows = pd.read_csv(out).to_dict('records') if out.exists() else []
done = {(r['omega'], r['method']) for r in rows}
dagents = pickle.load(open(HERE / 'results_fixed4' / 'agents' / 'D-PPO_seed0.pkl', 'rb'))
cagent0 = pickle.load(open(HERE / 'results_fixed4' / 'agents' / 'C-PPO_seed0.pkl', 'rb'))
for om in [(0.3, 0.4, 0.3), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1)]:
    tag = f"{om[0]}:{om[1]}:{om[2]}"; tre.OMEGA[:] = [om[1], om[0], om[2]]
    if (tag, 'D-PPO') not in done:
        agg, _ = evaluate(DistPolicy(dagents), central=False)
        for s in SLICES: rows.append(dict(omega=tag, method='D-PPO', slice=s, **agg[s]))
        pd.DataFrame(rows).to_csv(out, index=False); print('[priority]', tag, 'D-PPO', ' '.join(f"{s}={agg[s]['sla_viol']:.1f}%" for s in SLICES), flush=True)
    if (tag, 'C-PPO') not in done:
        if om == (0.5, 0.3, 0.2): agent = cagent0
        else: agent, _, _ = train_centralized(0, 200000, 'PPO')
        agg, _ = evaluate(CentPolicy(agent), central=True)
        for s in SLICES: rows.append(dict(omega=tag, method='C-PPO', slice=s, **agg[s]))
        pd.DataFrame(rows).to_csv(out, index=False); print('[priority]', tag, 'C-PPO', ' '.join(f"{s}={agg[s]['sla_viol']:.1f}%" for s in SLICES), flush=True)
