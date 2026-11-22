import sys, pickle, glob, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'environment')); sys.path.insert(0, str(HERE.parent / 'components'))
from train_campaign_numpy import run_episode, DistPolicy, CentPolicy, Static, Rule, Random, Clairvoyant, EVAL_STARTS, HORIZON, SLICES
from trace_resource_env import TraceResourceEnv
out = HERE / 'results_fixed4' / 'per_episode_records.csv'
rows = pd.read_csv(out).to_dict('records') if out.exists() else []
done = {(r['method'], int(r['seed'])) for r in rows}
jobs = [('static', 0, Static()), ('rule', 0, Rule()), ('random', 0, Random(0)), ('clairvoyant', 0, Clairvoyant())]
for fn in sorted(glob.glob(str(HERE / 'results_fixed4' / 'agents' / '*_seed*.pkl'))):
    m, seed = os.path.basename(fn)[:-4].rsplit('_seed', 1); ag = pickle.load(open(fn, 'rb'))
    jobs.append((m, int(seed), DistPolicy(ag) if isinstance(ag, list) else CentPolicy(ag, 'DQN' if m == 'DQN' else 'PPO')))
for m, seed, pol in jobs:
    if (m, seed) in done: continue
    env = TraceResourceEnv(horizon=HORIZON)
    if m == 'D-PPO-nogate': env.plans = lambda: ['OPTIMIZE'] * 3
    for ep, st in enumerate(EVAL_STARTS):
        r = run_episode(env, pol, st, mode_central=(m in ('C-PPO', 'DQN', 'PPO-Lagrangian', 'clairvoyant')))
        for i, s in enumerate(SLICES):
            rows.append(dict(method=m, seed=seed, episode=ep, start_index=st, slice=s, latency=r['latency'][i], sla_viol=r['sla_viol'][i], throughput=r['throughput'][i], isolation=r['isolation'][i], reward=r['reward'][i], reject_rate=r['reject_rate'][i], msgs_per_s=r['msgs_per_s']))
    pd.DataFrame(rows).to_csv(out, index=False); print('[records]', m, seed, flush=True)
