import sys, json, os, pickle, time, glob
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'environment')); sys.path.insert(0, str(HERE.parent / 'components'))
from train_campaign_numpy import evaluate, DistPolicy, CentPolicy, run_episode, HORIZON, EVAL_STARTS, SLICES
from trace_resource_env import TraceResourceEnv
src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results_fixed'
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'results_adapted'
(dst / 'logs').mkdir(parents=True, exist_ok=True); (dst / 'extras').mkdir(exist_ok=True)
summ = pd.read_csv(src / 'summary.csv')

p95, p99 = {}, {}
for fn in glob.glob(str(src / 'logs' / '*_seed0.csv.gz')):
    d = pd.read_csv(fn); d['sla'] = (d['miss'] == 0).astype(int); d['D'] = d['offered']
    m = os.path.basename(fn).replace('_seed0.csv.gz', '')
    for s in SLICES:
        g = d[(d.slice == s) & (d.rejected == 0) & (d.L > 0)]
        p95[(m, s)] = float(g.L.quantile(0.95)) if len(g) else np.nan; p99[(m, s)] = float(g.L.quantile(0.99)) if len(g) else np.nan
    d.to_csv(dst / 'logs' / os.path.basename(fn), index=False)
summ['p95_latency'] = [p95.get((m, s), np.nan) for m, s in zip(summ.method, summ.slice)]
summ['p99_latency'] = [p99.get((m, s), np.nan) for m, s in zip(summ.method, summ.slice)]
summ['n'] = 20

inf = {}
for fn in glob.glob(str(src / 'agents' / '*_seed0.pkl')):
    m = os.path.basename(fn).replace('_seed0.pkl', ''); ag = pickle.load(open(fn, 'rb'))
    if isinstance(ag, list):
        o = np.zeros(14); t0 = time.perf_counter()
        for _ in range(700):
            for a in ag: a.act(o, deterministic=True)
        inf[m] = (time.perf_counter() - t0) / 700 * 1e3
    else:
        o = np.zeros(42); t0 = time.perf_counter()
        for _ in range(2000): ag.act(o, deterministic=True)
        inf[m] = (time.perf_counter() - t0) / 2000 * 1e3
summ['inference_ms'] = [inf.get(m, np.nan) for m in summ.method]
summ.to_csv(dst / 'summary.csv', index=False)
for f in ['curves.json', 'setup.json']:
    if (src / f).exists(): os.system(f'cp "{src / f}" "{dst / f}"')
print('adapted', sorted(set(summ.method)))

rows = []
for m, central in [('D-PPO', False), ('C-PPO', True)]:
    fn = src / 'agents' / f'{m}_seed0.pkl'
    if not fn.exists(): continue
    ag = pickle.load(open(fn, 'rb')); pol = DistPolicy(ag) if not central else CentPolicy(ag)
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        env = TraceResourceEnv(horizon=HORIZON); eps = []
        for st in EVAL_STARTS:
            obs, _ = env.reset(options={'split': 'test', 'start_index': st, 'sla_scale': scale})
            done = False
            while not done:
                a = pol.act(env, obs); _, _, _, done, info = env.step(a)
            raw = env._raw(); served = raw[:, :, 1].sum(0); lat = raw[:, :, 3].sum(0) / np.maximum(served, 1); elig = raw[:, :, 4].sum(0); miss = raw[:, :, 5].sum(0)
            eps.append((lat, 100 * miss / np.maximum(elig, 1)))
        for i, s in enumerate(SLICES):
            rows.append(dict(method=m, scale=scale, slice=s, latency=float(np.mean([e[0][i] for e in eps])), sla_viol=float(np.mean([e[1][i] for e in eps]))))
        print(f'[thresholds] {m} scale={scale}: ' + ' '.join(f"{s}={rows[-3+i]['sla_viol']:.1f}%" for i, s in enumerate(SLICES)), flush=True)
    pd.DataFrame(rows).to_csv(dst / 'extras' / 'threshold_robustness.csv', index=False)
