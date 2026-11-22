import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 'environment')); sys.path.insert(0, str(HERE.parent / 'components'))
from trace_resource_env import TraceResourceEnv, PRB_RATE_MBPS
env = TraceResourceEnv(horizon=2000); rows = env.data[12000:15000]; commit = env.commit
D = np.minimum(rows[:, :, 0], commit[None, :]) / 3.0; need = np.ceil(D / PRB_RATE_MBPS)
tot = need.sum(axis=1); excess = np.maximum(tot - 100, 0)
pd.DataFrame(dict(interval=np.arange(12000, 15000), prbs_eMBB=need[:, 0], prbs_uRLLC=need[:, 1], prbs_mMTC=need[:, 2], prbs_total=tot, excess=excess)).to_csv(HERE / 'results_adapted4' / 'extras' / 'fluid_bound_per_interval.csv', index=False)
summary = dict(intervals=len(tot), intervals_over_capacity=int((tot > 100).sum()), share_over_capacity_pct=float(100 * (tot > 100).mean()), max_total_prbs=int(tot.max()), bound_miss_rate_pct=float(100 * excess.sum() / need.sum()))
pd.DataFrame([summary]).to_csv(HERE / 'results_adapted4' / 'extras' / 'fluid_bound.csv', index=False); print(summary)
