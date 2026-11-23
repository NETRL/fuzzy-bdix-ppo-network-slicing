import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ['D-PPO', 'C-PPO', 'D-PPO-crisp', 'D-PPO-nogate', 'D-PPO-noiso', 'D-PPO-euclid', 'D-PPO-Lagrangian']
METRICS = ['latency', 'sla_viol', 'throughput', 'isolation', 'reward', 'reject_rate', 'msgs_per_s']

parser = argparse.ArgumentParser(description='Replay an archived checkpoint through the explicit BDIx agents, validate the per-episode records and the interval log and write the decision statistics.')
parser.add_argument('--code', type=Path, required=True, help='Archive code directory containing bdix/, campaign/ and environment/')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--method', required=True, choices=METHODS)
parser.add_argument('--seed', type=int, required=True, choices=[0, 1, 2])
parser.add_argument('--no-cross-check', action='store_true', help='Skip the per-interval comparison with the decision path of the environment')
parser.add_argument('--trace-intervals', type=int, default=0, help='Write the decision rationale of the first intervals of window 0')
args = parser.parse_args()

root = args.code.resolve()
sys.path.insert(0, str(root))
import bdix
import gymnasium
import trace_resource_env

method, seed = args.method, args.seed
plan_gate = method != 'D-PPO-nogate'
trace_resource_env.ARBITER_MODE = 'euclid' if method == 'D-PPO-euclid' else 'guarded'
saved = root / 'campaign' / 'results_fixed4'
checkpoint = saved / 'agents' / f'{method}_seed{seed}.pkl'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checkpoint_hash = sha(checkpoint)

setup = json.loads((saved / 'setup.json').read_text())
starts = [int(x) for x in setup['eval_starts']]
env = trace_resource_env.TraceResourceEnv(horizon=int(setup['horizon']))
if not plan_gate:
    env.plans = lambda: ['OPTIMIZE'] * 3
controller = bdix.controller_from_policies(env, bdix.load_checkpoint(checkpoint), plan_gate=plan_gate)
system = bdix.BDIxSlicingSystem(env, controller, cross_check=not args.no_cross_check)

reference = pd.read_csv(saved / 'per_episode_records.csv')
reference = reference[(reference.method == method) & (reference.seed == seed)].set_index(['episode', 'slice'])
assert len(reference) == 60 and reference.index.is_unique
archived_log = saved / 'logs' / f'{method}_seed{seed}.csv.gz'
base_reference = pd.read_csv(archived_log) if archived_log.exists() else None

out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
tmp_path = out / f'{method}_seed{seed}.unverified.csv.gz'
worst = 0.
worst_detail = 0.
rows_count = 0
detail_count = 0
episodes = []
messages = collections.Counter()
inter_agent = 0
plan_counts = {name: collections.Counter() for name in trace_resource_env.SLICES}
plan_changes = collections.Counter()
gated = collections.Counter()
trace = [] if args.trace_intervals > 0 else None
with gzip.open(tmp_path, 'wt', encoding='utf8', newline='') as gz:
    for episode, start in enumerate(starts):
        log = []
        values = system.run_episode(start, log=log, trace=trace if episode == 0 else None, trace_intervals=args.trace_intervals)
        for i, name in enumerate(trace_resource_env.SLICES):
            row = reference.loc[(episode, name)]
            assert int(row.start_index) == start
            record = dict(method=method, seed=seed, episode=episode, start_index=start, slice=name)
            for metric in METRICS:
                actual = float(values[metric] if metric == 'msgs_per_s' else values[metric][i])
                err = abs(float(row[metric]) - actual)
                worst = max(worst, err)
                if err > 1e-9:
                    raise ValueError(f'Seed {seed} episode {episode} {name} {metric}: {actual} versus {row[metric]}')
                record[metric] = actual
            episodes.append(record)
        frame = pd.DataFrame(log)
        frame['episode'] = episode
        assert len(frame) == 6000 and not frame.duplicated(['slice', 't', 'episode']).any()
        frame.to_csv(gz, index=False, header=episode == 0)
        rows_count += len(frame)
        if base_reference is not None:
            old = base_reference[base_reference.episode == episode].reset_index(drop=True)
            assert list(old.columns) == list(frame.columns)
            for col in old:
                if pd.api.types.is_numeric_dtype(frame[col]):
                    np.testing.assert_allclose(frame[col], old[col], rtol=0, atol=1e-9)
                    worst_detail = max(worst_detail, float(np.abs(frame[col].to_numpy() - old[col].to_numpy()).max()))
                else:
                    np.testing.assert_array_equal(frame[col], old[col])
            detail_count += len(frame)
        messages.update(values['messages']['by_kind'])
        inter_agent += values['messages']['inter_agent']
        for name, group in frame.groupby('slice'):
            plans = group.sort_values('t').plan.to_numpy()
            plan_counts[name].update(plans.tolist())
            plan_changes[name] += int((plans[1:] != plans[:-1]).sum())
            gated[name] += int((group.action != group.eff_action).sum())
        if episode in [0, 5, 10, 15, 19]:
            print(f'{method} seed {seed}: checked {episode + 1}/20 episodes through the BDIx agents', flush=True)
assert checkpoint_hash == sha(checkpoint)
target = out / f'{method}_seed{seed}.csv.gz'
tmp_path.replace(target)
pd.DataFrame(episodes).to_csv(out / f'{method}_seed{seed}.episodes.csv', index=False)
intervals = len(starts) * env.horizon
summary = dict(
    method=method, seed=seed, architecture='centralized' if controller.central else 'distributed',
    plan_gate=plan_gate, arbiter=trace_resource_env.ARBITER_MODE, episodes=len(starts),
    archived_reference_metric_cells=420, episode_absolute_tolerance=1e-9, max_episode_abs_difference=worst,
    interval_slice_rows=rows_count, archived_reference_interval_rows=detail_count,
    max_archived_interval_abs_difference=worst_detail if base_reference is not None else None,
    cross_checked_intervals=system.cross_checked_intervals, intervals=intervals,
    messages_by_kind={k: int(v) for k, v in messages.items()}, inter_agent_messages=int(inter_agent),
    plan_share_pct={name: {p: 100 * plan_counts[name][p] / intervals for p in bdix.PLAN_ORDER} for name in plan_counts},
    plan_change_rate_pct={name: 100 * plan_changes[name] / (intervals - len(starts)) for name in plan_counts},
    gated_action_share_pct={name: 100 * gated[name] / intervals for name in plan_counts},
    checkpoint_sha256=checkpoint_hash, checkpoint_sha256_after=sha(checkpoint), new_log_sha256=sha(target),
    numpy_version=np.__version__, gymnasium_version=getattr(gymnasium, '__version__', 'unknown'))
(out / f'bdix_replay_{method}_seed{seed}.json').write_text(json.dumps(summary, indent=2) + '\n')
if trace is not None:
    (out / f'{method}_seed{seed}.decision_trace.json').write_text(json.dumps(trace, indent=1) + '\n')
print(f'PASS: {method} seed {seed} through the BDIx agents; largest episode difference {worst:.3g}, '
      f'largest interval difference {worst_detail:.3g}, {system.cross_checked_intervals:,} intervals cross-checked, '
      f'{inter_agent} messages between agents.')
