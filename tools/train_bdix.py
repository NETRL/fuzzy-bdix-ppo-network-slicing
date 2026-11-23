import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

METHODS = ['D-PPO', 'C-PPO', 'D-PPO-crisp', 'D-PPO-nogate', 'D-PPO-noiso', 'D-PPO-euclid']

parser = argparse.ArgumentParser(description='Train a controller through the explicit BDIx agents and, on request, compare the trained parameters bit for bit with those of the campaign driver.')
parser.add_argument('--code', type=Path, required=True, help='Archive code directory containing bdix/, campaign/ and environment/')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--method', required=True, choices=METHODS)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--steps', type=int, default=200000)
parser.add_argument('--compare-driver', action='store_true', help='Also train with campaign/train_campaign_numpy.py and compare the parameters')
parser.add_argument('--compare-checkpoint', type=Path, default=None, help='Compare the trained parameters with an existing checkpoint')
parser.add_argument('--skip-evaluation', action='store_true')
args = parser.parse_args()

root = args.code.resolve()
sys.path.insert(0, str(root))
import bdix
import trace_resource_env

method, seed, steps = args.method, args.seed, args.steps
central = method == 'C-PPO'
plan_gate = method != 'D-PPO-nogate'
trace_resource_env.ARBITER_MODE = 'euclid' if method == 'D-PPO-euclid' else 'guarded'
reward_kind = 'crisp' if method == 'D-PPO-crisp' else 'fuzzy'
isolation_enabled = method != 'D-PPO-noiso'


def parameters(policies):
    policies = policies if isinstance(policies, (list, tuple)) else [policies]
    return [p for policy in policies for net in (policy.actor, policy.critic) for p in net.params]


def digest(policies):
    h = hashlib.sha256()
    for p in parameters(policies):
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()


def identical(a, b):
    pa, pb = parameters(a), parameters(b)
    return len(pa) == len(pb) and all(x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(pa, pb))


if central:
    controller, curve, seconds = bdix.train_centralized(seed, steps, plan_gate=plan_gate)
    policies = controller.policy
else:
    agents, curve, seconds = bdix.train_distributed(seed, steps, reward_kind=reward_kind, isolation_enabled=isolation_enabled, plan_gate=plan_gate)
    policies = [agent.policy for agent in agents]

out = args.out.resolve()
(out / 'agents').mkdir(parents=True, exist_ok=True)
checkpoint = out / 'agents' / f'{method}_seed{seed}.pkl'
with checkpoint.open('wb') as stream:
    pickle.dump(policies, stream)
summary = dict(method=method, seed=seed, steps=steps, trained_through='BDIx agents (code/bdix/training.py)',
               train_seconds=seconds, rollouts=len(curve), parameter_sha256=digest(policies),
               final_reward_weighted=curve[-1]['reward_weighted'] if curve else None, numpy_version=np.__version__)
(out / f'{method}_seed{seed}.curve.json').write_text(json.dumps(curve) + '\n')

if args.compare_driver:
    import train_campaign_numpy as campaign
    if central:
        driver_policies, driver_curve, driver_seconds = campaign.train_centralized(seed, steps, 'PPO')
    else:
        driver_policies, driver_curve, driver_seconds = campaign.train_distributed(seed, steps, reward_kind=reward_kind, iso=isolation_enabled, gate=plan_gate)
    summary.update(driver_parameter_sha256=digest(driver_policies), driver_train_seconds=driver_seconds,
                   parameters_identical_to_driver=identical(policies, driver_policies), curve_identical_to_driver=curve == driver_curve)
    if not (summary['parameters_identical_to_driver'] and summary['curve_identical_to_driver']):
        (out / f'bdix_training_{method}_seed{seed}.json').write_text(json.dumps(summary, indent=2) + '\n')
        raise ValueError('Training through the BDIx agents differs from the campaign driver.')

if args.compare_checkpoint is not None:
    archived = bdix.load_checkpoint(args.compare_checkpoint)
    summary.update(compared_checkpoint=str(args.compare_checkpoint), compared_checkpoint_parameter_sha256=digest(archived),
                   parameters_identical_to_checkpoint=identical(policies, archived))

if not args.skip_evaluation:
    setup = json.loads((root / 'campaign' / 'results_fixed4' / 'setup.json').read_text())
    env = trace_resource_env.TraceResourceEnv(horizon=int(setup['horizon']))
    if not plan_gate:
        env.plans = lambda: ['OPTIMIZE'] * 3
    system = bdix.BDIxSlicingSystem(env, bdix.controller_from_policies(env, policies, plan_gate=plan_gate))
    episodes = [system.run_episode(int(start)) for start in setup['eval_starts']]
    evaluation = {}
    for i, name in enumerate(trace_resource_env.SLICES):
        evaluation[name] = {k: float(np.mean([e[k][i] for e in episodes])) for k in ('latency', 'sla_viol', 'throughput', 'isolation', 'reward')}
    evaluation['msgs_per_s'] = float(np.mean([e['msgs_per_s'] for e in episodes]))
    summary['evaluation'] = evaluation

(out / f'bdix_training_{method}_seed{seed}.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
