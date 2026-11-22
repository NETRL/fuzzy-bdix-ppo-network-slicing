import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Replay the archived Euclidean checkpoints, validate per-episode records and write detailed logs.')
parser.add_argument('--code', type=Path, required=True, help='Archive code directory containing campaign/ and environment/')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--seed', type=int, required=True, choices=[0, 1, 2])
args = parser.parse_args()
root = args.code.resolve()
campaign_dir = root / 'campaign'
sys.path.insert(0, str(campaign_dir))
import train_campaign_numpy as campaign
import trace_resource_env
import gymnasium
seed = args.seed
trace_resource_env.ARBITER_MODE = 'euclid'
saved = campaign_dir / 'results_fixed4'
checkpoint = saved / 'agents' / f'D-PPO-euclid_seed{seed}.pkl'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checkpoint_hash = sha(checkpoint)

class CheckpointReader(pickle.Unpickler):
    def find_class(self, module, name):
        trusted = {'numpy_agents': {'PPOAgent', 'MLP', 'Adam'}, 'numpy.random._pickle': {'__generator_ctor','__bit_generator_ctor'},
                   'numpy.random._pcg64': {'PCG64'}, 'numpy': {'ndarray','dtype'}, 'numpy.core.multiarray': {'_reconstruct','scalar'},
                   'numpy._core.multiarray': {'_reconstruct','scalar'}, 'numpy._core.numeric': {'_frombuffer'}, 'numpy.random.bit_generator': {'SeedSequence','__pyx_unpickle_SeedSequence'}}
        if name not in trusted.get(module, set()):
            raise ValueError(f'Unexpected serialized object {module}.{name}')
        return super().find_class(module,name)
with checkpoint.open('rb') as stream:
    agents = CheckpointReader(stream).load()
policy = campaign.DistPolicy(agents)
reference = pd.read_csv(saved / 'per_episode_records.csv')
reference = reference[(reference.method=='D-PPO-euclid') & (reference.seed==seed)].set_index(['episode','slice'])
assert len(reference) == 60 and reference.index.is_unique
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
rows_count = 0
worst = 0.
worst_detail = 0.
env = campaign.TraceResourceEnv(horizon=campaign.HORIZON)
base_reference = pd.read_csv(saved/'logs'/'D-PPO-euclid_seed0.csv.gz') if seed == 0 else None
tmp_path = out / f'D-PPO-euclid_seed{seed}.unverified.csv.gz'
detail_count = 0
with gzip.open(tmp_path,'wt',encoding='utf8',newline='') as gz:
    for episode,start in enumerate(campaign.EVAL_STARTS):
        log = []
        values = campaign.run_episode(env, policy, int(start), log=log, mode_central=False)
        for i, name in enumerate(campaign.SLICES):
            row=reference.loc[(episode,name)]
            assert int(row.start_index) == int(start)
            for metric in ['latency','sla_viol','throughput','isolation','reward','reject_rate','msgs_per_s']:
                actual = values[metric] if metric=='msgs_per_s' else values[metric][i]
                err=abs(float(row[metric])-float(actual))
                worst=max(worst,err)
                if err>1e-9:
                    raise ValueError(f'Seed {seed} episode {episode} {name} {metric}: {actual} versus {row[metric]}')
        frame = pd.DataFrame(log);frame['episode']=episode
        assert len(frame)==6000 and not frame.duplicated(['slice','t','episode']).any()
        assert set(frame.t)==set(range(2000)) and set(frame.slice)==set(campaign.SLICES)
        frame.to_csv(gz,index=False,header=episode==0)
        rows_count+=len(frame)
        if base_reference is not None:
            old=base_reference[base_reference.episode==episode].reset_index(drop=True)
            assert list(old.columns)==list(frame.columns)
            for col in old:
                if pd.api.types.is_numeric_dtype(frame[col]):
                    np.testing.assert_allclose(frame[col],old[col],rtol=0,atol=1e-9)
                    worst_detail=max(worst_detail,float(np.abs(frame[col].to_numpy()-old[col].to_numpy()).max()))
                else:
                    np.testing.assert_array_equal(frame[col],old[col])
            detail_count+=len(frame)
        if episode in [0,5,10,15,19]:
            print(f'Seed {seed}: checked {episode+1}/20 episodes',flush=True)
assert checkpoint_hash==sha(checkpoint)
target=out / f'D-PPO-euclid_seed{seed}.csv.gz'
tmp_path.replace(target)
m=dict(method='D-PPO-euclid',seed=seed,episodes=20,archived_reference_metric_cells=420,
       interval_slice_rows=rows_count,episode_absolute_tolerance=1e-9,max_episode_abs_difference=worst,
       checkpoint_sha256=checkpoint_hash,checkpoint_sha256_after=sha(checkpoint),
       new_log_sha256=sha(target),gymnasium_version=getattr(gymnasium,'__version__','unknown'),
       three_agent_value_loss_coefficients=[a.vf_coef for a in agents],huber_values=[a.huber_delta for a in agents],
       value_normalization=[a.value_norm for a in agents])
if seed==0:m.update(seed_0_reference_log_rows=detail_count,max_seed_0_interval_abs_difference=worst_detail)
(out / f'euclidean_replay_seed{seed}.json').write_text(json.dumps(m,indent=2)+'\n')
