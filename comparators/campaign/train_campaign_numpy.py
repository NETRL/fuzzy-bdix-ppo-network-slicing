from __future__ import annotations
import argparse, json, os, sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'environment')); sys.path.insert(0, str(HERE.parent / 'components'))
from trace_resource_env import TraceResourceEnv, SLICES, PRB_RATE_MBPS, PACKET_BITS, OMEGA
from numpy_agents import PPOAgent, DQNAgent

HORIZON = 2000; N_STEPS = 2048; OBS = 14; GOBS = 42
EVAL_STARTS = [int(x) for x in np.linspace(12000, 13000, 20)]

def local_obs(env):
    return env.local_observations()

def episode_metrics(env):
    raw = env._raw(); served = raw[:, :, 1].sum(0); lat = raw[:, :, 3].sum(0) / np.maximum(served, 1)
    elig = raw[:, :, 4].sum(0); miss = raw[:, :, 5].sum(0)
    tput = served * PACKET_BITS / (env.horizon * 0.01) / 1e6
    return lat, 100 * miss / np.maximum(elig, 1), tput

class Static:
    name = 'static'
    def act(self, env, obs): return [0, 0, 0]
class Rule:
    name = 'rule'
    def act(self, env, obs):
        u = env.last_util.mean(0); return [1 if u[i] > 0.8 else (2 if u[i] < 0.4 else 0) for i in range(3)]
class Random:
    name = 'random'
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def act(self, env, obs): return list(self.rng.integers(0, 4, 3))
class Clairvoyant:

    name = 'clairvoyant'
    def act(self, env, obs):
        row = env._current_row(); dem = np.minimum(row[:, 0], env.commit) / 3.0
        need = np.array([dem[0] / 0.7, dem[1] / 0.3, dem[2] / 0.5]) / PRB_RATE_MBPS
        need = np.maximum(np.ceil(need), 2); b = env._budget(env.steps)
        if need.sum() > b: need[0] = max(2, b - need[1] - need[2])
        if need.sum() > b: need[1] = max(2, b - need[0] - need[2])
        env._clair = need.astype(np.int32)
        return [0, 0, 0]

def run_episode(env, policy, start, central=False, log=None, mode_central=False):
    obs, info = env.reset(options={'split': 'test' if start >= 12000 else 'train', 'start_index': start}); done = False; t = 0
    msgs = 0.0; rew = np.zeros(3); rej = np.zeros(3); n_rej = 0
    while not done:
        a = policy.act(env, obs)
        if isinstance(policy, Clairvoyant):
            g = np.tile(env._clair, (3, 1)); before = env.allocations.copy()
            _, r, _, done, info = env.step_allocations(g, proposals=g, actions=np.zeros(3, dtype=int), plans=env.plans())
            changed = int((g != before).any())
        else:
            before = env.allocations.copy(); _, r, _, done, info = env.step(a)
            changed = (env.allocations != before).any(axis=0).sum()

        n_arb = int((env.last_leak.sum(0) > 0).sum())
        msgs += 3 + (1 if (mode_central and changed > 0) else changed) + n_arb
        rew += info['canonical_slice_rewards']; rej += env.rejected
        if log is not None:
            for k, s in enumerate(SLICES):
                log.append(dict(slice=s, t=t, plan=env.last_plans[k], action=int(a[k]), eff_action=int(env.last_effective[k]),
                                N=float(env.allocations[:, k].mean()), u=float(env.last_util[:, k].mean()), offered=float(env.offered[k]),
                                L=float(env.last_latency[k]), miss=float(info['eligible_misses'][k]),
                                elig=float(info['eligible_arrivals'][k]),
                                T=float(sum(info['served_bits'][c][k] for c in range(3)) / 0.01 / 1e6), I=float(env.last_iso[k]),
                                rejected=int(env.rejected[k]), reward=float(info['canonical_slice_rewards'][k]), lat_target=float(info['requested_latency_ms'][k])))
        t += 1
    lat, missp, tput = episode_metrics(env)
    return dict(latency=lat, sla_viol=missp, throughput=tput, isolation=env.isolation_sum / env.horizon,
                msgs_per_s=msgs / (env.horizon * 0.01), reward=rew / env.horizon, reject_rate=100 * rej / env.horizon)


def train_distributed(seed, steps, variant='D-PPO', reward_kind='fuzzy', iso=True, gate=True, hp=None):
    hp = {**dict(lr=3e-4, epochs=20, minibatch=64, ent_coef=0.01), **(hp or {})}
    env = TraceResourceEnv(reward_kind=reward_kind, isolation_enabled=iso, horizon=HORIZON)
    if not gate: env.plans = lambda: ['OPTIMIZE'] * 3
    mappo = variant == 'MAPPO'; lagr = variant == 'Lagrangian'
    if mappo:
        shared = PPOAgent(OBS, 4, 1, seed=seed, critic_obs_dim=GOBS + 3, lr=5e-4, epochs=15, minibatch=3 * N_STEPS, max_norm=10.0, huber_delta=10.0, value_norm=True); agents = [shared] * 3
        hp = {**hp, 'lr': 5e-4, 'decay': False}
    elif lagr:
        agents = [PPOAgent(OBS, 4, 1, seed=seed * 100 + i, lr=3e-4, lr_v=1e-3, epochs=80, minibatch=N_STEPS, kl_target=0.01, ent_coef=0.0) for i in range(3)]
        from numpy_agents import MLP, Adam
        cost_critics = [MLP([OBS, 64, 64, 1], np.random.default_rng(seed * 100 + i + 7), out_gain=1.0) for i in range(3)]
        cost_opts = [Adam(c.params, lr=1e-3, max_norm=None) for c in cost_critics]
        penalty_param = np.zeros(3) + np.log(np.e - 1.0)
        hp = {**hp, 'decay': False}
    else:
        agents = [PPOAgent(OBS, 4, 1, seed=seed * 100 + i, lr=hp['lr'], epochs=hp['epochs'], minibatch=hp['minibatch'], ent_coef=hp['ent_coef']) for i in range(3)]
    lam = np.full(3, 1.0); cost_limit = 0.025; penalty_lr = 0.05
    obs, _ = env.reset(seed=seed, options={'split': 'train'}); lo = local_obs(env)
    bufs = [dict(obs=[], cobs=[], a=[], logp=[], r=[], v=[], d=[], c=[]) for _ in range(3)]
    curve = []; t0 = time.time(); n = 0
    while n < steps:
        acts, lps, vs, cobs = [], [], [], []
        g = lo.reshape(-1)
        for i in range(3):
            a, lp = agents[i].act(lo[i]); cob = np.concatenate([g, np.eye(3)[i]]) if mappo else lo[i]
            acts.append(a); lps.append(lp); vs.append(agents[i].value(cob)); cobs.append(cob)
        _, _, _, done, info = env.step(acts); lo2 = local_obs(env)
        for i in range(3):
            cost = float(info['constraint_costs'][i]); r = float(info['slice_rewards'][i])
            b = bufs[i]; b['obs'].append(lo[i]); b['cobs'].append(cobs[i]); b['a'].append(acts[i]); b['logp'].append(lps[i]); b['r'].append(r); b['v'].append(vs[i]); b['d'].append(float(done)); b['c'].append(cost)
        if done:
            env.reset(options={'split': 'train'}); lo2 = local_obs(env)
        lo = lo2; n += 1
        if n % N_STEPS == 0:
            frac = max(0.0, 1.0 - n / steps)
            if hp.get('decay', True):
                for ag in set(agents): ag.opt_a.lr = hp['lr'] * frac; ag.opt_c.lr = (hp.get('lr_v') or hp['lr']) * frac
            rec = dict(step=n); g = lo.reshape(-1)
            if mappo: pooled = dict(obs=[], cobs=[], a=[], logp=[], adv=[], ret=[])
            for i in range(3):
                b = bufs[i]; last_v = agents[i].value(np.concatenate([g, np.eye(3)[i]]) if mappo else lo[i])
                adv, ret = PPOAgent.gae(np.array(b['r']), np.array(b['v']), np.array(b['d']), last_v)
                if lagr:
                    cv = cost_critics[i].forward(np.array(b['cobs']))[0][:, 0]
                    cadv, cret = PPOAgent.gae(np.array(b['c']), cv, np.array(b['d']), float(cost_critics[i].forward(lo[i])[0][0, 0]))
                    adv = (adv - lam[i] * cadv) / (1.0 + lam[i])
                    for _ in range(80):
                        vv, vc = cost_critics[i].forward(np.array(b['cobs'])); cost_opts[i].step(cost_critics[i].backward(vc, (vv - cret[:, None]) / len(cret)))
                if mappo:
                    for k, v in [('obs', b['obs']), ('cobs', b['cobs']), ('a', b['a']), ('logp', b['logp']), ('adv', adv), ('ret', ret)]: pooled[k].append(np.array(v))
                else:
                    agents[i].update(np.array(b['obs']), np.array(b['cobs']), np.array(b['a']), np.array(b['logp']), adv, ret)
                rec[f'reward_{SLICES[i]}'] = float(np.mean(b['r']))
                if lagr:
                    penalty_param[i] += penalty_lr * (np.mean(b['c']) - cost_limit)
                    lam[i] = float(np.log1p(np.exp(penalty_param[i]))); rec[f'lambda_{SLICES[i]}'] = lam[i]
                bufs[i] = dict(obs=[], cobs=[], a=[], logp=[], r=[], v=[], d=[], c=[])
            if mappo: shared.update(*[np.concatenate(pooled[k]) for k in ['obs', 'cobs', 'a', 'logp', 'adv', 'ret']])
            rec['reward_weighted'] = float(0.3 * rec['reward_eMBB'] + 0.5 * rec['reward_uRLLC'] + 0.2 * rec['reward_mMTC']); curve.append(rec)
    return agents, curve, time.time() - t0

def train_centralized(seed, steps, algo='PPO'):
    env = TraceResourceEnv(horizon=HORIZON); env.reset(seed=seed, options={'split': 'train'}); g = env.local_observations().reshape(-1)
    curve = []; t0 = time.time(); n = 0
    if algo == 'PPO':
        agent = PPOAgent(GOBS, 4, 3, seed=seed, epochs=20, minibatch=64); b = dict(obs=[], a=[], logp=[], r=[], v=[], d=[])
        while n < steps:
            a, lp = agent.act(g); v = agent.value(g); _, _, _, done, info = env.step(list(a))
            r = float(np.dot(OMEGA, info['slice_rewards'])); b['obs'].append(g); b['a'].append(a); b['logp'].append(lp); b['r'].append(r); b['v'].append(v); b['d'].append(float(done))
            if done: env.reset(options={'split': 'train'})
            g = env.local_observations().reshape(-1); n += 1
            if n % N_STEPS == 0:
                agent.opt_a.lr = 3e-4 * max(0.0, 1.0 - n / steps); agent.opt_c.lr = agent.opt_a.lr
                adv, ret = PPOAgent.gae(np.array(b['r']), np.array(b['v']), np.array(b['d']), agent.value(g))
                agent.update(np.array(b['obs']), np.array(b['obs']), np.array(b['a']), np.array(b['logp']), adv, ret)
                curve.append(dict(step=n, reward_weighted=float(np.mean(b['r'])))); b = dict(obs=[], a=[], logp=[], r=[], v=[], d=[])
    elif algo == 'Lagrangian':
        from numpy_agents import MLP, Adam
        agent = PPOAgent(GOBS, 4, 3, seed=seed, lr=3e-4, lr_v=1e-3, epochs=80, minibatch=N_STEPS, kl_target=0.01, ent_coef=0.0)
        cost_critic = MLP([GOBS, 64, 64, 1], np.random.default_rng(seed + 7), out_gain=1.0); cost_opt = Adam(cost_critic.params, lr=1e-3, max_norm=None)
        penalty_param = float(np.log(np.e - 1.0)); lam = 1.0; cost_limit = 0.025; penalty_lr = 0.05
        b = dict(obs=[], a=[], logp=[], r=[], v=[], d=[], c=[])
        while n < steps:
            a, lp = agent.act(g); v = agent.value(g); _, _, _, done, info = env.step(list(a))
            r = float(np.dot(OMEGA, info['slice_rewards'])); c = float(np.mean(info['constraint_costs']))
            b['obs'].append(g); b['a'].append(a); b['logp'].append(lp); b['r'].append(r); b['v'].append(v); b['d'].append(float(done)); b['c'].append(c)
            if done: env.reset(options={'split': 'train'})
            g = env.local_observations().reshape(-1); n += 1
            if n % N_STEPS == 0:
                adv, ret = PPOAgent.gae(np.array(b['r']), np.array(b['v']), np.array(b['d']), agent.value(g))
                cv = cost_critic.forward(np.array(b['obs']))[0][:, 0]
                cadv, cret = PPOAgent.gae(np.array(b['c']), cv, np.array(b['d']), float(cost_critic.forward(g)[0][0, 0]))
                adv = (adv - lam * cadv) / (1.0 + lam)
                for _ in range(80):
                    vv, vc = cost_critic.forward(np.array(b['obs'])); cost_opt.step(cost_critic.backward(vc, (vv - cret[:, None]) / len(cret)))
                agent.update(np.array(b['obs']), np.array(b['obs']), np.array(b['a']), np.array(b['logp']), adv, ret)
                penalty_param += penalty_lr * (float(np.mean(b['c'])) - cost_limit); lam = float(np.log1p(np.exp(penalty_param)))
                curve.append(dict(step=n, reward_weighted=float(np.mean(b['r'])), cost=float(np.mean(b['c'])), lam=lam)); b = dict(obs=[], a=[], logp=[], r=[], v=[], d=[], c=[])
    else:
        agent = DQNAgent(GOBS, 64, seed=seed); rs = []
        while n < steps:
            a = agent.act(g, steps); joint = np.unravel_index(a, (4, 4, 4)); _, _, _, done, info = env.step([int(x) for x in joint])
            r = float(np.dot(OMEGA, info['slice_rewards']))
            if done: env.reset(options={'split': 'train'})
            g2 = env.local_observations().reshape(-1); agent.store(g, a, r, g2, float(done)); rs.append(r); g = g2; n += 1
            if n % N_STEPS == 0: curve.append(dict(step=n, reward_weighted=float(np.mean(rs)))); rs = []
    return agent, curve, time.time() - t0

class DistPolicy:
    def __init__(self, agents): self.agents = agents
    def act(self, env, obs):
        lo = env.local_observations(); return [self.agents[i].act(lo[i], deterministic=True)[0] for i in range(3)]
class CentPolicy:
    def __init__(self, agent, algo='PPO'): self.agent, self.algo = agent, algo
    def act(self, env, obs):
        g = env.local_observations().reshape(-1)
        if self.algo == 'PPO': a, _ = self.agent.act(g, deterministic=True); return [int(x) for x in a]
        return [int(x) for x in np.unravel_index(self.agent.act(g, deterministic=True), (4, 4, 4))]

VAL_STARTS = [int(x) for x in np.linspace(8000, 10000, 10)]

def evaluate(policy, central=False, gate=True, keep_log=False, seed=0, split='test'):
    env = TraceResourceEnv(horizon=HORIZON)
    if not gate: env.plans = lambda: ['OPTIMIZE'] * 3
    eps, logs = [], []
    for ep, st in enumerate(EVAL_STARTS if split == 'test' else VAL_STARTS):
        log = [] if keep_log else None
        eps.append(run_episode(env, policy, st, log=log, mode_central=central))
        if keep_log:
            d = pd.DataFrame(log); d['episode'] = ep; logs.append(d)
    agg = {s: {k: float(np.mean([e[k][i] for e in eps])) for k in ['latency', 'sla_viol', 'throughput', 'isolation', 'reward', 'reject_rate']} for i, s in enumerate(SLICES)}
    for i, s in enumerate(SLICES): agg[s]['sla_viol_sd'] = float(np.std([e['sla_viol'][i] for e in eps]))
    agg['msgs_per_s'] = float(np.mean([e['msgs_per_s'] for e in eps]))
    return agg, (pd.concat(logs, ignore_index=True) if keep_log else None)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='campaign/results_fixed'); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--steps-d', type=int, default=200000); ap.add_argument('--steps-c', type=int, default=200000)
    ap.add_argument('--methods', nargs='+', default=['static', 'rule', 'random', 'clairvoyant', 'D-PPO', 'C-PPO', 'MAPPO', 'D-PPO-Lagrangian', 'DQN', 'D-PPO-crisp', 'D-PPO-nogate', 'D-PPO-noiso'])
    args = ap.parse_args(); out = Path(args.out); (out / 'logs').mkdir(parents=True, exist_ok=True); (out / 'agents').mkdir(exist_ok=True)
    rows = pd.read_csv(out / 'summary.csv').to_dict('records') if (out / 'summary.csv').exists() else []
    curves = json.load(open(out / 'curves.json')) if (out / 'curves.json').exists() else {}
    done = {(r['method'], int(r['seed'])) for r in rows}
    env0 = TraceResourceEnv(horizon=HORIZON); env0.reset(options={'split': 'train'})
    json.dump(dict(configuration=env0.configuration(), preprocessing=env0.preprocessing, horizon=HORIZON, rollout=N_STEPS, eval_starts=EVAL_STARTS,
                   steps_distributed=args.steps_d, steps_centralized=args.steps_c, seeds=args.seeds), open(out / 'setup.json', 'w'), indent=1, default=str)
    def record(method, seed, agg, extra=None):
        for s in SLICES:
            row = dict(method=method, seed=seed, slice=s, **agg[s], msgs_per_s=agg['msgs_per_s']); row.update(extra or {}); rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
        print(f"[{time.strftime('%H:%M:%S')}] {method} seed={seed}: " + ' | '.join(f"{s}: lat={agg[s]['latency']:.2f} viol={agg[s]['sla_viol']:.1f}% tput={agg[s]['throughput']:.1f} iso={agg[s]['isolation']:.3f}" for s in SLICES) + f" | msgs/s={agg['msgs_per_s']:.1f}", flush=True)
    for m in args.methods:
        if m in ('static', 'rule', 'random', 'clairvoyant'):
            if (m, 0) in done: continue
            pol = {'static': Static(), 'rule': Rule(), 'random': Random(0), 'clairvoyant': Clairvoyant()}[m]
            agg, log = evaluate(pol, central=(m == 'clairvoyant'), keep_log=True); log.to_csv(out / 'logs' / f'{m}_seed0.csv.gz', index=False); record(m, 0, agg); continue
        for seed in args.seeds:
            if (m, seed) in done: continue
            t0 = time.time()
            if m == 'C-PPO': agent, curve, tt = train_centralized(seed, args.steps_c, 'PPO'); pol, central, gate = CentPolicy(agent), True, True; npar = agent.n_params()
            elif m == 'DQN': agent, curve, tt = train_centralized(seed, args.steps_c, 'DQN'); pol, central, gate = CentPolicy(agent, 'DQN'), True, True; npar = agent.n_params()
            elif m == 'PPO-Lagrangian': agent, curve, tt = train_centralized(seed, args.steps_c, 'Lagrangian'); pol, central, gate = CentPolicy(agent), True, True; npar = agent.n_params()
            else:
                kw = dict(variant='D-PPO', reward_kind='fuzzy', iso=True, gate=True)
                if m == 'MAPPO': kw['variant'] = 'MAPPO'
                if m == 'D-PPO-Lagrangian': kw['variant'] = 'Lagrangian'
                if m == 'D-PPO-crisp': kw['reward_kind'] = 'crisp'
                if m == 'D-PPO-nogate': kw['gate'] = False
                if m == 'D-PPO-noiso': kw['iso'] = False
                agents, curve, tt = train_distributed(seed, args.steps_d, **kw); pol, central, gate = DistPolicy(agents), False, kw['gate']
                npar = agents[0].n_params() if m == 'MAPPO' else sum(a.n_params() for a in agents); agent = agents
            agg, log = evaluate(pol, central=central, gate=gate, keep_log=(seed == args.seeds[0]))
            if log is not None: log.to_csv(out / 'logs' / f'{m}_seed{seed}.csv.gz', index=False)
            curves[f'{m}_seed{seed}'] = curve; json.dump(curves, open(out / 'curves.json', 'w'))
            pickle.dump(agent, open(out / 'agents' / f'{m}_seed{seed}.pkl', 'wb'))
            record(m, seed, agg, dict(train_seconds=tt, n_params=npar, total_seconds=time.time() - t0))

if __name__ == '__main__':
    main()
