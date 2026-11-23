"""Invariant tests of the explicit BDIx agent layer.

Run with `python tests/test_bdix_invariants.py` or with pytest. The tests need
the packages of the checkpoint replay (NumPy, pandas, openpyxl, Gymnasium) and a
C++17 compiler.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import bdix  # noqa: E402
import fuzzy_system as fz  # noqa: E402
import trace_resource_env as tre  # noqa: E402
from bdix.messages import agent_endpoint  # noqa: E402
from bdix.system import CONTROLLER  # noqa: E402

SLICES = tre.SLICES


class RandomPolicy:
    """Uniformly random stand-in for a PPO policy with one four-way head."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, state, deterministic=False):
        return int(self.rng.integers(0, 4)), 0.0

    def value(self, state):
        return 0.0


def random_system(seed=0, horizon=400, cross_check=True, **reset_options):
    env = tre.TraceResourceEnv(horizon=horizon)
    agents = [bdix.BDIxSliceAgent(config, RandomPolicy(seed * 10 + i)) for i, config in enumerate(bdix.slice_configs(env))]
    system = bdix.BDIxSlicingSystem(env, bdix.DistributedBDIxController(agents), cross_check=cross_check)
    env.reset(seed=seed, options={'split': 'train', **reset_options})
    return env, agents, system


def test_agent_refuses_foreign_and_global_telemetry():
    env, agents, system = random_system()
    reports = bdix.kpi_reports(env)
    for i, agent in enumerate(agents):
        agent.perceive(reports[i])
        for j in range(3):
            if j != i:
                try:
                    agent.perceive(reports[j])
                except bdix.ObservabilityError:
                    pass
                else:
                    raise AssertionError(f'{SLICES[i]} agent accepted the telemetry of {SLICES[j]}')
        for global_view in (env, env.local_observations(), {'allocations': env.allocations}):
            try:
                agent.perceive(global_view)
            except bdix.ObservabilityError:
                pass
            else:
                raise AssertionError('agent accepted a global view of the environment')
    fields = set(bdix.KPIReport.__dataclass_fields__)
    assert not {'allocations', 'last_util', 'raw', 'observations'} & fields


def test_report_reads_only_the_own_slice():
    """Poison every entry of the other slices; the report and the state of slice s must not change."""
    env, agents, system = random_system(seed=3)
    for _ in range(150):
        system.step(deterministic=False)
    row = env._current_row()
    arrays = dict(row=row, raw=env.last_raw.copy(), last_latency=env.last_latency.copy(),
                  timely_fraction=env.timely_fraction.copy(), allocations=env.allocations.copy(), last_iso=env.last_iso.copy())
    trace_index = env.start_index + env.steps
    for s, agent in enumerate(agents):
        clean = bdix.kpi_report_from_arrays(s, budget_prbs=env.budget_prbs, interval=env.steps, trace_index=trace_index, **arrays)
        poisoned = {k: v.copy() for k, v in arrays.items()}
        others = [j for j in range(3) if j != s]
        poisoned['row'][others, :] = np.nan
        poisoned['raw'][:, others, :] = np.nan
        for name in ('last_latency', 'timely_fraction', 'last_iso'):
            poisoned[name][others] = np.nan
        poisoned['allocations'][:, others] = -12345
        dirty = bdix.kpi_report_from_arrays(s, budget_prbs=env.budget_prbs, interval=env.steps, trace_index=trace_index, **poisoned)
        for field in bdix.KPIReport.__dataclass_fields__:
            assert np.array_equal(getattr(clean, field), getattr(dirty, field)), field
        agent.perceive(clean); agent.deliberate(); reference = agent.state_vector().tobytes()
        agent.perceive(dirty); agent.deliberate(); state = agent.state_vector()
        assert np.isfinite(state).all() and state.tobytes() == reference
        assert state.shape == (bdix.STATE_SIZE,) == (14,)


def test_agent_decisions_equal_the_environment_path_under_random_actions():
    """State vectors, plans and grant requests are compared bit for bit in every interval."""
    visited = set()
    for seed, options in [(0, {}), (1, {'traffic_scale': 2.0}), (2, {'sla_scale': 0.8})]:
        env, agents, system = random_system(seed=seed, horizon=400, **options)
        done = False
        while not done:
            committed = np.minimum(env._current_row()[:, 0], env.commit * env.traffic_scale)
            result = system.step(deterministic=False)
            done = result.done
            visited.update((c.plan, c.policy_action) for c in result.commands)
            # the demand admitted by the environment is the share stated by the plan of the agent
            assert all(env.offered[s] == c.admitted_share * committed[s] for s, c in enumerate(result.commands))
            assert [bool(r) for r in env.rejected] == [c.plan == 'REJECT' for c in result.commands]
        assert system.cross_checked_intervals == 400
    assert len(visited) == 16, 'every plan was not met with every action'


def test_capacity_conservation_over_randomized_stress_slots():
    for mode in ('guarded', 'euclid'):
        tre.ARBITER_MODE = mode
        try:
            rng = np.random.default_rng(7)
            for _ in range(1000):
                proposals = rng.integers(0, 140, 3)
                budget = int(rng.integers(6, 101))
                granted, leak = tre.project_integer(proposals, budget, rng.uniform(0, 3, 3))
                assert granted.sum() <= budget and (granted >= tre.N_MIN).all() and (leak >= 0).all()
            env, agents, system = random_system(seed=11, horizon=1000, cross_check=False, traffic_scale=3.0)
            done = False
            while not done:
                result = system.step(deterministic=False)
                done = result.done
                grants = np.asarray(result.info['grants'])
                assert (grants.sum(axis=1) <= env.budget_prbs).all() and (grants >= tre.N_MIN).all()
                assert result.info['capacity_feasible'] and result.info['queue_conservation']
        finally:
            tre.ARBITER_MODE = 'guarded'


def test_fuzzy_rule_coverage_continuity_and_plan_library():
    library = bdix.PlanLibrary(tre.RESIDUAL)
    grid = np.linspace(0, 1, 11)
    fired = np.zeros((3, 3, 3, 3), dtype=bool)
    scores = np.empty((11, 11, 11, 11))
    for idx in itertools.product(range(11), repeat=4):
        x = [grid[i] for i in idx]
        degrees = [fz.membership_degrees(v) for v in x]
        assert all(abs(d.sum() - 1) < 1e-12 for d in degrees)
        fired |= np.einsum('i,j,k,l->ijkl', *degrees) > 0
        scores[idx] = fz.plan_score(*x, False)
    assert fired.all(), 'a rule without fault never fires on the grid'
    assert fz.plan_score(0.3, 0.3, 0.9, 0.5, True) == 0.0 and len(fz.rule_records()) == 162
    assert scores.min() >= 0 and scores.max() <= 100
    for axis, slope in enumerate((30.0, 10.0, 20.0, 20.0)):
        assert np.abs(np.diff(scores, axis=axis)).max() <= slope * 0.1 + 1e-9, 'plan score has a step discontinuity'
    for score in np.round(np.arange(0, 100.001, 0.05), 2):
        plan = library.select(score, False)
        assert plan.name == str(fz.select_plan(score, False)) and plan.score_above < score <= plan.score_up_to
        assert library.select(score, True).name == 'REJECT'
    gate = {(p, a): library.gate(p, a) for p in bdix.PLAN_ORDER for a in range(4)}
    assert all(gate[('OPTIMIZE', a)] == (a, 1) for a in range(4))
    assert all(gate[('DEFER', a)] == (0, 1) for a in range(4))
    assert [gate[('REJECT', a)] for a in range(4)] == [(0, 1), (0, 1), (2, 1), (0, 1)]
    assert [gate[('IMMEDIATE', a)] for a in range(4)] == [(0, 1), (1, 2), (2, 1), (3, 1)]
    assert library['DEFER'].admitted_share == library['REJECT'].admitted_share == tre.RESIDUAL
    assert library['OPTIMIZE'].admitted_share == library['IMMEDIATE'].admitted_share == 1.0


def test_rationale_reproduces_the_plan_score():
    env, agents, system = random_system(seed=5)
    met = set()
    for _ in range(300):
        reports = bdix.kpi_reports(env)
        for agent, report in zip(agents, reports):
            score, fault = bdix.plan_score(report, agent.config.latency_scale_ms)
            why = bdix.rationale(report, agent.config.latency_scale_ms)
            assert abs(why['score'] - score) < 1e-9 and why['plan'] == str(fz.select_plan(score, fault))
            met.add(why['plan'])
        system.step(deterministic=False)
    assert met == set(bdix.PLAN_ORDER), 'every plan must occur in the trace'


def test_no_message_between_agents_and_message_count():
    log = bdix.MessageLog()
    for sender, receiver in [(agent_endpoint('eMBB'), agent_endpoint('uRLLC')), (CONTROLLER, agent_endpoint('mMTC')), (bdix.GNB, bdix.GNB)]:
        try:
            log.send(bdix.GRANT_COMMAND, sender, receiver)
        except bdix.InterAgentMessageError:
            pass
        else:
            raise AssertionError('a message without the gNB at one end was accepted')
    env, agents, system = random_system(seed=2, horizon=300, cross_check=False)
    system.messages.reset()
    expected = 0
    done = False
    while not done:
        before = env.allocations.copy()
        done = system.step(deterministic=False).done
        expected += 3 + int((env.allocations != before).any(axis=0).sum()) + int((env.last_leak.sum(0) > 0).sum())
    summary = system.messages.summary()
    assert summary['total'] == expected and summary['inter_agent'] == 0
    assert summary['by_kind'][bdix.KPI_REPORT] == 3 * 300
    assert all((bdix.GNB + ' ->' in link) != ('-> ' + bdix.GNB in link) for link in summary['by_link'])


def test_agents_share_nothing():
    env, agents, system = random_system(seed=4, cross_check=False)
    for a, b in itertools.combinations(agents, 2):
        assert a.policy is not b.policy and a.buffer is not b.buffer and a.beliefs is not b.beliefs
        assert a.plan_library is not b.plan_library and a._score_cache is not b._score_cache
    reports = bdix.kpi_reports(env)
    alone = [bdix.BDIxSliceAgent(agents[0].config, RandomPolicy(0)).cycle(reports[0], deterministic=False)]
    agents[1].beliefs.values[:] = 99.0
    agents[2].intention = None
    again = bdix.BDIxSliceAgent(agents[0].config, RandomPolicy(0)).cycle(reports[0], deterministic=False)
    assert alone[0] == again
    rewards = []
    for _ in range(5):
        result = system.step(deterministic=False, learning=True)
        rewards.append(np.asarray(result.info['slice_rewards']).copy())
        for i, agent in enumerate(agents):
            agent.remember(result.info['slice_rewards'][i], result.done)
    for i, agent in enumerate(agents):
        assert agent.buffer['r'] == [float(r[i]) for r in rewards] and len(agent.buffer['obs']) == 5


def test_replay_through_agents_matches_the_archived_record():
    import pandas as pd
    saved = ROOT / 'code/campaign/results_fixed4'
    reference = pd.read_csv(saved / 'per_episode_records.csv')
    env = tre.TraceResourceEnv(horizon=2000)
    for method in ('D-PPO', 'C-PPO'):
        controller = bdix.controller_from_policies(env, bdix.load_checkpoint(saved / 'agents' / f'{method}_seed0.pkl'))
        values = bdix.BDIxSlicingSystem(env, controller, cross_check=True).run_episode(12000)
        rows = reference[(reference.method == method) & (reference.seed == 0) & (reference.episode == 0)].set_index('slice')
        for i, name in enumerate(SLICES):
            for metric in ('latency', 'sla_viol', 'throughput', 'isolation', 'reward', 'reject_rate', 'msgs_per_s'):
                actual = values[metric] if metric == 'msgs_per_s' else values[metric][i]
                assert abs(float(rows.loc[name, metric]) - float(actual)) <= 1e-9, (method, name, metric)
        assert values['messages']['inter_agent'] == 0


def test_training_through_agents_equals_the_campaign_driver():
    import train_campaign_numpy as campaign
    steps = 2 * campaign.N_STEPS
    original, curve_o, _ = campaign.train_distributed(3, steps)
    agents, curve_b, _ = bdix.train_distributed(3, steps)
    assert curve_o == curve_b
    for policy, agent in zip(original, agents):
        for net in ('actor', 'critic'):
            for p, q in zip(getattr(policy, net).params, getattr(agent.policy, net).params):
                assert p.tobytes() == q.tobytes()
    central_o, curve_co, _ = campaign.train_centralized(3, steps, 'PPO')
    controller, curve_cb, _ = bdix.train_centralized(3, steps)
    assert curve_co == curve_cb
    for net in ('actor', 'critic'):
        for p, q in zip(getattr(central_o, net).params, getattr(controller.policy, net).params):
            assert p.tobytes() == q.tobytes()


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith('test_') and callable(fn)]
    for name, fn in tests:
        fn()
        print('PASS', name, flush=True)
    print(f'PASS: {len(tests)} invariant tests of the BDIx agent layer.')
