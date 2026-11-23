"""Closed control loop of the BDIx slicing system.

In every control interval the gNB reports to each slice its own KPIs and its
request, the agents decide, the gNB applies the grant update and the PRB
arbiter and the packet service of the interval produces the next KPIs. The
grant update, the arbiter and the packet service are the archived ones of
environment/trace_resource_env.py; they are called here with the plans and the
effective actions decided by the agents.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import trace_resource_env as tre

from .agent import BDIxSliceAgent, CentralizedBDIxController, DistributedBDIxController, SliceConfig
from .beliefs import KPIReport
from .messages import ARBITRATION_EVENT, GNB, GRANT_COMMAND, KPI_REPORT, MessageLog, agent_endpoint

SLICES = tre.SLICES
CONTROLLER = "controller"


class EquivalenceError(AssertionError):
    """Raised when a decision of the agents differs from the archived decision path."""


def kpi_report_from_arrays(s, row, raw, last_latency, timely_fraction, allocations, last_iso,
                           budget_prbs, interval, trace_index):
    """KPI report of slice s. Only entries of slice s are read from the arrays."""
    return KPIReport(
        slice_name=SLICES[s], interval=int(interval), trace_index=int(trace_index),
        requested_rate_mbps=row[s, 0], requested_latency_ms=row[s, 1], requested_reliability_pct=row[s, 2],
        fault_flag=row[s, 3], isolation_level=row[s, 4], background_load=row[s, 5],
        backlog_packets=raw[:, s, 2], oldest_age_ms=raw[:, s, 7],
        served_mean_delay_ms=last_latency[s], timely_fraction=timely_fraction[s],
        granted_prbs=allocations[:, s], isolation_index=last_iso[s], cell_budget_prbs=budget_prbs)


def kpi_reports(env):
    """The three KPI reports of the current control interval, one per slice."""
    row = env._current_row()
    trace_index = min(env.start_index + env.steps, env.start_index + env.horizon - 1)
    return [kpi_report_from_arrays(s, row, env.last_raw, env.last_latency, env.timely_fraction, env.allocations,
                                   env.last_iso, env.budget_prbs, env.steps, trace_index)
            for s in range(len(SLICES))]


def slice_configs(env, plan_gate=True):
    """Static configuration of the three slice agents for an environment."""
    return [SliceConfig(slice_name=name, latency_scale_ms=env.latency_scale[s], prb_rate_mbps=tre.PRB_RATE_MBPS,
                        n_cells=3, delta_prb=tre.DELTA_PRB, residual_share=tre.RESIDUAL, plan_gate=plan_gate)
            for s, name in enumerate(SLICES)]


def controller_from_policies(env, policies, plan_gate=True):
    """Wrap archived PPO policies in BDIx agents (a list of three) or in the centralized controller."""
    configs = slice_configs(env, plan_gate)
    if isinstance(policies, (list, tuple)):
        if len(policies) != len(SLICES):
            raise ValueError("one policy per slice is required")
        return DistributedBDIxController([BDIxSliceAgent(c, p) for c, p in zip(configs, policies)])
    return CentralizedBDIxController(configs, policies)


@dataclass
class StepResult:
    commands: list
    info: dict
    reward: float
    done: bool


class BDIxSlicingSystem:
    """BDIx agents, gNB grant update and arbiter and the packet service in one loop."""

    def __init__(self, env, controller, cross_check=False):
        self.env = env
        self.controller = controller
        self.cross_check = bool(cross_check)
        self.messages = MessageLog()
        self.cross_checked_intervals = 0

    # ------------------------------------------------------------------ one control interval
    def step(self, deterministic=True, learning=False, trace=None):
        env = self.env
        central = self.controller.central
        reports = kpi_reports(env)
        for report in reports:
            self.messages.send(KPI_REPORT, GNB, CONTROLLER if central else agent_endpoint(report.slice_name))
        commands = self.controller.decide(reports, deterministic=deterministic, learning=learning)
        plans = [c.plan for c in commands]
        policy_actions = np.array([c.policy_action for c in commands], dtype=int)
        effective = np.array([c.effective_action for c in commands], dtype=int)
        before = env.allocations.copy()

        # gNB: grant update of every slice with the donor rule of a reallocation
        proposals, leak, applied = env.propose(effective, plans)
        if not np.array_equal(applied, effective):
            raise EquivalenceError("The gNB applied an action that differs from the command of the agent.")
        if self.cross_check:
            self._cross_check(policy_actions, plans, proposals, leak, applied)
        if trace is not None:
            for state, command in zip(self.controller.states, commands):
                trace.append({**state.explain(), "policy_action": command.policy_action,
                              "effective_action": command.effective_action, "step_prbs": command.step_prbs,
                              "admitted_share": command.admitted_share})
        # gNB: the arbiter projects the requested grants onto the budget of each cell
        budget = env._budget(env.steps)
        grants = np.zeros((3, 3), dtype=np.int32)
        for cell in range(3):
            granted, reclaimed = tre.project_integer(proposals[cell], budget, env.last_util[cell])
            grants[cell] = granted
            leak[cell] += reclaimed
        # packet service of the interval, admission by the plans of the agents
        _, reward, _, done, info = env.step_allocations(grants, proposals=proposals, actions=policy_actions,
                                                        plans=plans, leakage=leak, effective=applied)
        changed = (env.allocations != before).any(axis=0)
        reclaimed_from = env.last_leak.sum(0) > 0
        if central:
            if changed.any():
                self.messages.send(GRANT_COMMAND, CONTROLLER, GNB)
        else:
            for s in np.flatnonzero(changed):
                self.messages.send(GRANT_COMMAND, agent_endpoint(SLICES[s]), GNB)
        for s in np.flatnonzero(reclaimed_from):
            self.messages.send(ARBITRATION_EVENT, GNB, CONTROLLER if central else agent_endpoint(SLICES[s]))
        return StepResult(commands=commands, info=info, reward=reward, done=done)

    def _cross_check(self, policy_actions, plans, proposals, leak, applied):
        """Compare the decision of the agents with the archived decision path of the environment."""
        env = self.env
        native_plans = [str(p) for p in env.plans()]
        if native_plans != [str(p) for p in plans]:
            raise EquivalenceError(f"plans differ at interval {env.steps}: agents {plans}, environment {native_plans}")
        native_state = env.local_observations()
        agent_state = np.stack([s.state_vector() for s in self.controller.states])
        if native_state.dtype != agent_state.dtype or native_state.tobytes() != agent_state.tobytes():
            raise EquivalenceError(f"BDIx state vectors differ from the environment observation at interval {env.steps}")
        native_proposals, native_leak, native_applied = env.propose(policy_actions, native_plans)
        if not (np.array_equal(native_proposals, proposals) and np.array_equal(native_leak, leak)
                and np.array_equal(native_applied, applied)):
            raise EquivalenceError(f"grant requests differ from the environment gate at interval {env.steps}")
        self.cross_checked_intervals += 1

    # ------------------------------------------------------------------ one evaluation episode
    def run_episode(self, start, log=None, trace=None, trace_intervals=0):
        """Replay one episode with greedy policies and return its KPIs."""
        env = self.env
        env.reset(options={"split": "test" if start >= 12000 else "train", "start_index": int(start)})
        self.messages.reset()
        done = False
        t = 0
        reward_sum = np.zeros(3)
        rejections = np.zeros(3)
        while not done:
            result = self.step(deterministic=True, trace=trace if (trace is not None and t < trace_intervals) else None)
            info, done = result.info, result.done
            reward_sum += info["canonical_slice_rewards"]
            rejections += env.rejected
            if log is not None:
                for k, name in enumerate(SLICES):
                    log.append(dict(
                        slice=name, t=t, plan=env.last_plans[k], action=int(result.commands[k].policy_action),
                        eff_action=int(env.last_effective[k]), N=float(env.allocations[:, k].mean()),
                        u=float(env.last_util[:, k].mean()), offered=float(env.offered[k]), L=float(env.last_latency[k]),
                        miss=float(info["eligible_misses"][k]), elig=float(info["eligible_arrivals"][k]),
                        T=float(sum(info["served_bits"][c][k] for c in range(3)) / 0.01 / 1e6), I=float(env.last_iso[k]),
                        rejected=int(env.rejected[k]), reward=float(info["canonical_slice_rewards"][k]),
                        lat_target=float(info["requested_latency_ms"][k])))
            t += 1
        raw = env._raw()
        served = raw[:, :, 1].sum(0)
        latency = raw[:, :, 3].sum(0) / np.maximum(served, 1)
        eligible = raw[:, :, 4].sum(0)
        missed = raw[:, :, 5].sum(0)
        throughput = served * tre.PACKET_BITS / (env.horizon * 0.01) / 1e6
        return dict(latency=latency, sla_viol=100 * missed / np.maximum(eligible, 1), throughput=throughput,
                    isolation=env.isolation_sum / env.horizon, msgs_per_s=self.messages.total / (env.horizon * 0.01),
                    reward=reward_sum / env.horizon, reject_rate=100 * rejections / env.horizon,
                    messages=self.messages.summary())
