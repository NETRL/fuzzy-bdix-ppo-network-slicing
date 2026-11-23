"""BDIx slice agents of the Fuzzy-BDIx-PPO controller.

BDIxSliceState holds the beliefs, the desires and the intention of one slice
and runs the plan layer. BDIxSliceAgent adds the PPO policy and the private
rollout buffer of the slice, so one agent per slice perceives its own KPI
report, deliberates, selects an action and learns without receiving anything
from the other agents. CentralizedBDIxController is the architectural variant
of the evaluation: one controller holds the BDIx state of the three slices and
one joint policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beliefs import BELIEF_NAMES, BeliefBase, KPIReport, ObservabilityError
from .desires import DesireSet
from .plans import PLAN_ORDER, Intention, PlanLibrary, plan_score, rationale

STATE_SIZE = len(BELIEF_NAMES) + len(PLAN_ORDER)   # 14 inputs per slice


@dataclass(frozen=True)
class SliceConfig:
    """Static configuration of one slice agent."""

    slice_name: str
    latency_scale_ms: float      # training 95th percentile of the requested latency of the slice
    prb_rate_mbps: float         # rate of one PRB
    n_cells: int = 3
    delta_prb: int = 5           # step of a grant change in PRBs
    residual_share: float = 0.3  # share of the committed demand offered under DEFER and REJECT
    plan_gate: bool = True       # False reproduces the variant without the BDIx plan gate


@dataclass(frozen=True)
class GrantCommand:
    """Decision of one slice for one control interval, sent to the gNB."""

    slice_name: str
    interval: int
    plan: str
    plan_score: float
    policy_action: int
    effective_action: int
    step_prbs: int
    admitted_share: float


class BDIxSliceState:
    """Beliefs, desires and intention of one slice with its plan layer."""

    def __init__(self, config):
        self.config = config
        self.slice_name = config.slice_name
        self.beliefs = BeliefBase(config.slice_name, config.latency_scale_ms, config.prb_rate_mbps, config.n_cells)
        self.desires = None
        self.intention = None
        self.plan_library = PlanLibrary(config.residual_share)
        self._score_cache = {}

    def perceive(self, report):
        """Update the desires from the request and revise the beliefs from the KPI report."""
        if not isinstance(report, KPIReport):
            raise ObservabilityError("A slice agent perceives the KPI report of its own slice only.")
        if report.slice_name != self.slice_name:
            raise ObservabilityError(f"The {self.slice_name} agent was given the telemetry of {report.slice_name}.")
        desires = DesireSet.from_report(report)
        self.beliefs.revise(report, desires)
        self.desires = desires
        return self.beliefs

    def deliberate(self):
        """Fuzzy plan inference over the request; the selected plan becomes the intention."""
        report = self.beliefs.report
        if report is None:
            raise RuntimeError("perceive must be called before deliberate")
        if not self.config.plan_gate:
            # Variant without the plan gate: every request is admitted in full and no action is gated.
            score, fault, plan = None, bool(report.fault_flag > 0.5), self.plan_library["OPTIMIZE"]
        else:
            # The plan is a property of the request, so it is computed once per request.
            key = (float(report.background_load), float(report.requested_latency_ms),
                   float(report.requested_reliability_pct), float(report.isolation_level), float(report.fault_flag))
            cached = self._score_cache.get(key)
            if cached is None:
                cached = self._score_cache[key] = plan_score(report, self.config.latency_scale_ms)
            score, fault = cached
            plan = self.plan_library.select(score, fault)
        self.intention = Intention(plan=plan.name, score=score, fault=fault, interval=report.interval,
                                   trace_index=report.trace_index, admitted_share=plan.admitted_share,
                                   counts_as_rejection=plan.counts_as_rejection)
        return self.intention

    def state_vector(self):
        """BDIx state vector: ten normalized beliefs and the four-element plan indicator."""
        return np.concatenate([self.beliefs.values, self.intention.indicator]).astype(np.float32)

    def command(self, policy_action):
        """Gate the action of the policy by the intention and form the grant command."""
        effective, multiplier = self.plan_library.gate(self.intention.plan, policy_action)
        return GrantCommand(slice_name=self.slice_name, interval=self.intention.interval, plan=self.intention.plan,
                            plan_score=self.intention.score, policy_action=int(policy_action),
                            effective_action=int(effective), step_prbs=int(multiplier * self.config.delta_prb),
                            admitted_share=self.intention.admitted_share)

    def explain(self):
        """Rationale of the current decision in terms of beliefs, desires and the plan inference."""
        report = self.beliefs.report
        out = {"slice": self.slice_name, "interval": report.interval, "trace_index": report.trace_index,
               "beliefs": self.beliefs.as_dict(), "desires": dict(self.desires.__dict__),
               "belief_desire_gap": self.desires.gap(report), "intention": self.intention.plan}
        if self.config.plan_gate:
            out["plan_inference"] = rationale(report, self.config.latency_scale_ms)
        return out


class BDIxSliceAgent(BDIxSliceState):
    """One autonomous agent per slice: BDIx state, PPO policy and private rollout buffer."""

    def __init__(self, config, policy):
        super().__init__(config)
        self.policy = policy        # numpy_agents.PPOAgent with one four-way head
        self.clear_buffer()
        self._pending = None

    def clear_buffer(self):
        self.buffer = dict(obs=[], a=[], logp=[], r=[], v=[], d=[])

    def cycle(self, report, deterministic=True, learning=False):
        """One reasoning cycle: perceive, deliberate, select an action and gate it."""
        self.perceive(report)
        self.deliberate()
        state = self.state_vector()
        action, logp = self.policy.act(state, deterministic=deterministic)
        if learning:
            self._pending = (state, action, logp, self.policy.value(state))
        return self.command(action)

    def remember(self, reward, done):
        """Store the transition of the interval in the private buffer of the agent."""
        state, action, logp, value = self._pending
        b = self.buffer
        b["obs"].append(state); b["a"].append(action); b["logp"].append(logp)
        b["r"].append(float(reward)); b["v"].append(value); b["d"].append(float(done))
        self._pending = None

    def bootstrap_value(self, report):
        """Critic value of the state that follows the last stored transition."""
        self.perceive(report)
        self.deliberate()
        return self.policy.value(self.state_vector())

    def learn(self, last_value):
        """Generalized advantage estimation and the clipped PPO update on the own rollout."""
        b = self.buffer
        adv, ret = self.policy.gae(np.array(b["r"]), np.array(b["v"]), np.array(b["d"]), last_value)
        stats = self.policy.update(np.array(b["obs"]), np.array(b["obs"]), np.array(b["a"]), np.array(b["logp"]), adv, ret)
        mean_reward = float(np.mean(b["r"]))
        self.clear_buffer()
        return mean_reward, stats


class DistributedBDIxController:
    """Three independent slice agents. No state, reward, gradient or message is exchanged between them."""

    central = False

    def __init__(self, agents):
        self.agents = list(agents)
        self.states = self.agents

    def decide(self, reports, deterministic=True, learning=False):
        return [agent.cycle(report, deterministic=deterministic, learning=learning)
                for agent, report in zip(self.agents, reports)]


class CentralizedBDIxController:
    """Centralized variant: the BDIx state of the three slices and one joint policy."""

    central = True

    def __init__(self, configs, policy):
        self.states = [BDIxSliceState(c) for c in configs]
        self.policy = policy        # numpy_agents.PPOAgent with one four-way head per slice
        self.buffer = dict(obs=[], a=[], logp=[], r=[], v=[], d=[])
        self._pending = None

    def global_state(self, reports):
        for state, report in zip(self.states, reports):
            state.perceive(report)
            state.deliberate()
        return np.concatenate([s.state_vector() for s in self.states])

    def decide(self, reports, deterministic=True, learning=False):
        g = self.global_state(reports)
        actions, logp = self.policy.act(g, deterministic=deterministic)
        if learning:
            self._pending = (g, actions, logp, self.policy.value(g))
        return [state.command(int(a)) for state, a in zip(self.states, actions)]

    def remember(self, reward, done):
        g, actions, logp, value = self._pending
        b = self.buffer
        b["obs"].append(g); b["a"].append(actions); b["logp"].append(logp)
        b["r"].append(float(reward)); b["v"].append(value); b["d"].append(float(done))
        self._pending = None

    def learn(self, reports):
        b = self.buffer
        last_value = self.policy.value(self.global_state(reports))
        adv, ret = self.policy.gae(np.array(b["r"]), np.array(b["v"]), np.array(b["d"]), last_value)
        stats = self.policy.update(np.array(b["obs"]), np.array(b["obs"]), np.array(b["a"]), np.array(b["logp"]), adv, ret)
        mean_reward = float(np.mean(b["r"]))
        self.buffer = dict(obs=[], a=[], logp=[], r=[], v=[], d=[])
        return mean_reward, stats
