"""Plan layer of a BDIx slice agent: fuzzy plan inference, plan library and intention.

The plan of a request is produced by the zero-order Sugeno inference of
components/fuzzy_system.py over the request attributes (Fuzzy-BDIx Plan
Selection). The selected plan is the intention of the agent for the interval.
The plan library states, for each of the four plans, the share of the committed
demand that is admitted and how the action of the policy is gated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import fuzzy_system as fz

ACTION_NAMES = ("IDLE", "SCALE_UP", "SCALE_DOWN", "REALLOCATE")
IDLE, SCALE_UP, SCALE_DOWN, REALLOCATE = range(4)
# Order of the four-element plan indicator in the state vector.
PLAN_ORDER = ("REJECT", "DEFER", "OPTIMIZE", "IMMEDIATE")
PLAN_INPUT_NAMES = ("load", "latency_requirement", "reliability_requirement", "isolation")


@dataclass(frozen=True)
class Plan:
    """One entry of the plan library."""

    name: str
    score_above: float           # the plan applies for a plan score above this value
    score_up_to: float           # and up to this value
    admitted_share: float        # share of the committed demand offered to the queues
    counts_as_rejection: bool
    rule: str

    def gate(self, action):
        """Return the effective action and the step multiplier for a policy action."""
        action = int(action)
        if self.name == "REJECT" and action in (SCALE_UP, REALLOCATE):
            return IDLE, 1
        if self.name == "DEFER":
            return IDLE, 1
        if self.name == "IMMEDIATE" and action == SCALE_UP:
            return SCALE_UP, 2
        return action, 1


class PlanLibrary:
    """The four plans of the Fuzzy-BDIx layer and their gate semantics."""

    def __init__(self, residual_share=0.3):
        self.residual_share = float(residual_share)
        self.plans = {
            "IMMEDIATE": Plan("IMMEDIATE", 76.0, 100.0, 1.0, False,
                              "A SCALE_UP action is executed with twice the step size."),
            "OPTIMIZE": Plan("OPTIMIZE", 50.0, 76.0, 1.0, False,
                             "The action selected by the policy is executed unchanged."),
            "DEFER": Plan("DEFER", 30.0, 50.0, self.residual_share, False,
                          "The action is replaced by IDLE and only the residual share of the committed demand is offered."),
            "REJECT": Plan("REJECT", -np.inf, 30.0, self.residual_share, True,
                           "SCALE_UP and REALLOCATE are replaced by IDLE and only the residual share of the committed demand is offered."),
        }

    def __getitem__(self, name):
        return self.plans[str(name)]

    def select(self, score, fault=False):
        """Plan category of a plan score (the fault flag forces REJECT)."""
        return self.plans[str(fz.select_plan(score, fault))]

    def gate(self, plan_name, action):
        return self.plans[str(plan_name)].gate(action)


@dataclass(frozen=True)
class Intention:
    """Committed plan of the agent for the request of one control interval."""

    plan: str
    score: float
    fault: bool
    interval: int
    trace_index: int
    admitted_share: float
    counts_as_rejection: bool

    @property
    def indicator(self):
        """Four-element plan indicator, the intention component of the BDIx state."""
        onehot = np.zeros(len(PLAN_ORDER))
        onehot[PLAN_ORDER.index(self.plan)] = 1.0
        return onehot


def plan_inputs(report, latency_scale_ms):
    """Normalized inputs of the plan inference for the request of the report.

    Returns the background load, the latency requirement relative to the
    training 95th percentile, the reliability requirement, the requested
    isolation level and the fault flag.
    """
    load = np.clip(report.background_load, 0, 1)
    latency = np.minimum(report.requested_latency_ms / latency_scale_ms, 1)
    reliability = np.clip(report.requested_reliability_pct / 100, 0, 1)
    isolation = report.isolation_level
    fault = bool(report.fault_flag > 0.5)
    return load, latency, reliability, isolation, fault


def plan_score(report, latency_scale_ms):
    """Plan score of the request between 0 and 100 and its fault flag."""
    load, latency, reliability, isolation, fault = plan_inputs(report, latency_scale_ms)
    return float(fz.plan_score(load, latency, reliability, isolation, fault)), fault


def rationale(report, latency_scale_ms, top=3):
    """Auditable rationale of the plan decision for one request.

    The six stages of the Fuzzy-BDIx pipeline are evaluated explicitly: the
    request is ingested, its attributes are fuzzified, the firing strengths of
    the rules are computed by product inference, the plan score is the weighted
    average of the rule consequents, the plan is selected by the thresholds and
    the selected plan is dispatched through the gate.
    """
    load, latency, reliability, isolation, fault = plan_inputs(report, latency_scale_ms)
    crisp = (load, latency, reliability, isolation)
    degrees = [fz.membership_degrees(x) for x in crisp]
    firing = np.einsum("i,j,k,l->ijkl", *degrees)
    consequents = np.zeros((3, 3, 3, 3)) if fault else fz._NO_FAULT_CONSEQUENTS
    score = float(np.clip(np.sum(firing * consequents), 0.0, 100.0))
    order = np.argsort(firing, axis=None)[::-1][:top]
    rules = []
    for flat in order:
        idx = np.unravel_index(flat, firing.shape)
        if firing[idx] <= 0:
            break
        rules.append({
            **{name: fz.LEVEL_NAMES[i] for name, i in zip(PLAN_INPUT_NAMES, idx)},
            "fault": bool(fault),
            "firing_strength": float(firing[idx]),
            "consequent": float(consequents[idx]),
        })
    return {
        "request": {name: float(x) for name, x in zip(PLAN_INPUT_NAMES, crisp)} | {"fault": bool(fault)},
        "memberships": {name: {lvl: float(d) for lvl, d in zip(fz.LEVEL_NAMES, deg)}
                        for name, deg in zip(PLAN_INPUT_NAMES, degrees)},
        "strongest_rules": rules,
        "score": score,
        "plan": str(fz.select_plan(score, fault)),
    }
