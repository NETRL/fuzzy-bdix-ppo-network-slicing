"""Beliefs of a BDIx slice agent.

A slice agent never reads the environment. In every control interval it
receives one KPI report from the gNB, restricted to its own slice, and revises
its ten normalized beliefs from that report (Algorithm 1, Step 2a, BDIx State
Construction). The beliefs are normalized by the desires of the agent, which
are the SLA targets of the current request.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BELIEF_NAMES = (
    "backlog",                # queued packets in thousands, log-compressed
    "oldest_age",             # age of the oldest queued packet over the requested latency, log-compressed
    "served_delay",           # last served mean delay over the requested latency, log-compressed
    "timely_share",           # timely-delivered share over the requested reliability
    "requested_rate",         # requested rate over the total network capacity, log-compressed
    "requested_latency",      # requested latency over its training 95th percentile
    "requested_reliability",  # requested reliability as a fraction
    "granted_fraction",       # mean granted fraction of the cell budget
    "isolation",              # isolation index of the preceding interval
    "fault",                  # fault flag of the request
)


class ObservabilityError(RuntimeError):
    """Raised when an agent is given telemetry that does not belong to its slice."""


@dataclass(frozen=True)
class KPIReport:
    """Local telemetry of one slice for one control interval.

    The report holds the attributes of the request that arrives in the interval
    and the measured state of the slice (its own queues, grants and isolation
    index). It has no field that describes another slice.
    """

    slice_name: str
    interval: int
    trace_index: int
    # request of the interval (exogenous inputs replayed from the trace)
    requested_rate_mbps: float
    requested_latency_ms: float
    requested_reliability_pct: float
    fault_flag: float
    isolation_level: float
    background_load: float
    # measured state of the slice
    backlog_packets: np.ndarray       # one value per cell
    oldest_age_ms: np.ndarray         # one value per cell
    served_mean_delay_ms: float
    timely_fraction: float
    granted_prbs: np.ndarray          # one value per cell
    isolation_index: float
    # static configuration of the cell
    cell_budget_prbs: int


class BeliefBase:
    """Ten normalized beliefs of one slice, revised from the KPI report of the interval."""

    def __init__(self, slice_name, latency_scale_ms, prb_rate_mbps, n_cells=3):
        self.slice_name = str(slice_name)
        self.latency_scale_ms = latency_scale_ms
        self.prb_rate_mbps = prb_rate_mbps
        self.n_cells = int(n_cells)
        self.values = np.zeros(len(BELIEF_NAMES), dtype=np.float64)
        self.report = None
        self.revisions = 0

    def revise(self, report, desires):
        """Belief revision from the report of the own slice, normalized by the desires."""
        if not isinstance(report, KPIReport):
            raise ObservabilityError("A slice agent perceives KPI reports only.")
        if report.slice_name != self.slice_name:
            raise ObservabilityError(
                f"The {self.slice_name} agent was given the telemetry of {report.slice_name}.")
        latency_bound = desires.latency_bound_ms
        network_capacity = report.cell_budget_prbs * self.n_cells * self.prb_rate_mbps
        v = self.values
        v[0] = np.log1p(np.minimum(report.backlog_packets.sum() / 1000, 100))
        v[1] = np.log1p(np.minimum(report.oldest_age_ms.max() / latency_bound, 100))
        v[2] = np.log1p(np.minimum(report.served_mean_delay_ms / latency_bound, 100))
        v[3] = np.minimum(report.timely_fraction / np.maximum(desires.reliability_floor, 1e-9), 2)
        v[4] = np.log1p(np.minimum(report.requested_rate_mbps / network_capacity, 20))
        v[5] = np.minimum(latency_bound / self.latency_scale_ms, 20)
        v[6] = desires.reliability_floor
        v[7] = report.granted_prbs.mean() / report.cell_budget_prbs
        v[8] = report.isolation_index
        v[9] = report.fault_flag
        self.report = report
        self.revisions += 1
        return v

    def as_dict(self):
        return {name: float(value) for name, value in zip(BELIEF_NAMES, self.values)}
