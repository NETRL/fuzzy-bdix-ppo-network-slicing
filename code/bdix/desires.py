"""Desires of a BDIx slice agent.

The desires are the SLA targets of the request of the interval. They enter the
state of the agent through the normalization of its beliefs, and the distance
between the achieved KPIs and the targets is the belief-desire gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesireSet:
    """SLA targets of the current request of one slice."""

    latency_bound_ms: float      # requested latency of the request
    reliability_floor: float     # requested reliability as a fraction
    requested_rate_mbps: float   # requested downlink rate
    isolation_level: float       # requested isolation level (0, 0.5 or 1)

    @classmethod
    def from_report(cls, report):
        return cls(latency_bound_ms=report.requested_latency_ms,
                   reliability_floor=report.requested_reliability_pct / 100,
                   requested_rate_mbps=report.requested_rate_mbps,
                   isolation_level=report.isolation_level)

    def gap(self, report):
        """Belief-desire gap: achieved KPIs relative to the targets of the request.

        The latency ratio uses the larger of the served mean delay and the age
        of the oldest queued packet. A latency ratio above one or a reliability
        ratio below one means that the corresponding target is missed.
        """
        latency = np.maximum(report.served_mean_delay_ms, report.oldest_age_ms.max())
        return {
            "latency_ratio": float(latency / np.maximum(self.latency_bound_ms, 1e-9)),
            "reliability_ratio": float(report.timely_fraction / np.maximum(self.reliability_floor, 1e-9)),
        }
