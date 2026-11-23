"""Typed control-plane messages of the BDIx slicing system.

Three kinds of messages exist and every one of them has the gNB at one end:
the KPI report of a slice (gNB to agent), the grant command of a slice whose
grant changes (agent to gNB, or one batched command of the centralized
controller) and the arbitration event of a slice from which PRBs are reclaimed
(gNB to agent). A message between two agents is refused, which makes the
independence of the learners a checked property of the code.
"""
from __future__ import annotations

from collections import Counter

GNB = "gNB"
KPI_REPORT = "KPI_REPORT"
GRANT_COMMAND = "GRANT_COMMAND"
ARBITRATION_EVENT = "ARBITRATION_EVENT"
KINDS = (KPI_REPORT, GRANT_COMMAND, ARBITRATION_EVENT)


class InterAgentMessageError(RuntimeError):
    """Raised when a message would be exchanged between two agents."""


def agent_endpoint(slice_name):
    return f"agent:{slice_name}"


class MessageLog:
    """Counts the messages of an episode by kind and by pair of endpoints."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.by_kind = Counter()
        self.by_link = Counter()
        self.total = 0
        self.inter_agent = 0

    def send(self, kind, sender, receiver):
        if kind not in KINDS:
            raise ValueError(f"Unknown message kind {kind!r}")
        if (sender == GNB) == (receiver == GNB):
            self.inter_agent += 1
            raise InterAgentMessageError(f"{kind} from {sender} to {receiver}: one endpoint must be the gNB")
        self.by_kind[kind] += 1
        self.by_link[(sender, receiver)] += 1
        self.total += 1

    def summary(self):
        return {"total": int(self.total), "inter_agent": int(self.inter_agent),
                "by_kind": {k: int(self.by_kind[k]) for k in KINDS},
                "by_link": {f"{a} -> {b}": int(n) for (a, b), n in sorted(self.by_link.items())}}
