"""Explicit BDIx agent layer of the Fuzzy-BDIx-PPO slicing controller.

The package states the belief-desire-intention extended (BDIx) structure of the
controller as code: one agent per slice with its beliefs, its desires, its
intention, the plan library and its PPO policy. It adds no mechanism to the
method. The packet service, the grant update, the PRB arbiter, the fuzzy rules
and the PPO learner are the archived ones, and a replay through the agents
gives the archived episode records (tools/replay_bdix.py).
"""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent
for _name in ("campaign", "components", "environment"):
    _path = str(_CODE / _name)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from .beliefs import BELIEF_NAMES, BeliefBase, KPIReport, ObservabilityError  # noqa: E402
from .desires import DesireSet  # noqa: E402
from .plans import ACTION_NAMES, PLAN_ORDER, Intention, Plan, PlanLibrary, plan_inputs, plan_score, rationale  # noqa: E402
from .agent import (STATE_SIZE, BDIxSliceAgent, BDIxSliceState, CentralizedBDIxController,  # noqa: E402
                    DistributedBDIxController, GrantCommand, SliceConfig)
from .messages import (ARBITRATION_EVENT, GNB, GRANT_COMMAND, KPI_REPORT, InterAgentMessageError,  # noqa: E402
                       MessageLog)
from .system import (BDIxSlicingSystem, EquivalenceError, controller_from_policies, kpi_report_from_arrays,  # noqa: E402
                     kpi_reports, slice_configs)
from .checkpoints import load_checkpoint  # noqa: E402
from .training import train_centralized, train_distributed  # noqa: E402

__all__ = [
    "ACTION_NAMES", "ARBITRATION_EVENT", "BELIEF_NAMES", "BDIxSliceAgent", "BDIxSliceState", "BDIxSlicingSystem",
    "BeliefBase", "CentralizedBDIxController", "DesireSet", "DistributedBDIxController", "EquivalenceError", "GNB",
    "GRANT_COMMAND", "GrantCommand", "Intention", "InterAgentMessageError", "KPI_REPORT", "KPIReport", "MessageLog",
    "ObservabilityError", "PLAN_ORDER", "Plan", "PlanLibrary", "STATE_SIZE", "SliceConfig", "controller_from_policies",
    "kpi_report_from_arrays", "kpi_reports", "load_checkpoint", "plan_inputs", "plan_score", "rationale",
    "slice_configs", "train_centralized", "train_distributed",
]
