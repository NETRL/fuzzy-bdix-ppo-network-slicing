from __future__ import annotations

from itertools import product
from pathlib import Path
import csv
import json
import numpy as np

SLICE_NAMES = ("eMBB", "uRLLC", "mMTC")
ACTION_NAMES = ("IDLE", "SCALE_UP", "SCALE_DOWN", "REALLOCATE")
COMPONENT_NAMES = ("latency", "reliability", "load", "isolation")
PLAN_NAMES = ("REJECT", "DEFER", "OPTIMIZE", "IMMEDIATE")
LEVEL_NAMES = ("Low", "Medium", "High")
LEVEL_CENTERS = np.array([0.0, 0.5, 1.0], dtype=float)


WEIGHTS = {
    "eMBB": np.array([0.36, 0.36, 0.18, 0.10], dtype=float),
    "uRLLC": np.array([0.54, 0.27, 0.09, 0.10], dtype=float),
    "mMTC": np.array([0.27, 0.45, 0.18, 0.10], dtype=float),
}
for _weights in WEIGHTS.values():
    _weights.flags.writeable = False

ALLOWED_ACTIONS = {
    "IMMEDIATE": (0, 1, 2, 3),
    "OPTIMIZE": (0, 1, 2, 3),
    "DEFER": (0, 1, 2, 3),
    "REJECT": (0, 2),
}


def _finite(*values):
    arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in values])
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("Fuzzy inputs must contain only finite numerical values")
    return arrays


def _scalar_if_zero_dim(value):
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value


def slice_weights(slice_type="uRLLC", isolation_enabled=True):

    names = np.asarray(slice_type, dtype=object)
    try:
        rows = np.stack([WEIGHTS[str(s)] for s in names.reshape(-1)], axis=0)
    except KeyError as exc:
        raise ValueError(f"Unknown slice type {exc.args[0]!r}; use {SLICE_NAMES}") from None
    rows = rows.reshape(names.shape + (4,)).copy()
    if not isolation_enabled:
        rows[..., 3] = 0.0
        rows /= rows.sum(axis=-1, keepdims=True)
    return rows


def reward_components(latency_ratio, reliability_ratio, load_ratio, isolation):

    lat, rel, load, iso = _finite(latency_ratio, reliability_ratio, load_ratio, isolation)
    latency_score = np.clip((1.3 - lat) / 0.6, 0.0, 1.0)
    reliability_score = np.clip((rel - 0.99) / 0.01, 0.0, 1.0)
    load_score = np.clip(np.minimum(np.minimum((load - 0.05) / 0.25, 1.0), (2.0 - load) / 1.2), 0.0, 1.0)
    isolation_score = np.clip((iso - 0.70) / 0.25, 0.0, 1.0)
    return np.stack([latency_score, reliability_score, load_score, isolation_score], axis=-1)


def crisp_components(latency_ratio, reliability_ratio, load_ratio, isolation):

    lat, rel, load, iso = _finite(latency_ratio, reliability_ratio, load_ratio, isolation)
    eps = 1e-12
    return np.stack([lat <= 1.0 + eps, rel >= 1.0 - eps,
                     load <= 1.0 + eps, iso >= 1.0 - eps], axis=-1).astype(float)


def fuzzy_reward(latency_ratio, reliability_ratio, load_ratio, isolation,
                 slice_type="uRLLC", isolation_enabled=True):

    components = reward_components(latency_ratio, reliability_ratio, load_ratio, isolation)
    weights = slice_weights(slice_type, isolation_enabled)
    return _scalar_if_zero_dim(np.sum(components * weights, axis=-1))


def crisp_reward(latency_ratio, reliability_ratio, load_ratio, isolation,
                 slice_type="uRLLC", isolation_enabled=True):

    components = crisp_components(latency_ratio, reliability_ratio, load_ratio, isolation)
    weights = slice_weights(slice_type, isolation_enabled)
    return _scalar_if_zero_dim(np.sum(components * weights, axis=-1))


def membership_degrees(value):

    (x,) = _finite(value)
    x = np.clip(x, 0.0, 1.0)
    low = np.maximum(1.0 - 2.0 * x, 0.0)
    medium = 1.0 - np.abs(2.0 * x - 1.0)
    high = np.maximum(2.0 * x - 1.0, 0.0)
    return np.stack([low, medium, high], axis=-1)


def _fault_array(fault, shape):
    value = np.asarray(fault)
    if not np.all(np.isin(value, [False, True, 0, 1])):
        raise ValueError("fault must be a binary flag")
    try:
        return np.broadcast_to(value.astype(bool), shape)
    except ValueError:
        raise ValueError("fault shape is not broadcastable to the plan inputs") from None


def _consequent(load, latency_requirement, reliability_requirement, isolation, fault):
    if fault:
        return 0.0

    return float(np.clip(63.0 - 30.0 * load - 10.0 * latency_requirement
                         + 20.0 * reliability_requirement + 20.0 * isolation,
                         0.0, 100.0))


_NO_FAULT_CONSEQUENTS = np.empty((3, 3, 3, 3), dtype=float)
for _idx in product(range(3), repeat=4):
    _NO_FAULT_CONSEQUENTS[_idx] = _consequent(*[LEVEL_CENTERS[i] for i in _idx], False)
_NO_FAULT_CONSEQUENTS.flags.writeable = False


def plan_score(load, latency_requirement, reliability_requirement, isolation, fault=False):

    values = _finite(load, latency_requirement, reliability_requirement, isolation)
    degrees = [membership_degrees(v) for v in values]
    numerator = np.einsum("...i,...j,...k,...l,ijkl->...", *degrees,
                          _NO_FAULT_CONSEQUENTS, optimize=True)
    score = np.where(_fault_array(fault, values[0].shape), 0.0,
                     np.clip(numerator, 0.0, 100.0))
    return _scalar_if_zero_dim(score)


def select_plan(score, fault=False):

    (score,) = _finite(score)
    score = np.clip(score, 0.0, 100.0)
    fault = _fault_array(fault, score.shape)
    result = np.full(score.shape, "REJECT", dtype="<U9")
    result = np.where(score > 30.0, "DEFER", result)
    result = np.where(score > 50.0, "OPTIMIZE", result)
    result = np.where(score > 76.0, "IMMEDIATE", result)
    result = np.where(fault, "REJECT", result)
    return _scalar_if_zero_dim(result)


def allowed_masks(plans):

    values = np.asarray(plans, dtype=object)
    result = np.zeros(values.shape + (4,), dtype=bool)
    flat = result.reshape(-1, 4)
    for i, name in enumerate(values.reshape(-1)):
        if str(name) not in ALLOWED_ACTIONS:
            raise ValueError(f"Unknown plan {name!r}")
        flat[i, list(ALLOWED_ACTIONS[str(name)])] = True
    return result


def gate_actions(actions, plans, mode="deterministic", rng=None, return_details=False):

    actions, plans = np.broadcast_arrays(np.asarray(actions), np.asarray(plans, dtype=object))
    if not np.all(np.isfinite(actions.astype(float))) or not np.all(actions == actions.astype(int)):
        raise ValueError("Actions must be finite integers")
    actions = actions.astype(int)
    if np.any((actions < 0) | (actions > 3)):
        raise ValueError("Action index must be between 0 and 3")
    if mode not in ("deterministic", "random"):
        raise ValueError("mode must be deterministic or random")
    if mode == "random" and not isinstance(rng, np.random.Generator):
        raise ValueError("random override requires a caller-owned numpy Generator")
    masks = allowed_masks(plans)
    executed = actions.copy()
    replacements = np.zeros(actions.shape, dtype=bool)
    for index in np.ndindex(actions.shape):
        if not masks[index + (int(actions[index]),)]:
            choices = np.flatnonzero(masks[index])
            executed[index] = int(choices[0] if mode == "deterministic" else rng.choice(choices))
            replacements[index] = True
    if return_details:
        return {"proposed": _scalar_if_zero_dim(actions),
                "executed": _scalar_if_zero_dim(executed),
                "replaced": _scalar_if_zero_dim(replacements), "allowed_masks": masks}
    return _scalar_if_zero_dim(executed)


def rule_records():

    records = []
    for indices in product(range(3), range(3), range(3), range(3), range(2)):
        load, lat, rel, iso = [float(LEVEL_CENTERS[i]) for i in indices[:4]]
        fault = bool(indices[4])
        score = _consequent(load, lat, rel, iso, fault)
        records.append({"rule_id": len(records) + 1,
                        "load": LEVEL_NAMES[indices[0]],
                        "latency_requirement": LEVEL_NAMES[indices[1]],
                        "reliability_requirement": LEVEL_NAMES[indices[2]],
                        "isolation": LEVEL_NAMES[indices[3]],
                        "fault": int(fault), "score": score,
                        "plan": select_plan(score, fault=fault)})
    return records


def export_specification(directory):

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = rule_records()
    with (directory / "complete_162_rule_base.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    specification = {
        "status": "Fuzzy request-plan and achieved-service reward specification",
        "rule_count": 162,
        "continuous_terms": {"centers": LEVEL_CENTERS.tolist(),
                             "names": list(LEVEL_NAMES),
                             "domain": [0.0, 1.0],
                             "out_of_range": "finite values saturate"},
        "inference": "Product firing weights; zero-order Sugeno weighted average",
        "no_fault_consequent": "clip(63 - 30*load - 10*latency_requirement + 20*reliability_requirement + 20*isolation,0,100)",
        "fault_consequent": 0,
        "score_categories": {"REJECT": "0 <= q <= 30", "DEFER": "30 < q <= 50",
                             "OPTIMIZE": "50 < q <= 76", "IMMEDIATE": "76 < q <= 100"},
        "reference_helper_action_masks": {k: list(v) for k, v in ALLOWED_ACTIONS.items()},
        "packet_replay_actions": "DEFER executes IDLE. REJECT overrides SCALE_UP and REALLOCATE with IDLE. IMMEDIATE doubles the SCALE_UP increment. OPTIMIZE applies the normal command.",
        "action_names": list(ACTION_NAMES),
        "reward_component_order": list(COMPONENT_NAMES),
        "reward_weights": {k: v.tolist() for k, v in WEIGHTS.items()},
        "fuzzy_reward_components": ["clip((1.3 - latency_ratio)/0.6,0,1)",
                                    "clip((reliability_ratio - 0.99)/0.01,0,1)",
                                    "clip(min((load_ratio - 0.05)/0.25,1,(2.0 - load_ratio)/1.2),0,1)",
                                    "clip((isolation - 0.70)/0.25,0,1)"],
        "crisp_reward_components": ["latency_ratio <= 1", "reliability_ratio >= 1",
                                    "load_ratio <= 1", "isolation >= 1"],
        "isolation_ablation": "Remove isolation term then renormalize the three remaining weights; the original relative weights are recovered.",
        "physical_feasibility": "Plan masks encode request semantics. The common environment allocator independently enforces the integer resource budget."
    }
    (directory / "fuzzy_specification.json").write_text(json.dumps(specification, indent=2) + "\n")
    return specification


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Complete, bounded fuzzy reward and request-plan components for validation.\n\nThe rule base supplies all 162 combinations of four ternary inputs and a binary\nfault flag. The module exposes the reward components, request-plan inference\nand an optional reference action-mask helper.\n\nReward inputs describe achieved service: latency/target, achieved on-time\nreliability/target, offered-load/service-capacity ratio and resource-isolation\nsatisfaction. Plan inputs describe request feasibility: load, latency\nrequirement, reliability requirement and isolation, each normalized to [0, 1],\nplus a binary fault flag. Lower normalized latency requirement means a tighter\nrequest. Fuzzification saturates finite values outside the normalized domain.\n')
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    export_specification(args.export)
