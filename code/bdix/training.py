"""Training of the BDIx agents (Algorithm 1, Distributed Fuzzy-BDIx-PPO).

Every agent collects its own rollout of 2,048 intervals, estimates its
advantages by generalized advantage estimation and applies the clipped PPO
update to its own actor and critic. The reward of an agent is the reward of its
own slice computed by the environment. Settings and random streams are those of
campaign/train_campaign_numpy.py, so the same seed gives the same policies.
"""
from __future__ import annotations

import time

import numpy as np

import trace_resource_env as tre
from numpy_agents import PPOAgent

from .agent import STATE_SIZE, BDIxSliceAgent, CentralizedBDIxController, DistributedBDIxController
from .system import SLICES, BDIxSlicingSystem, kpi_reports, slice_configs

HORIZON = 2000
ROLLOUT = 2048


def train_distributed(seed, steps, reward_kind="fuzzy", isolation_enabled=True, plan_gate=True, hp=None,
                      cross_check=False):
    """Train one BDIx agent per slice. Returns the agents, the reward curve and the training time."""
    hp = {**dict(lr=3e-4, epochs=20, minibatch=64, ent_coef=0.01), **(hp or {})}
    env = tre.TraceResourceEnv(reward_kind=reward_kind, isolation_enabled=isolation_enabled, horizon=HORIZON)
    if not plan_gate:
        env.plans = lambda: ["OPTIMIZE"] * 3
    agents = [BDIxSliceAgent(config, PPOAgent(STATE_SIZE, 4, 1, seed=seed * 100 + i, lr=hp["lr"], epochs=hp["epochs"],
                                              minibatch=hp["minibatch"], ent_coef=hp["ent_coef"]))
              for i, config in enumerate(slice_configs(env, plan_gate))]
    system = BDIxSlicingSystem(env, DistributedBDIxController(agents), cross_check=cross_check)
    env.reset(seed=seed, options={"split": "train"})
    curve = []
    started = time.time()
    n = 0
    while n < steps:
        result = system.step(deterministic=False, learning=True)
        for i, agent in enumerate(agents):
            agent.remember(result.info["slice_rewards"][i], result.done)
        if result.done:
            env.reset(options={"split": "train"})
        n += 1
        if n % ROLLOUT == 0:
            fraction = max(0.0, 1.0 - n / steps)
            for agent in agents:
                agent.policy.opt_a.lr = hp["lr"] * fraction
                agent.policy.opt_c.lr = (hp.get("lr_v") or hp["lr"]) * fraction
            reports = kpi_reports(env)
            record = dict(step=n)
            for i, agent in enumerate(agents):
                mean_reward, _ = agent.learn(agent.bootstrap_value(reports[i]))
                record[f"reward_{SLICES[i]}"] = mean_reward
            record["reward_weighted"] = float(0.3 * record["reward_eMBB"] + 0.5 * record["reward_uRLLC"]
                                              + 0.2 * record["reward_mMTC"])
            curve.append(record)
    return agents, curve, time.time() - started


def train_centralized(seed, steps, plan_gate=True, cross_check=False):
    """Train the centralized variant. Returns the controller, the reward curve and the training time."""
    env = tre.TraceResourceEnv(horizon=HORIZON)
    if not plan_gate:
        env.plans = lambda: ["OPTIMIZE"] * 3
    env.reset(seed=seed, options={"split": "train"})
    policy = PPOAgent(len(SLICES) * STATE_SIZE, 4, len(SLICES), seed=seed, epochs=20, minibatch=64)
    controller = CentralizedBDIxController(slice_configs(env, plan_gate), policy)
    system = BDIxSlicingSystem(env, controller, cross_check=cross_check)
    curve = []
    started = time.time()
    n = 0
    while n < steps:
        result = system.step(deterministic=False, learning=True)
        controller.remember(float(np.dot(tre.OMEGA, result.info["slice_rewards"])), result.done)
        if result.done:
            env.reset(options={"split": "train"})
        n += 1
        if n % ROLLOUT == 0:
            policy.opt_a.lr = 3e-4 * max(0.0, 1.0 - n / steps)
            policy.opt_c.lr = policy.opt_a.lr
            mean_reward, _ = controller.learn(kpi_reports(env))
            curve.append(dict(step=n, reward_weighted=mean_reward))
    return controller, curve, time.time() - started
