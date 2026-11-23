# Explicit BDIx agents

This package states the belief-desire-intention extended (BDIx) structure of the Fuzzy-BDIx-PPO controller as code. Each slice has one agent that holds its beliefs, its desires and its intention, runs the fuzzy plan inference, selects a resource action with its own PPO policy and learns from its own rollout. The package adds no mechanism to the method of the manuscript. The fuzzy rules, the PPO learner, the grant update, the PRB arbiter and the packet service are the archived ones.

## Where each part of the manuscript is in the code

| Manuscript | Code |
| --- | --- |
| Request of the interval and measured KPIs of one slice | `beliefs.KPIReport`, `system.kpi_reports` |
| Ten normalized beliefs (Algorithm 1, Step 2a) | `beliefs.BeliefBase.revise` |
| Desires, the SLA targets of the request, and the belief-desire gap | `desires.DesireSet` |
| Fuzzy-BDIx plan selection with its six pipeline stages | `plans.plan_score`, `plans.rationale`, `components/fuzzy_system.py` |
| Plan categories IMMEDIATE, OPTIMIZE, DEFER and REJECT and the plan gate | `plans.PlanLibrary`, `plans.Plan.gate` |
| Intention, the plan of the current request as a four-element indicator | `plans.Intention` |
| BDIx state vector of 14 inputs | `agent.BDIxSliceState.state_vector` |
| One agent per slice with its own policy, rollout, advantage estimation and clipped update | `agent.BDIxSliceAgent`, `training.train_distributed`, `campaign/numpy_agents.py` |
| Centralized variant with 42 inputs and a joint action | `agent.CentralizedBDIxController`, `training.train_centralized` |
| Grant update, donor rule, PRB arbiter and packet service | `environment/trace_resource_env.py`, called by `system.BDIxSlicingSystem.step` |
| Control messages: KPI reports, grant commands and arbitration events | `messages.MessageLog` |

## How the agents relate to the archived code path

In `campaign/train_campaign_numpy.py` the same quantities are produced inside the environment by `local_observations`, `plans` and `propose`. In this package they are produced by the agents: an agent receives the KPI report of its own slice, forms its state, infers the plan and gates the action of its policy. The gNB then applies the grant update and the arbiter and the packet service runs with the plans of the agents. With `cross_check=True`, the state vectors, the plans and the grant requests of the agents are compared with those of the environment in every control interval.

The agents exchange nothing with one another. A KPI report holds the fields of one slice only and an agent refuses the report of another slice. Every message has the gNB at one end and `messages.MessageLog` refuses any other message. Cooperative communication between the agents is outside the method of the manuscript.

## Check the agents

```sh
python tools/replay_bdix.py --code code --out regenerated/bdix --method D-PPO --seed 0 --trace-intervals 20
python tools/reproduce_tables_bdix.py --out regenerated/bdix_tables
python tests/test_bdix_invariants.py
python tools/train_bdix.py --code code --out regenerated/bdix_training --method D-PPO --seed 0 --compare-driver
```

The first command replays an archived checkpoint through the agents. It stops if a per-episode value differs from `campaign/results_fixed4/per_episode_records.csv` by more than 1e-9 or an interval row differs from the archived interval log. It also writes the message counts, the share of each plan and, with `--trace-intervals`, the rationale of each decision in terms of beliefs, desires, rule firing strengths and plan score. The methods are `D-PPO`, `C-PPO`, `D-PPO-crisp`, `D-PPO-nogate`, `D-PPO-noiso`, `D-PPO-euclid` and `D-PPO-Lagrangian`. The second command rebuilds Tables VIII, XI and XII and the rows of the proposed controller in Table IX from the records of these replays, which are archived in `validation/bdix/`. The third command runs the invariant tests: local observability, bitwise agreement with the environment, capacity conservation, rule coverage, the plan gate, the message count, the independence of the agents and training. The fourth command trains through the agents and through the campaign driver with the same seed and compares the parameters bit for bit.

## Archived verification

`validation/bdix/` holds the outputs of these checks for the archived checkpoints.

| File | Contents |
| --- | --- |
| `per_episode_records.csv` | 1,260 episode records obtained through the agents: 21 checkpoints, 20 windows and three slices. |
| `replay_summary.json` | For each checkpoint, the largest difference from the archived episode records and interval logs, the number of cross-checked intervals, the message counts by kind and the share of each plan. |
| `decision_trace_D-PPO_seed0.json` | Rationale of the decisions of the three agents in the first 20 intervals of window 0. |
| `training_summary.json` | Retraining of `D-PPO` and `C-PPO` through the agents with seeds 0, 1 and 2 and 200,000 interactions. The trained parameters are identical bit for bit to those of the archived checkpoints. |
| `invariant_tests.txt` | Output of the invariant tests. |
