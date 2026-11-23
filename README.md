# Fuzzy-BDIx-PPO network slicing replay

This repository accompanies *Distributed Artificial Intelligence-Aided Network Slicing for Beyond Fifth-Generation Networks*. It contains the packet simulator, controllers, request trace, trained checkpoints and evaluation records used to compare radio resource allocation across three network slices. You can reproduce the reported tables directly from the episode records, replay saved policies to regenerate interval logs or train new policies.

## Reproduce Tables IX, XI and XVI

From the repository root, run the following commands with Python 3.11 or later. These commands use only the standard library.

```sh
python tools/check_payload.py --deep
python tools/reproduce_table_ix.py --out regenerated/table_IX
python tools/reproduce_table_xi.py --out regenerated/table_XI
python tools/reproduce_table_xvi.py --out regenerated/table_XVI
```

The first command verifies the file hashes, the trace workbook, the structure of 30 checkpoint files and the row counts and checksums of 48 compressed interval logs. The next two commands create aggregate and per-seed CSVs and compare the output against `reference_tables/`. Table IX contains 48 numerical entries for the learning-controller comparison. Table XI contains 72 entries for the component comparisons. For each controller and slice, the scripts average the 20 windows within each training seed and then average the three seed means. The reported quantities are mean served-packet delay in milliseconds, packet deadline misses in percent, throughput in Mbit/s and a dimensionless isolation index. The last command recalculates the 18 class scores of Table XVI, nine for each classifier, from the confusion matrix counts in `supplementary_d2d/results/confusion_matrices.csv` and compares them with `reference_tables/`.

## How the simulator works

1. The workbook `code/data/slicing_dataset_pre_rl.xlsx` provides 15,000 ordered requests for each of three slice classes. The first 12,000 requests per slice are used for training and the last 3,000 for evaluation. Every trained policy is evaluated on the same 20 overlapping windows of 2,000 intervals. Overlap lets controllers see the same requests at matched offsets. Independently trained seeds provide the between-run comparison.
2. At each 10 ms interval, the fuzzy component maps the request attributes into one of four plans, IMMEDIATE, OPTIMIZE, DEFER or REJECT. The plan determines how the selected resource action is applied and how much demand enters service. Each distributed learner receives ten local observations and a four-element plan indicator. The centralized controller observes all 42 inputs together.
3. The arbiter enforces a budget of 100 physical resource blocks per cell across three cells. The guarded arbiter uses the preceding utilization relative to each slice's guard when reclaiming grants. The rounded Euclidean variant projects over-budget requests and floors the resulting grants to whole blocks. Flooring can leave spare blocks unused.
4. The C++ engine serves the resulting queues in 1 ms ticks. Each block supplies 1.08 Mbit/s from 360 kHz of bandwidth at 3 bit/s/Hz. Arrivals follow the trace. Applied grants and queue history determine the resulting throughput and delay. Separate counters track deadline-eligible packets, missed deadlines and isolation losses during reallocation or arbitration.

## Find the code and records

| Location | Contents |
| --- | --- |
| `code/environment/` | Python replay environment and C++ packet server for the proposed controller and its variants. |
| `code/components/` | Sugeno plan rules, membership functions, reward weights and component checks. |
| `code/campaign/` | NumPy learning algorithms, training and evaluation driver, priority and reward-weight sweeps and fluid capacity calculation. |
| `code/bdix/` | Explicit BDIx agents of the proposed controller: beliefs, desires, intention, plan library, one agent per slice with its PPO policy, the centralized variant, the message log and the control loop. `code/validation/bdix/` holds the episode records obtained through the agents and `tests/` their invariant tests. |
| `code/campaign/results_fixed4/` | Core checkpoints, original per-episode results, seed summaries, training curves, control settings and interval logs. |
| `code/campaign/results_adapted4/` | Inputs for the manuscript figures and reports. The `extras/` directory contains the sensitivity, computational scaling and fluid-capacity results. |
| `comparators/` | Campaign and simulation version for the deep Q-network (DQN), multi-agent PPO (MAPPO) and single-agent PPO-Lagrangian results in Table IX. The workbook in this tree matches the one in `code/data/`. |
| `code/report_tools/fbp/` and `tools/` | Plotting, table reconstruction, checkpoint replay and integrity checks. |
| `reference_figures/` and `reconstructed/` | Manuscript plots used for comparison and one interval export regenerated from a saved policy. |
| `supplementary_d2d/` | Generator, cases, classifiers, predictions and scores of the supplementary D2D classification, with ten repeated datasets, other link conditions and a second scenario. |

Use the driver and environment from the same tree as the saved checkpoint. The Table IX comparators and the rounded Euclidean ablation come from different campaign versions. `artifact_coverage.csv` identifies the available files and seeds for each method. `MANIFEST.sha256` records the file digests.

## Generate the plots and check the fuzzy component

The commands below use a POSIX shell such as Bash on Linux or WSL. Reporting was tested with Python 3.12. The requirements file pins the plotting, numerical and workbook dependencies. The original campaign's dependency list is preserved separately in `code/requirements_locked.txt`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-report.txt
PYTHONPATH=code/report_tools MPLBACKEND=Agg python -m fbp.report --res code/campaign/results_adapted4 --out regenerated/report
python tools/extra_cdfs.py --log code/campaign/results_adapted4/logs/D-PPO_seed0.csv.gz --out regenerated/curves
python code/components/validate_fuzzy_system.py --output regenerated/fuzzy
```

The first report writes 32 PDF figure and panel files and 12 CSV or LaTeX outputs. The second script redraws Figures 17(a) and 18(a), comparing offered and achieved eMBB rates and requested and observed uRLLC delays. It uses the admitted seed-0 intervals with departures. The fuzzy check exports the rule table and specification, reproduces the scores at all 162 rule centers and tests the plan/action and reward behavior. The packet environment maps a DEFER action to IDLE and assesses the next request independently.

The reporting program retains the original output filenames. Use this mapping when comparing its output with the manuscript.

| Manuscript result | Reporting output or input |
| --- | --- |
| Table VIII, centralized and distributed controllers | `table_VII_kpi_c_vs_d.csv` |
| Table IX, learning controllers | `table_VIII_comparison.csv` |
| Table X, fixed and demand-proportional reference controllers | `table_VIII_references.tex` |
| Table XI, ablation | `table_XIII_variants.csv` |
| Table XII, rewards | `table_IX_reward.csv` |
| Table XIII, inference timings | `table_X_inference.csv` |
| Tables XIV and XV, scaling and latency targets | `code/campaign/results_adapted4/extras/scalability.csv` and `threshold_robustness.csv` |

## Replay trained checkpoints

Activate the environment above, install Gymnasium and make a C++17 compiler named `g++` available on the system path. The first replay builds the local packet-service library. Each command writes a compressed interval log and a verification JSON containing the checkpoint digest and the maximum difference from the original episode measurements.

```sh
python -m pip install gymnasium==0.29.1
for seed in 0 1 2; do
  python tools/replay_euclidean.py --code code --out regenerated/euclid --seed "$seed"
done
for method in DQN MAPPO PPO-Lagrangian; do
  for seed in 0 1 2; do
    python tools/replay_comparators.py --code comparators --out regenerated/table_IX_intervals --method "$method" --seed "$seed"
  done
done
```

Each log contains 120,000 slice/interval rows across 20 windows. Each replay compares the regenerated episode measurements with the recorded ones and stops if a difference exceeds 1e-9.

### Choose the matching records

| Experiment | Checkpoints and logs |
| --- | --- |
| Distributed PPO, centralized PPO and the crisp-reward, disabled-plan, omitted-isolation and rounded Euclidean configurations | All have checkpoints and episode records for seeds 0, 1 and 2. Euclidean seed 0 retains its original interval log. Its seed 1 and 2 interval logs were regenerated from the original checkpoints. The auxiliary distributed Lagrangian method also has three seeds. |
| DQN, MAPPO and single-agent PPO-Lagrangian | Use `comparators/campaign/results_fixed4/` for all three seeds and `per_episode_records.csv` for the Table IX evaluation. |
| Fixed allocation, threshold control, demand-proportional allocation and random policy | Use `code/campaign/results_adapted4/logs/` for the gated seed-0 interval logs and the comparator episode records for Table X. The same-named core logs in `code/campaign/results_fixed4/logs/` contain a separate ungated evaluation. The gated and ungated threshold controller produced 27.56% and 22.03% uRLLC misses, respectively. |

DQN seed 0 has two valid stochastic evaluations. With exploration probability 0.05, the first pass gave 9.51% uRLLC deadline misses. Its original interval file was used for plotting. The second pass started from the saved random generator state and gave 10.11% with the same weights and windows. Table IX uses the second pass. Its detailed reconstruction is `reconstructed/table_IX/DQN_seed0.csv.gz`.

## Run the controller through the BDIx agents

`code/bdix/` expresses the proposed controller as one BDIx agent per slice. In every control interval an agent receives the KPI report of its own slice, revises its ten beliefs, takes the service targets of the request as its desires, infers the plan of the request as its intention and gates the action of its PPO policy with that plan. The environment, the fuzzy rules and the learner are the files described above and are unchanged. `code/bdix/README.md` lists where each part of the manuscript is in the code. Activate the replay environment above and run the following commands.

```sh
for method in D-PPO C-PPO D-PPO-crisp D-PPO-nogate D-PPO-noiso D-PPO-euclid; do
  for seed in 0 1 2; do
    python tools/replay_bdix.py --code code --out regenerated/bdix --method "$method" --seed "$seed"
  done
done
python tools/reproduce_tables_bdix.py --records regenerated/bdix --out regenerated/bdix_tables
python tests/test_bdix_invariants.py
```

Each replay runs an archived checkpoint through the agents and compares its 420 per-episode values and its 120,000 interval rows with the archived ones. In each of the 40,000 control intervals it also compares the state vectors, the plans and the grant requests of the agents with those of the environment. It stops if a difference exceeds 1e-9. The table script checks 120 printed cells of Tables VIII, XI and XII and of the proposed controller in Table IX. Without `--records` it reads the records of these replays from `code/validation/bdix/` and uses only the standard library. The agents exchange no message with one another: every counted message is a KPI report, a grant command or an arbitration event with the gNB at one end, and their sum gives the control-message rate of Table VIII.

To train through the agents, run `python tools/train_bdix.py --code code --out regenerated/bdix_training --method D-PPO --seed 0 --compare-driver`. With `--compare-driver` the same seed is also trained with the campaign driver and the parameters are compared bit for bit.

## Train policies

The following commands use the original budget of 200,000 environment interactions per seed. Each writes to a separate directory so the archived run records remain available for comparison. Training produces new checkpoints and evaluation logs. The reproduction scripts at the start of this README use the original recorded runs.

```sh
python code/campaign/train_campaign_numpy.py --out regenerated/core_training --seeds 0 1 2 --steps-d 200000 --steps-c 200000 --methods D-PPO C-PPO D-PPO-crisp D-PPO-nogate D-PPO-noiso
python code/campaign/train_campaign_numpy.py --out regenerated/euclid_training --seeds 0 1 2 --steps-d 200000 --steps-c 200000 --arbiter euclid --methods D-PPO-euclid
python comparators/campaign/train_campaign_numpy.py --out regenerated/comparator_training --seeds 0 1 2 --steps-d 200000 --steps-c 200000 --methods DQN MAPPO PPO-Lagrangian
```

The original priority-weight, reward-weight and fluid-bound scripts write directly to `code/campaign/results_adapted4/extras/`. Run them in a working copy if you want to retain the archived sweep files for comparison.

## Supplementary D2D classification

The supplementary D2D classification of the manuscript (Figure 19 and Table XVI) is a synthetic experiment in `supplementary_d2d/`. The directory contains the generator, the 45,000 generated cases, the training and evaluation code, the four fitted classifiers, the per-case predictions and the score tables. All paths of the reported dataset belong to one network. Without hop count, the accuracy is 98.86% for Random Forest and 98.30% for XGBoost. The directory also holds the scores of ten further datasets generated with other seeds, of eleven single networks with other link conditions (89.02% to 99.80%) and of a second scenario in which the link conditions are drawn independently for every path (87.31% and 78.49%). `reference_figures/image28_hd.pdf` and `reference_figures/image29_hd.pdf` are the two panels of Figure 19. The commands are given in `supplementary_d2d/README.md` and the link model is defined in `supplementary_d2d/MODEL.md`. That separate experiment supplies no input to the network slicing simulator.
