# Table IX comparator records

This folder contains the DQN, MAPPO and single-agent PPO-Lagrangian checkpoints, training curves and original detailed interval records for seeds 0, 1 and 2. Each seed has 20 held-out evaluation windows. The summary and episode CSV files also retain rows from other methods in this campaign. The root coverage table lists the interval and checkpoint files supplied for each method.

DQN uses an exploration probability of 0.05 during evaluation. For seed 0, the original detailed log comes from the first evaluation. The tabulated result uses a second evaluation initialized from the saved generator state. Its original per-episode rows are in `per_episode_records.csv` and its regenerated interval rows are in the root `reconstructed/table_IX/` directory. The root README gives the replay commands and coverage of the core and comparator campaigns.
