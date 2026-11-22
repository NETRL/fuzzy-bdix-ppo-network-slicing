# Supplementary D2D classification

This directory contains the supplementary device-to-device (D2D) classification of the manuscript (Subsections IV-E and V-E, Figure 19 and Table XVI). The experiment is synthetic. The generator supplies the inputs for fitting and evaluating Random Forest and XGBoost. Each classifier is trained twice, once with hop count and once using five path measurements alone. The network slicing simulator runs independently of these classifiers.

## Run the experiment

Use Python 3.12. From the repository root, run these commands in a Python environment of your choice.

```sh
python -m pip install -r supplementary_d2d/requirements.txt
python supplementary_d2d/reproduce.py --out regenerated/d2d
python supplementary_d2d/repeat.py --out regenerated/d2d_repeats
python supplementary_d2d/operating_points.py --out regenerated/d2d_networks
python supplementary_d2d/reproduce.py --scenario mixed_networks --out regenerated/d2d_mixed
```

Choose an empty or new output directory for each command. The first command regenerates the dataset, trains the four models, checks that each saved model reproduces its predictions, builds the score tables and draws the two panels of Figure 19. It also verifies the scores independently and compares the regenerated data, predictions and numerical tables byte for byte with the included CSV files. The second command repeats the experiment on ten datasets generated with the root seeds 628105 to 628114 and compares the scores with `results/repeats.csv`. The third command runs the experiment for eleven single networks with other link conditions and compares the scores with `results/operating_points.csv`. The fourth command runs the experiment in the second scenario of the generator and compares its three score tables with `results_mixed_networks/`.

To verify the included data and scores without installing the training dependencies, run the following commands. They use only Python's standard library.

```sh
python supplementary_d2d/verify.py --data supplementary_d2d/data/cases.csv --results supplementary_d2d/results
python tools/reproduce_table_xvi.py --out regenerated/table_XVI
```

## Data, training and evaluation

The generator creates 36,000 training cases and 9,000 separate test cases from independent random streams. The test set has 20 clients, 280 single-relay paths and 8,700 multihop paths. In the reported scenario, `single_network`, all paths belong to one network with a bandwidth of 10 MHz, a mean signal-to-noise ratio of 14.5 dB and a mean utilization of 0.485. Each hop deviates from these means with standard deviations of 3 dB and 0.09. The five features available in both runs are latency, throughput, loss fraction, load fraction and a synthetic airtime-isolation indicator. The hop-count runs additionally expose the field that defines the class. The fitting code lists its allowed inputs explicitly. The case ID, partition name and true role are excluded from the predictors. [Generator and field definitions](MODEL.md)

The two model families use fixed hyperparameters and seed 317. Each training case receives the inverse-frequency weight $n/(3n_c)$, where $n$ is the number of training cases and $n_c$ is its class count. Each class therefore contributes equal total training weight. All four models use the same partitions. The entire test set is held out until scoring and each test case receives equal weight in the evaluation. Class counts in the experiment are fixed design choices. They are not estimates of role prevalence in an operational network.

| Classifier | Hop count | Correct / 9,000 | Accuracy | Balanced accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Forest | Included | 9,000 | 100.00% | 100.00% | 100.00% |
| Random Forest | Excluded | 8,897 | 98.86% | 94.03% | 93.18% |
| XGBoost | Included | 9,000 | 100.00% | 100.00% | 100.00% |
| XGBoost | Excluded | 8,847 | 98.30% | 95.45% | 91.27% |

With hop count visible, the label follows directly from the generator's rule. With hop count removed, both classifiers recognize 19 of the 20 clients and assign no other path to the client role. The remaining errors lie between the relay and multihop roles, whose measurements overlap because of the per-hop deviations and the random terms of the measurements. Random Forest assigns 68 multihop paths to relay and 34 relay paths to multihop. XGBoost assigns 132 and 20, respectively, which gives a relay precision of 66.16%. Always predicting multihop would give 96.67% accuracy and 33.33% balanced accuracy on this deliberately imbalanced test set. Balanced accuracy is mean recall over the three roles. Macro F1 is the unweighted mean of the three class F1 scores. Only 20 test cases belong to the client class, so one case changes its recall by five percentage points.

## Repeated datasets

`results/repeats.csv` holds the scores of ten further datasets, generated with the root seeds 628105 to 628114 and evaluated with the same code and model seed.

| Classifier | Hop count | Accuracy, mean | Accuracy, range | Balanced accuracy, mean | Balanced accuracy, range |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Forest | Included | 99.99% | 99.97% to 100.00% | 99.17% | 95.00% to 100.00% |
| Random Forest | Excluded | 98.73% | 98.46% to 99.01% | 92.66% | 87.79% to 96.20% |
| XGBoost | Included | 100.00% | 100.00% to 100.00% | 100.00% | 100.00% to 100.00% |
| XGBoost | Excluded | 98.42% | 98.20% to 98.62% | 94.33% | 89.18% to 97.89% |

With hop count, Random Forest misses three clients in one of the ten datasets and two in another. The balanced accuracy varies more than the accuracy. One of the 20 client cases changes it by 1.67 percentage points.

## Other link conditions

`results/operating_points.csv` holds the scores of eleven single networks with other link conditions, generated with the root seed 628104. Nine networks combine mean signal-to-noise ratios of 5, 14.5 and 24 dB with mean utilizations of 0.12, 0.485 and 0.85 at 10 MHz. Two networks use 5 and 20 MHz at the reported means. The table lists the runs without hop count. With hop count, every run classifies all 9,000 test cases correctly.

| Bandwidth | Mean signal-to-noise ratio | Mean utilization | Random Forest accuracy | Random Forest balanced accuracy | XGBoost accuracy | XGBoost balanced accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 MHz | 5 dB | 0.12 | 92.56% | 76.07% | 89.02% | 78.93% |
| 10 MHz | 5 dB | 0.485 | 92.19% | 79.96% | 89.27% | 82.34% |
| 10 MHz | 5 dB | 0.85 | 92.36% | 84.38% | 89.80% | 84.76% |
| 10 MHz | 14.5 dB | 0.12 | 99.24% | 96.63% | 99.43% | 98.08% |
| 10 MHz | 14.5 dB | 0.485 | 98.86% | 94.03% | 98.30% | 95.45% |
| 10 MHz | 14.5 dB | 0.85 | 95.13% | 85.33% | 93.92% | 91.86% |
| 10 MHz | 24 dB | 0.12 | 99.80% | 98.55% | 99.80% | 98.66% |
| 10 MHz | 24 dB | 0.485 | 99.67% | 96.26% | 99.80% | 99.01% |
| 10 MHz | 24 dB | 0.85 | 97.16% | 94.13% | 96.70% | 95.24% |
| 5 MHz | 14.5 dB | 0.485 | 98.76% | 93.88% | 98.28% | 95.55% |
| 20 MHz | 14.5 dB | 0.485 | 98.79% | 93.77% | 98.30% | 95.56% |

The separability depends on the link conditions. The accuracy without hop count ranges from 89.02% at a mean signal-to-noise ratio of 5 dB to 99.80% at 24 dB. In the three networks at 5 dB and in the network at 14.5 dB with a mean utilization of 0.85, both accuracies are below the 96.67% of always predicting multihop. The bandwidth changes the accuracy by at most 0.10 percentage points.

## Paths from different networks

The scenario `mixed_networks` draws the bandwidth (5, 10 or 20 MHz), the mean signal-to-noise ratio (5 to 24 dB) and the mean utilization (0.12 to 0.85) independently for every path. Everything else is unchanged. Its score tables are in `results_mixed_networks/`.

| Classifier | Hop count | Correct / 9,000 | Accuracy | Balanced accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Forest | Included | 9,000 | 100.00% | 100.00% | 100.00% |
| Random Forest | Excluded | 7,858 | 87.31% | 49.05% | 40.41% |
| XGBoost | Included | 9,000 | 100.00% | 100.00% | 100.00% |
| XGBoost | Excluded | 7,064 | 78.49% | 58.87% | 39.41% |

Without hop count, both accuracies are below the 96.67% of always predicting multihop. The five measurements therefore separate the roles best when all paths share one network with a medium or high signal-to-noise ratio and a low or medium utilization. They separate the roles least when the link conditions differ from path to path as widely as in this scenario.

## Files

| File | Contents |
| --- | --- |
| `data/cases.csv` | All 45,000 synthetic case IDs, partitions, true roles, hop counts and five path measurements. |
| `generate_data.py` | Link model, two scenarios, an option for other single networks, parameter distributions, random streams and case generation. |
| `train_evaluate.py` | Feature selection, training, model export, inference and scoring. |
| `results/predictions.csv` | One row per held-out case with its ID, true role and four predictions. Join to the input CSV using `case_id`. |
| `results/models/` | Two fitted Random Forest models in joblib format and two native XGBoost models in JSON. Numeric class order is 0 for client, 1 for relay and 2 for multihop relay. |
| `results/summary.csv`, `results/class_scores.csv` | Overall and per-role scores, with proportions on the 0 to 1 scale. |
| `results/confusion_matrices.csv` | All four matrices as integer counts. |
| `results/repeats.csv` | Scores of the ten repeated datasets. |
| `results/operating_points.csv` | Scores of the eleven single networks with other link conditions. |
| `results_mixed_networks/` | The three score tables of the `mixed_networks` scenario. |
| `plot_matrices.py` | Draws the two panels of Figure 19 from the count table. The panels are stored as `reference_figures/image28_hd.pdf` and `reference_figures/image29_hd.pdf` in the repository root. |
| `verify.py` | Independently checks case IDs, partitions, class totals, feature ranges, predictions and every reported count and score. |
| `reproduce.py` | Runs the generator, the training, the verification and the plotting and compares the outputs with the included files. |
| `repeat.py` | Repeats the generation, the training and the verification for ten other root seeds and compares the scores with the included file. |
| `operating_points.py` | Repeats the generation, the training and the verification for eleven single networks with other link conditions and compares the scores with the included file. |
