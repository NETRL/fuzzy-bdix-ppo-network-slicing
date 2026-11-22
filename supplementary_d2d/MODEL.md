# Synthetic D2D path model

Each row describes one independently generated device-to-device (D2D) path. The experiment asks how well five path measurements distinguish three transmission modes when the classifier cannot see the hop count. The generator implements a simplified scenario. All link conditions, measurements and case identifiers are generated for this experiment. The class frequencies and the model parameters below are design assumptions.

## Paths and sample sizes

The label identifies the arrangement of the whole path. A direct path has one radio hop and the label `client`. A path through one forwarding relay has two hops and the label `relay`. A path through multiple forwarding relays has three, four or five hops and the label `multihop_relay`. The last three hop counts are sampled with equal probability. These are operational definitions for this generator.

| Partition | Client | Relay | Multihop relay | Total |
| --- | ---: | ---: | ---: | ---: |
| Training | 80 | 1,120 | 34,800 | 36,000 |
| Test | 20 | 280 | 8,700 | 9,000 |

NumPy's `SeedSequence` spawns one PCG64 random stream for training and one for testing from the default root seed 628104. Each stream shuffles its hop counts before generating independent path conditions. Case identifiers follow that shuffled order. The classifier feature set consists of the five measurements below and, in the baseline with hop count, the integer hop count. The label is determined by hop count, so this baseline tests an explicitly available rule. The rare client class has only 20 test cases. One additional error changes client recall by five percentage points.

## Link and traffic assumptions

The generator has two scenarios. In `single_network`, the default and the scenario reported in the manuscript, all paths belong to one network. Every path has a bandwidth of 10 MHz, a mean signal-to-noise ratio of 14.5 dB and a mean utilization of 0.485. In `mixed_networks`, each case draws bandwidth uniformly from 5, 10 and 20 MHz, mean signal-to-noise ratio uniformly from 5 to 24 dB and mean utilization uniformly from 0.12 to 0.85. These three path-level draws are independent of the role. The values of the first scenario are the middle bandwidth and the midpoints of the two ranges of the second. The option `--network` of the generator sets other values for a single network. In both scenarios, each traversed hop receives independent Gaussian deviations of 3 dB in signal-to-noise ratio and 0.09 in utilization, both expressed as standard deviations. Clip the resulting hop signal-to-noise ratio to [-6, 35] dB and utilization to [0.02, 0.97]. Multipath fading, mobility dynamics and retransmissions have no explicit state in this model.

For bandwidth $B$ in MHz, hop signal-to-noise ratio $s_i$ in dB and utilization $u_i$, the idealized link capacity in Mbit/s is $C_i=B\log_2(1+10^{s_i/10})$. Background airtime use reduces the available rate to $R_i=C_i(1-u_i)$. The links use a single channel with serial forwarding and no spatial reuse. The active-hop sum $T=\sum_i 1/R_i$ is the transmission time in seconds per Mbit before losses. The simulator draws five sets of per-hop variables for every case to simplify vectorization and includes only the traversed hops in each path calculation.

## Case fields

| Field | Definition |
| --- | --- |
| `case_id` | Unique row identifier assigned after the hop counts are shuffled, such as `test-00001`. |
| `split` | Partition name, either `train` or `test`. |
| `role` | Path label under the hop-count convention above. |
| `hop_count` | Number of traversed radio hops. |
| `latency_ms` | For a 1,500-byte packet, begin with $12T$ ms. Add independent per-hop processing delays sampled uniformly from 0.12 to 0.75 ms and one endpoint delay sampled uniformly from 0.2 to 2.0 ms. Multiply by an independent lognormal timing factor with log mean 0 and log standard deviation 0.15. |
| `throughput_mbps` | Expected end-to-end goodput $p/T$ in Mbit/s, where $p$ is the product of the per-hop delivery probabilities. |
| `loss_fraction` | Fraction lost in a synthetic burst of 512 probes. Its count is a binomial draw with loss probability $1-p$. Each hop has loss probability $0.08/[1+\exp((s_i-6)/3)]+0.03u_i^3$. Delivery outcomes are modeled as independent across hops given the path conditions. |
| `load_fraction` | Arithmetic mean of utilization across the traversed hops. |
| `isolation_fraction` | Synthetic indicator of airtime protection. Draw one independent reservation fraction $r$ uniformly from 0.20 to 0.98. The path retains the smallest value of $1-(1-r)u_i$ across traversed hops. This indicator describes the D2D model's path conditions and has a separate definition from the slice isolation KPI. |

The per-hop equations and parameter distributions are shared by all classes. Hop count changes how many links contribute to each measurement. In the `single_network` scenario, the overlap between the classes comes from the per-hop deviations and from the random terms of the measurements in the table above. In the `mixed_networks` scenario, the path-level draws add a further source. Between the extremes of the three ranges, the residual rate of a hop differs by a factor of about 90, whereas the hop count changes the number of contributing links by a factor of at most five.
