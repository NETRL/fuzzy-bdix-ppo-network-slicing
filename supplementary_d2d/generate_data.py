import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


MAX_HOPS = 5
SEED = 628104
SCENARIOS = {
    "single_network": ((10.0,), (14.5, 14.5), (0.485, 0.485)),
    "mixed_networks": ((5.0, 10.0, 20.0), (5.0, 24.0), (0.12, 0.85)),
}
FIELDS = (
    "case_id", "split", "role", "hop_count", "latency_ms", "throughput_mbps",
    "loss_fraction", "load_fraction", "isolation_fraction",
)


def sample_partition(split, random, conditions):
    bandwidths, snr_range, load_range = conditions
    scale = {"train": 4, "test": 1}[split]
    hops = np.concatenate((
        np.ones(20 * scale, dtype=int),
        np.full(280 * scale, 2, dtype=int),
        random.integers(3, MAX_HOPS + 1, size=8700 * scale),
    ))
    random.shuffle(hops)
    count = len(hops)
    shape = (count, MAX_HOPS)
    active = np.arange(MAX_HOPS) < hops[:, None]

    bandwidth_mhz = random.choice(bandwidths, size=(count, 1))
    mean_snr_db = random.uniform(*snr_range, size=(count, 1))
    snr_db = np.clip(mean_snr_db + random.normal(0.0, 3.0, size=shape), -6.0, 35.0)
    mean_load = random.uniform(*load_range, size=(count, 1))
    utilization = np.clip(mean_load + random.normal(0.0, 0.09, size=shape), 0.02, 0.97)
    capacity_mbps = bandwidth_mhz * np.log2(1.0 + 10.0 ** (snr_db / 10.0))
    residual_mbps = capacity_mbps * (1.0 - utilization)
    time_per_mbit = np.sum(np.where(active, 1.0 / residual_mbps, 0.0), axis=1)

    radio_loss = 0.08 / (1.0 + np.exp((snr_db - 6.0) / 3.0))
    link_loss = radio_loss + 0.03 * utilization ** 3
    path_success = np.prod(np.where(active, 1.0 - link_loss, 1.0), axis=1)
    loss = random.binomial(512, 1.0 - path_success) / 512.0
    throughput = path_success / time_per_mbit

    processing_ms = random.uniform(0.12, 0.75, size=shape)
    endpoint_ms = random.uniform(0.2, 2.0, size=count)
    latency = endpoint_ms + 12.0 * time_per_mbit
    latency += np.sum(np.where(active, processing_ms, 0.0), axis=1)
    latency *= random.lognormal(mean=0.0, sigma=0.15, size=count)

    load = np.sum(np.where(active, utilization, 0.0), axis=1) / hops
    reservation = random.uniform(0.20, 0.98, size=(count, 1))
    retained_airtime = 1.0 - (1.0 - reservation) * utilization
    isolation = np.min(np.where(active, retained_airtime, 1.0), axis=1)
    roles = np.select([hops == 1, hops == 2], ["client", "relay"], default="multihop_relay")

    for index in range(count):
        kpis = (latency[index], throughput[index], loss[index], load[index], isolation[index])
        yield (
            f"{split}-{index + 1:05d}", split, roles[index], int(hops[index]),
            *(f"{value:.8f}" for value in kpis),
        )


def generate(output, seed, scenario, network=None):
    conditions = SCENARIOS[scenario]
    if network is not None:
        bandwidth_mhz, mean_snr_db, mean_load = network
        conditions = ((bandwidth_mhz,), (mean_snr_db, mean_snr_db), (mean_load, mean_load))
    output.mkdir(parents=True, exist_ok=True)
    partitions = ("train", "test")
    streams = np.random.SeedSequence(seed).spawn(len(partitions))
    with TemporaryDirectory(dir=output) as staging:
        staged_cases = Path(staging) / "cases.csv"
        with staged_cases.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerow(FIELDS)
            for split, stream in zip(partitions, streams):
                random = np.random.Generator(np.random.PCG64(stream))
                writer.writerows(sample_partition(split, random, conditions))
        staged_cases.replace(output / "cases.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic D2D path cases.")
    parser.add_argument("--out", type=Path, required=True, help="Directory for cases.csv.")
    parser.add_argument("--seed", type=int, default=SEED, help="Nonnegative root seed.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="single_network",
                        help="Link conditions shared by all paths or drawn for every path.")
    parser.add_argument("--network", type=float, nargs=3, metavar=("MHZ", "SNR_DB", "UTILIZATION"),
                        help="Bandwidth, mean signal-to-noise ratio and mean utilization of a single network.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.network is not None:
        if args.scenario != "single_network":
            parser.error("--network describes a single network")
        if args.network[0] <= 0 or not 0 < args.network[2] < 1:
            parser.error("--network needs a positive bandwidth and a utilization between 0 and 1")
    generate(args.out, args.seed, args.scenario, args.network)
