import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('latency', 'sla_viol', 'throughput', 'isolation')


def read_records(path):
    files = sorted(path.glob('*.episodes.csv')) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f'No episode records in {path}')
    summaries = defaultdict(dict)
    for file in files:
        with file.open(newline='') as f:
            for row in csv.DictReader(f):
                key = row['method'], row['slice'], int(row['seed'])
                episode = int(row['episode'])
                if episode in summaries[key]:
                    raise ValueError(f'Duplicate episode {episode} in {key}')
                values = {m: float(row[m]) for m in (*METRICS, 'reward', 'msgs_per_s')}
                if not all(math.isfinite(x) for x in values.values()):
                    raise ValueError(f'Missing measurement in {key} episode {episode}')
                summaries[key][episode] = values
    return summaries


def seed_mean(summaries, method, slice_name, metric):
    per_seed = []
    for seed in range(3):
        episodes = summaries[(method, slice_name, seed)]
        if set(episodes) != set(range(20)):
            raise ValueError(f'Expected 20 evaluation windows (0-19) for {method}/{slice_name} seed {seed}')
        per_seed.append(fmean(x[metric] for x in episodes.values()))
    return fmean(per_seed)


def reference(name):
    with (ROOT / 'reference_tables' / name).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description='Regenerate Tables VIII, XI and XII and the rows of the proposed controller in Table IX from the episode records obtained through the BDIx agents. Python standard library only.')
    parser.add_argument('--records', type=Path, default=ROOT / 'code/validation/bdix/per_episode_records.csv',
                        help='Episode records of tools/replay_bdix.py: one CSV file or a directory of *.episodes.csv files.')
    parser.add_argument('--out', type=Path, required=True, help='New output directory for the reconstructed tables.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = read_records(args.records)
    cells = 0

    with (args.out / 'table_VIII.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('method', 'slice', *METRICS, 'msgs_per_s'))
        for row in reference('table_VIII_as_printed.csv'):
            shown = [f'{seed_mean(summaries, row["method"], row["slice"], m):.2f}' for m in METRICS]
            shown.append(f'{seed_mean(summaries, row["method"], row["slice"], "msgs_per_s"):.1f}')
            expected = [row[m] for m in (*METRICS, 'msgs_per_s')]
            if shown != expected:
                raise ValueError(f'Table VIII {row["method"]}/{row["slice"]} calculated {shown}, manuscript {expected}')
            writer.writerow((row['method'], row['slice'], *shown))
            cells += len(shown)

    with (args.out / 'table_IX_proposed.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('method', 'slice', *METRICS))
        for row in reference('table_IX_as_printed.csv'):
            if row['method'] != 'D-PPO':
                continue
            shown = [f'{seed_mean(summaries, row["method"], row["slice"], m):.2f}' for m in METRICS]
            expected = [row[m] for m in METRICS]
            if shown != expected:
                raise ValueError(f'Table IX {row["method"]}/{row["slice"]} calculated {shown}, manuscript {expected}')
            writer.writerow((row['method'], row['slice'], *shown))
            cells += len(shown)

    with (args.out / 'table_XI.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('method', 'slice', *METRICS))
        for row in reference('table_XI_as_printed.csv'):
            shown = [f'{seed_mean(summaries, row["method"], row["slice"], m):.2f}' for m in METRICS]
            expected = [row[m] for m in METRICS]
            if shown != expected:
                raise ValueError(f'Table XI {row["method"]}/{row["slice"]} calculated {shown}, manuscript {expected}')
            writer.writerow((row['method'], row['slice'], *shown))
            cells += len(shown)

    with (args.out / 'table_XII.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        slices = ('eMBB', 'uRLLC', 'mMTC')
        writer.writerow(('method', *slices))
        for row in reference('table_XII_as_printed.csv'):
            shown = [f'{100 * seed_mean(summaries, row["method"], s, "reward"):.2f}' for s in slices]
            expected = [row[s] for s in slices]
            if shown != expected:
                raise ValueError(f'Table XII {row["method"]} calculated {shown}, manuscript {expected}')
            writer.writerow((row['method'], *shown))
            cells += len(shown)

    print(f'PASS: all {cells} printed cells of Tables VIII, XI and XII and of the proposed controller in Table IX match the records obtained through the BDIx agents.')


if __name__ == '__main__':
    main()
