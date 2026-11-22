import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('latency', 'sla_viol', 'throughput', 'isolation')

def main():
    parser = argparse.ArgumentParser(description='Regenerate Table XI from the recorded test windows. Python standard library only.')
    parser.add_argument('--out', type=Path, required=True, help='New output directory for the reconstructed tables.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = defaultdict(dict)
    data = ROOT / 'code/campaign/results_fixed4/per_episode_records.csv'
    with data.open(newline='') as f:
        for row in csv.DictReader(f):
            key = row['method'], row['slice'], int(row['seed'])
            episode = int(row['episode'])
            if episode in summaries[key]:
                raise ValueError(f'Duplicate episode {episode} in {key}')
            metrics = {m: float(row[m]) for m in METRICS}
            if not all(math.isfinite(x) for x in metrics.values()):
                raise ValueError(f'Missing measurement in {key} episode {episode}')
            summaries[key][episode] = metrics
    with (ROOT / 'reference_tables/table_XI_as_printed.csv').open(newline='') as f:
        expected_rows = list(csv.DictReader(f))
    if len(expected_rows) != 18:
        raise ValueError('Expected 18 reference method/slice rows')
    with (args.out/'table_XI.csv').open('w', newline='') as table, (args.out/'table_XI_by_seed.csv').open('w', newline='') as seeds:
        result = csv.writer(table)
        detail = csv.writer(seeds)
        result.writerow(('method', 'slice', *METRICS))
        detail.writerow(('method', 'slice', 'seed', *METRICS))
        for row in expected_rows:
            averages = []
            for seed in range(3):
                key = row['method'], row['slice'], seed
                episodes = summaries[key]
                if set(episodes) != set(range(20)):
                    raise ValueError(f'Expected 20 evaluation windows (0–19) in {key}')
                means = [fmean(x[k] for x in episodes.values()) for k in METRICS]
                averages.append(means)
                detail.writerow((*key, *means))
            rounded = [f'{fmean(col):.2f}' for col in zip(*averages)]
            expected = [row[k] for k in METRICS]
            if rounded != expected:
                raise ValueError(f"{row['method']}/{row['slice']} calculated {rounded}, manuscript {expected}")
            result.writerow((row['method'], row['slice'], *rounded))
    print('PASS: all 72 printed KPI cells in Table XI match the recorded 3-seed × 20-window means.')

if __name__ == '__main__':
    main()
