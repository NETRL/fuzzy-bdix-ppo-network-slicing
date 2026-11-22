import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('client', 'multihop_relay', 'relay')
LABELS = {'client': 'D2D_Client', 'multihop_relay': 'D2D_MultiHop_Relay', 'relay': 'D2D_Relay'}
RUNS = {'rf_without_hop': 'Random Forest', 'xgb_without_hop': 'XGBoost'}

def percent(value):
    return f'{100 * value:.2f}'

def load_counts(path):
    counts = {run: {role: {} for role in ROLES} for run in RUNS}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            run, truth, predicted = row['run'], row['true_role'], row['predicted_role']
            if run not in RUNS:
                continue
            if truth not in ROLES or predicted not in ROLES or predicted in counts[run][truth]:
                raise ValueError(f'Unexpected or repeated cell {run}/{truth}/{predicted}')
            counts[run][truth][predicted] = int(row['count'])
            if counts[run][truth][predicted] < 0:
                raise ValueError(f'Negative count in {run}/{truth}/{predicted}')
    if any(set(counts[run][role]) != set(ROLES) for run in RUNS for role in ROLES):
        raise ValueError('Expected nine cells for each classifier')
    return counts

def main():
    parser = argparse.ArgumentParser(description='Recalculate Table XVI from the confusion matrices of the supplementary D2D classification. Python standard library only.')
    parser.add_argument('--out', type=Path, required=True, help='New output directory for the recalculated tables.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts = load_counts(ROOT / 'supplementary_d2d/results/confusion_matrices.csv')
    with (ROOT / 'reference_tables/table_XVI_as_printed.csv').open(newline='') as f:
        expected = {(row['classifier'], row['class']): [row['precision'], row['recall'], row['f1']] for row in csv.DictReader(f)}
    if set(expected) != {(model, label) for model in RUNS.values() for label in LABELS.values()}:
        raise ValueError('Expected three reference classes for each classifier')
    with (args.out/'table_XVI.csv').open('w', newline='') as table:
        result = csv.writer(table)
        result.writerow(('classifier', 'class', 'support', 'precision', 'recall', 'f1'))
        for run, model in RUNS.items():
            matrix = counts[run]
            for role in ROLES:
                correct = matrix[role][role]
                support = sum(matrix[role].values())
                predicted = sum(matrix[other][role] for other in ROLES)
                if support == 0 or predicted == 0:
                    raise ValueError(f'Empty class {role}')
                scores = [percent(correct / predicted), percent(correct / support), percent(2 * correct / (support + predicted))]
                if scores != expected[model, LABELS[role]]:
                    raise ValueError(f'{model} {LABELS[role]} calculated {scores}, manuscript {expected[model, LABELS[role]]}')
                result.writerow((model, LABELS[role], support, *scores))
    with (args.out/'figure_19_accuracy.csv').open('w', newline='') as table:
        result = csv.writer(table)
        result.writerow(('model', 'cases', 'correct', 'accuracy', 'balanced_accuracy'))
        for run, model in RUNS.items():
            cases = sum(sum(row.values()) for row in counts[run].values())
            correct = sum(counts[run][role][role] for role in ROLES)
            recalls = [counts[run][role][role] / sum(counts[run][role].values()) for role in ROLES]
            result.writerow((model, cases, correct, percent(correct / cases), percent(sum(recalls) / len(ROLES))))
    print('PASS: all 18 printed scores in Table XVI follow from the two confusion matrices of Figure 19.')

if __name__ == '__main__':
    main()
