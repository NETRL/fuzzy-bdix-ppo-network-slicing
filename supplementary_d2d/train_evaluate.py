import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import Booster, XGBClassifier


ROLES = ("client", "relay", "multihop_relay")
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
KPIS = ["latency_ms", "throughput_mbps", "loss_fraction", "load_fraction", "isolation_fraction"]
MODEL_SEED = 317


def make_estimator(family):
    if family == "rf":
        return RandomForestClassifier(
            n_estimators=120, max_depth=10, min_samples_leaf=3, max_features="sqrt",
            bootstrap=True, random_state=MODEL_SEED, n_jobs=1,
        )
    return XGBClassifier(
        n_estimators=120, max_depth=5, learning_rate=0.08, min_child_weight=1.0,
        subsample=1.0, colsample_bytree=1.0, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", tree_method="hist", device="cpu", max_bin=256,
        random_state=MODEL_SEED, n_jobs=1,
    )


def predict_from_saved_model(model, name, folder, cases):
    if isinstance(model, RandomForestClassifier):
        path = folder / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        return joblib.load(path).predict(cases)
    path = folder / f"{name}.json"
    model.get_booster().save_model(path)
    restored = Booster(params={"nthread": 1, "device": "cpu"}, model_file=path)
    return restored.inplace_predict(cases).argmax(axis=1)


def write_scores(actual, predictions, directory):
    summaries, scores, matrices = [], [], []
    for run, labels in predictions.items():
        matrix = confusion_matrix(actual, labels, labels=list(ROLES))
        precision, recall, f1, support = precision_recall_fscore_support(
            actual, labels, labels=list(ROLES), zero_division=0,
        )
        correct = int(matrix.trace())
        summaries.append([run, len(actual), correct, correct / len(actual), recall.mean(), f1.mean()])
        for index, role in enumerate(ROLES):
            scores.append([run, role, support[index], precision[index], recall[index], f1[index]])
        for i, j in product(range(len(ROLES)), repeat=2):
            matrices.append([run, ROLES[i], ROLES[j], matrix[i, j]])
    tables = (
        ("summary.csv", summaries, ["run", "n_test", "correct", "accuracy", "balanced_accuracy", "macro_f1"]),
        ("class_scores.csv", scores, ["run", "role", "support", "precision", "recall", "f1"]),
        ("confusion_matrices.csv", matrices, ["run", "true_role", "predicted_role", "count"]),
    )
    for filename, rows, columns in tables:
        pd.DataFrame(rows, columns=columns).to_csv(
            directory / filename, index=False, float_format="%.10f", lineterminator="\n",
        )


def train_evaluate(data_path, output):
    cases = pd.read_csv(data_path)
    if cases.case_id.isna().any() or cases.case_id.duplicated().any():
        raise ValueError("Expected a unique identifier for each case.")
    if set(cases.split) != {"train", "test"} or not cases.role.isin(ROLES).all():
        raise ValueError("Unexpected partition or role.")
    train = cases.loc[cases.split.eq("train")]
    test = cases.loc[cases.split.eq("test")]
    if set(train.role) != set(ROLES) or set(test.role) != set(ROLES):
        raise ValueError("Each partition must contain every role.")
    if not np.isfinite(cases[[*KPIS, "hop_count"]].to_numpy()).all():
        raise ValueError("Nonfinite predictor value.")
    targets = train.role.map(ROLE_INDEX).to_numpy()
    weights = compute_sample_weight("balanced", targets)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        model_dir = stage / "models"
        model_dir.mkdir()
        predictions = {}
        for family in ("rf", "xgb"):
            for feature_set, columns in (("with_hop", [*KPIS, "hop_count"]), ("without_hop", KPIS)):
                run = f"{family}_{feature_set}"
                model = make_estimator(family)
                model.fit(train[columns], targets, sample_weight=weights)
                encoded = model.predict(test[columns])
                reloaded_predictions = predict_from_saved_model(model, run, model_dir, test[columns])
                np.testing.assert_array_equal(encoded, reloaded_predictions)
                predictions[run] = np.asarray(ROLES)[encoded.astype(int)]
        output_rows = test[["case_id", "role"]].rename(columns={"role": "true_role"})
        for run, labels in predictions.items():
            output_rows[run] = labels
        output_rows.to_csv(stage / "predictions.csv", index=False, lineterminator="\n")
        write_scores(test.role, predictions, stage)
        (output / "models").mkdir(exist_ok=True)
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                path.replace(output / path.relative_to(stage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit and evaluate D2D role classifiers on the synthetic benchmark.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    train_evaluate(arguments.data, arguments.out)
