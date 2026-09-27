"""
Train a logistic regression probe on the feature/label dataset produced by
Dataset_builder.py (a .pt file: list of dicts with at least
"feature_vector" and "is_hallucinated").

Usage:
  python train_probe.py \
      --data halueval_manual_features.pt \
      --output probe.joblib
"""

import argparse
import json

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def load_records(data_path):
    records = torch.load(data_path, map_location="cpu", weights_only=False)
    if not records:
        raise ValueError(f"{data_path} contains no records.")

    missing_fv = [r for r in records if "feature_vector" not in r or r["feature_vector"] is None]
    if missing_fv:
        raise ValueError(
            f"{len(missing_fv)} of {len(records)} records have no feature_vector "
            "— re-run Dataset_builder.py against the updated feature extractor first."
        )

    X = np.stack([r["feature_vector"].to(torch.float32).numpy() for r in records])
    y = np.array([int(r["is_hallucinated"]) for r in records])
    groups = np.array([r.get("question", str(i)) for i, r in enumerate(records)])
    return X, y, groups


def split(X, y, groups, test_size, seed):
    """Group-split by question when there are duplicate questions (so the
    same question can't leak between train and test); otherwise falls back
    to a plain stratified split."""
    n_unique_groups = len(set(groups))
    if n_unique_groups < len(groups):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y))
    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the .pt dataset from Dataset_builder.py")
    parser.add_argument("--output", type=str, default="probe.joblib",
                        help="Where to save the trained probe (scaler + classifier)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--class_weight", type=str, default="balanced",
                        choices=["balanced", "none"],
                        help="'balanced' reweights classes — recommended unless your "
                             "hallucinated/correct counts are already close to even")
    args = parser.parse_args()

    print(f"Loading records from {args.data}...")
    X, y, groups = load_records(args.data)
    n_pos = int(y.sum())
    print(f"Loaded {len(y)} examples: {n_pos} hallucinated, {len(y) - n_pos} correct "
          f"({n_pos / len(y) * 100:.1f}% positive). Feature dim: {X.shape[1]}")

    if len(set(y)) < 2:
        raise ValueError("Only one class present in the labels — need both hallucinated "
                          "and correct examples to train a classifier.")

    train_idx, test_idx = split(X, y, groups, args.test_size, args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train: {len(y_train)} ({y_train.sum()} hallucinated) | "
          f"Test: {len(y_test)} ({y_test.sum()} hallucinated)")

    # Standardize — matters a lot here: raw residual-stream dims have very
    # different scales/variances, unlike sparse SAE features which are
    # mostly-zero and roughly comparable in magnitude already.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    class_weight = None if args.class_weight == "none" else "balanced"
    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=args.seed,
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present in y_test

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["correct", "hallucinated"]))
    print("Confusion matrix (rows=true, cols=pred, order=[correct, hallucinated]):")
    print(confusion_matrix(y_test, y_pred))

    # 5-fold-ish sanity check on train set via the classifier's own score,
    # to flag obvious overfitting if train accuracy is much higher than test
    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    print(f"\nTrain accuracy: {train_acc:.4f} (vs test {acc:.4f})")
    if train_acc - acc > 0.15:
        print("Warning: large train/test gap — consider more data, stronger "
              "regularization (lower --C), or checking for leakage.")

    bundle = {
        "scaler": scaler,
        "classifier": clf,
        "feature_dim": X.shape[1],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_metrics": {"accuracy": acc, "f1": f1, "roc_auc": auc},
        "class_weight": class_weight,
        "C": args.C,
    }
    joblib.dump(bundle, args.output)
    print(f"\nSaved probe to {args.output}")

    metrics_path = args.output.rsplit(".", 1)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "accuracy": acc, "f1": f1, "roc_auc": auc,
                "n_train": len(y_train), "n_test": len(y_test),
                "feature_dim": int(X.shape[1]),
            },
            f, indent=2,
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
