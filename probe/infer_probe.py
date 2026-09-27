"""
Run the trained hallucination probe.

Two modes:

1) Score a .pt file of records that already have feature_vector computed
   (e.g. more output from Dataset_builder.py, or a held-out .pt file).
   No GPU / model loading needed:

     python infer_probe.py --probe probe.joblib --data new_records.pt

2) Take raw questions, generate an answer with Qwen, extract the feature
   vector, and classify — needs a GPU (loads QwenResidualFeatureExtractor):

     python infer_probe.py --probe probe.joblib \
         --questions "What is the capital of Australia?" "Who wrote Hamlet?"

     python infer_probe.py --probe probe.joblib --questions_file qs.txt

3) Interactive: load the model once, then keep typing questions like a
   chatbot — no reload between questions. Type 'quit' or 'exit' to stop:

     python infer_probe.py --probe probe.joblib --interactive
"""

import argparse
import csv
import sys

import joblib
import numpy as np
import torch


def load_probe(probe_path):
    bundle = joblib.load(probe_path)
    return bundle["scaler"], bundle["classifier"], bundle.get("feature_dim")


def classify(scaler, clf, X, feature_dim=None):
    X = np.asarray(X, dtype=np.float32)
    if feature_dim is not None and X.shape[1] != feature_dim:
        raise ValueError(
            f"Feature dim mismatch: probe was trained on dim {feature_dim}, "
            f"got dim {X.shape[1]}. Are you using the same layer/model as training?"
        )
    X_s = scaler.transform(X)
    preds = clf.predict(X_s)
    probs = clf.predict_proba(X_s)[:, 1]  # P(hallucinated)
    return preds, probs


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} results to {out_path}")


def mode_from_data_file(args, scaler, clf, feature_dim):
    records = torch.load(args.data, map_location="cpu", weights_only=False)
    missing = [r for r in records if "feature_vector" not in r or r["feature_vector"] is None]
    if missing:
        raise ValueError(f"{len(missing)} records have no feature_vector; can't score them.")

    X = np.stack([r["feature_vector"].to(torch.float32).numpy() for r in records])
    preds, probs = classify(scaler, clf, X, feature_dim)

    rows = []
    n_correct_labeled = 0
    for r, pred, prob in zip(records, preds, probs):
        row = {
            "question": r.get("question", ""),
            "answer": r.get("generated_answer", r.get("answer_text", "")),
            "predicted_hallucinated": int(pred),
            "prob_hallucinated": round(float(prob), 4),
        }
        if "is_hallucinated" in r and r["is_hallucinated"] is not None:
            row["true_hallucinated"] = int(r["is_hallucinated"])
            n_correct_labeled += int(row["true_hallucinated"] == row["predicted_hallucinated"])
        rows.append(row)
        print(f"[{'HALLUCINATED' if pred else 'correct':>12}] "
              f"p={prob:.3f}  {row['question'][:70]}")

    if any("true_hallucinated" in r for r in rows):
        labeled = [r for r in rows if "true_hallucinated" in r]
        acc = sum(r["true_hallucinated"] == r["predicted_hallucinated"] for r in labeled) / len(labeled)
        print(f"\nAccuracy against ground-truth labels in the file: {acc:.4f} ({len(labeled)} examples)")

    if args.output:
        write_csv(rows, args.output)


def mode_from_questions(args, scaler, clf, feature_dim):
    from llama_features import QwenResidualFeatureExtractor

    questions = list(args.questions) if args.questions else []
    if args.questions_file:
        with open(args.questions_file) as f:
            questions += [line.strip() for line in f if line.strip()]
    if not questions:
        print("No questions given (use --questions or --questions_file).", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Qwen (layer {args.layer})...")
    extractor = QwenResidualFeatureExtractor(layer=args.layer)

    rows = []
    batch_size = args.batch_size
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        results = extractor.batch_generate_and_extract_features(batch, max_new_tokens=32)
        X = np.stack([r["feature_vector"].cpu().to(torch.float32).numpy() for r in results])
        preds, probs = classify(scaler, clf, X, feature_dim)
        for q, r, pred, prob in zip(batch, results, preds, probs):
            row = {
                "question": q,
                "answer": r["answer_text"],
                "predicted_hallucinated": int(pred),
                "prob_hallucinated": round(float(prob), 4),
            }
            rows.append(row)
            print(f"\n[{'HALLUCINATED' if pred else 'correct':>12}] p={prob:.3f}")
            print(f"  Q: {q}")
            print(f"  A: {r['answer_text']}")

    if args.output:
        write_csv(rows, args.output)


def mode_interactive(args, scaler, clf, feature_dim):
    from llama_features import QwenResidualFeatureExtractor

    print(f"Loading Qwen (layer {args.layer})... (one-time load, stays in memory)")
    extractor = QwenResidualFeatureExtractor(layer=args.layer)
    print("\nReady. Type a question and press Enter. Type 'quit' or 'exit' to stop.\n")

    rows = []
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        results = extractor.batch_generate_and_extract_features([question], max_new_tokens=32)
        r = results[0]
        X = r["feature_vector"].cpu().to(torch.float32).numpy()[None, :]
        preds, probs = classify(scaler, clf, X, feature_dim)
        pred, prob = int(preds[0]), float(probs[0])

        label = "HALLUCINATED" if pred else "correct"
        print(f"  A: {r['answer_text']}")
        print(f"  -> [{label}]  P(hallucinated)={prob:.3f}\n")

        rows.append({
            "question": question,
            "answer": r["answer_text"],
            "predicted_hallucinated": pred,
            "prob_hallucinated": round(prob, 4),
        })

    if args.output and rows:
        write_csv(rows, args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=str, required=True, help="Path to trained probe .joblib")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a .pt file of records with precomputed feature_vector")
    parser.add_argument("--questions", type=str, nargs="*", default=None,
                        help="One or more questions to generate answers for and classify")
    parser.add_argument("--questions_file", type=str, default=None,
                        help="Text file, one question per line")
    parser.add_argument("--interactive", action="store_true",
                        help="Load the model once, then classify questions typed in a loop")
    parser.add_argument("--layer", type=int, default=20,
                        help="Must match the layer the probe was trained on")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional CSV path to save results (for --data/--questions modes; "
                             "for --interactive, saved when you quit)")
    args = parser.parse_args()

    modes_given = sum(bool(x) for x in (args.data, args.questions, args.questions_file, args.interactive))
    if modes_given == 0:
        parser.error("Provide one of --data, --questions/--questions_file, or --interactive.")
    if modes_given > 1:
        parser.error("Use only one of --data, --questions/--questions_file, or --interactive.")

    scaler, clf, feature_dim = load_probe(args.probe)

    if args.data:
        mode_from_data_file(args, scaler, clf, feature_dim)
    elif args.interactive:
        mode_interactive(args, scaler, clf, feature_dim)
    else:
        mode_from_questions(args, scaler, clf, feature_dim)


if __name__ == "__main__":
    main()
