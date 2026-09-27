"""
Interactive manual hallucination detection pipeline.
Generate answers and manually verify hallucinations.

Usage:
  python build_hallucination_dataset_manual.py \
      --local_path HaluEval-main/data \
      --max_questions 100 \
      --batch_size 1 \
      --output_path halueval_manual_features.pt
"""

import argparse
import glob
import os

import torch
import numpy as np
from datasets import load_dataset

from llama_features import LlamaSAEFeatureExtractor

HALUEVAL_HF_CANDIDATES = [
    ("pminervini/HaluEval", "qa"),
    ("notrickai/HaluEval", "qa"),
]


def _resolve_local_files(local_path, preferred_filename="qa_data.json"):
    if os.path.isdir(local_path):
        preferred = os.path.join(local_path, preferred_filename)
        if os.path.isfile(preferred):
            return preferred
        candidates = sorted(
            glob.glob(os.path.join(local_path, "*.json"))
            + glob.glob(os.path.join(local_path, "*.jsonl"))
        )
        if not candidates:
            raise FileNotFoundError(f"No .json/.jsonl files found directly inside {local_path}")
        raise FileNotFoundError(
            f"Couldn't find {preferred_filename} inside {local_path}. "
            f"Found instead: {[os.path.basename(c) for c in candidates]}."
        )
    return local_path


def load_halueval_qa(local_path=None, split="data", file_name="qa_data.json"):
    if local_path:
        data_files = _resolve_local_files(local_path, preferred_filename=file_name)
        print(f"Loading local file: {data_files}")
        return load_dataset("json", data_files=data_files, split="train")

    last_err = None
    for repo, config in HALUEVAL_HF_CANDIDATES:
        try:
            return load_dataset(repo, config, split=split)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(
        "Could not auto-load HaluEval from the Hub. Pass --local_path instead. "
        f"Last error: {last_err}"
    )


def get_manual_label():
    """Get user input for hallucination status."""
    while True:
        print("  Options:")
        print("    [1] Hallucinated (wrong answer)")
        print("    [0] Correct (right answer)")
        print("    [s] Skip")
        print("    [q] Quit")
        user_input = input("\nEnter choice (1/0/s/q): ").strip().lower()
        if user_input == '1':
            return 1, True  # hallucinated, continue
        elif user_input == '0':
            return 0, True  # not hallucinated, continue
        elif user_input == 's':
            return None, True  # skip this, continue
        elif user_input == 'q':
            return None, False  # quit
        else:
            print("  ❌ Invalid input. Type '1' (hallucinated), '0' (correct), 's' (skip), or 'q' (quit)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path", type=str, default=None,
                        help="Path to local HaluEval data directory or file")
    parser.add_argument("--file_name", type=str, default="qa_data.json",
                        help="Name of the data file within local_path")
    parser.add_argument("--max_questions", type=int, default=100,
                        help="Total number of questions to process")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start from this index (0-based). Use to resume from checkpoint")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End at this index. If None, uses start_idx + max_questions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (recommended: 1 for manual review)")
    parser.add_argument("--output_path", type=str, default="halueval_manual_features.pt",
                        help="Output path for the saved dataset")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Checkpoint every N samples")
    args = parser.parse_args()

    print("Loading HaluEval dataset...")
    dataset = load_halueval_qa(local_path=args.local_path, file_name=args.file_name)
    print(f"Columns: {dataset.column_names}")
    print(f"Total dataset size: {len(dataset)}")
    
    # Calculate range
    end_idx = args.end_idx if args.end_idx else min(args.start_idx + args.max_questions, len(dataset))
    dataset = dataset.select(range(args.start_idx, end_idx))
    print(f"Processing questions {args.start_idx} to {end_idx-1} ({len(dataset)} questions)\n")

    print("Loading Qwen (layer 20 residual stream)...")
    extractor = LlamaSAEFeatureExtractor()

    records = []
    n = len(dataset)
    hallucination_count = 0
    skipped_count = 0
    sample_count = 0
    
    for idx in range(n):
        sample = dataset[idx]
        actual_sample_num = args.start_idx + idx + 1  # Actual position in original dataset
        question = sample["question"]
        ground_truth = sample["right_answer"]

        print(f"\n{'='*80}")
        print(f"Sample {actual_sample_num} (Range: {args.start_idx}-{end_idx-1})")
        print(f"{'='*80}")

        print(f"\n❓ Question:")
        print(f"   {question}\n")

        print(f"✓ Ground Truth:")
        print(f"   {ground_truth}\n")

        # Generate answer
        print("🔄 Generating answer with Qwen...")
        gen_result = extractor.batch_generate_and_extract_features(
            [question],
            max_new_tokens=32,
        )[0]

        generated_answer = gen_result["answer_text"]
        feature_vector = gen_result["feature_vector"]

        print(f"\n🤖 Generated Answer:")
        print(f"   {generated_answer}\n")

        # Get manual label
        is_hallucinated, should_continue = get_manual_label()

        if not should_continue:
            print("\n⏹️  Quitting...")
            break

        if is_hallucinated is None:
            print("⏭️  Skipping this sample...")
            skipped_count += 1
            continue

        # Save record
        record = {
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "is_hallucinated": is_hallucinated,
            "manual_label": True,  # Marked as manually labeled
            "feature_vector": feature_vector.cpu(),
        }
        records.append(record)
        sample_count += 1

        if is_hallucinated:
            hallucination_count += 1
            print("✅ Labeled as: HALLUCINATED")
        else:
            print("✅ Labeled as: CORRECT")

        # Checkpoint
        if sample_count % args.save_every == 0:
            torch.save(records, args.output_path)
            print(f"\n💾 Checkpoint saved ({sample_count} labeled samples)")

    # Final save
    torch.save(records, args.output_path)
    
    print(f"\n{'='*80}")
    print(f"✓ Done! Dataset saved to {args.output_path}")
    print(f"{'='*80}")
    print(f"Total labeled: {sample_count}")
    print(f"  - Hallucinated: {hallucination_count} ({hallucination_count/sample_count*100:.1f}%)" if sample_count > 0 else "")
    print(f"  - Correct: {sample_count - hallucination_count} ({(sample_count-hallucination_count)/sample_count*100:.1f}%)" if sample_count > 0 else "")
    print(f"Total skipped: {skipped_count}")
    print(f"Detection method: Manual review")
    print(f"Layer: 20 (raw residual stream, no SAE)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()