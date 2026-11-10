"""
kb_coverage_sweep.py — does the Blackboard actually mitigate hallucinations once the knowledge base has the answer?

The closed-book run (empty KB) can only turn hallucinations into abstentions. This sweep gives the pipeline
domain knowledge instead: the same model answers (stage 1 runs ONCE) are pushed through the Blackboard with the
supporting passages of 0%, 50% and 100% of the questions in ChromaDB (--kb_coverage, see eval_pipeline.py).

    0%    closed-book reference (nothing to retrieve)
    50%   items with their passage in the KB vs items without, inside one run
    100%  every question's passage is in the KB (retrieval still has to find it among all the others)

What to look for
    covered items      hallucinations should become CORRECT answers (corrections from evidence), not abstentions
    not-covered items  should abstain (or use web search with --stage2_args "--enable_web"), not hallucinate
    the gap between the two is what the knowledge base buys you; the 0% row is the old closed-book result.

Usage
    # GPU box: generate once, then sweep
    python kb_coverage_sweep.py --dataset HaluEval-main/data/qa_data.json --probe probe/probe.joblib \\
        --stage1_args "--limit 300 --shuffle --seed 0" --out_root eval_runs/kb_sweep

    # reuse existing answers (e.g. generations.jsonl from an earlier stage 1, no GPU needed)
    python kb_coverage_sweep.py --answers_file eval_runs/pilot/generations.jsonl --out_root eval_runs/kb_sweep

    # a real domain corpus as extra passages, different coverage levels
    python kb_coverage_sweep.py --answers_file gens.jsonl --kb_file my_domain_docs.txt --coverages 0,0.25,0.5,0.75,1

The dataset needs a `knowledge` field per question (HaluEval-QA has one). Finished runs resume from
<out_root>/cov_*/bb_rows.jsonl; eval_pipeline refuses to reuse them if the settings changed.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "eval_pipeline.py"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _arrow(e: Dict[str, Any], key: str) -> str:
    return f"{_pct(e['without_blackboard'][key])} -> {_pct(e['with_blackboard'][key])}"


def build_summary(reports: List[Tuple[float, Dict[str, Any]]]) -> str:
    """reports: [(coverage, metrics.json dict), ...] in the order they should appear."""
    n = reports[0][1]["without_blackboard"]["n"] if reports else 0
    lines = [f"# Knowledge-base coverage sweep (n={n} questions, same model answers in every row)", "",
             "## All questions (without -> with blackboard)", "",
             "| KB coverage | accuracy | hallucination | abstention | hallucination reduction (95% CI) |",
             "|---|---|---|---|---|"]
    for cov, rep in reports:
        lo, hi = 100 * rep["abs_reduction_ci95_low"], 100 * rep["abs_reduction_ci95_high"]
        lines.append(f"| {cov:g} | {_arrow(rep, 'accuracy')} | {_arrow(rep, 'hallucination_rate')} | "
                     f"{_arrow(rep, 'abstention_rate')} | {100 * rep['absolute_hallucination_reduction']:.1f} pts "
                     f"({lo:.1f} to {hi:.1f}) |")
    lines += ["", "## Split by whether the item's own passage is in the KB (with blackboard; baseline in brackets)", "",
              "| KB coverage | items | n | accuracy | hallucination | abstention |", "|---|---|---|---|---|---|"]
    for cov, rep in reports:
        sp = rep.get("by_kb_coverage")
        groups = ([("gold passage in KB", sp["covered"]), ("gold passage NOT in KB", sp["not_covered"])] if sp else
                  [("gold passage in KB" if cov >= 1 else "gold passage NOT in KB",
                    {"n": rep["without_blackboard"]["n"], "without_blackboard": rep["without_blackboard"],
                     "with_blackboard": rep["with_blackboard"]})])
        for label, e in groups:
            w, b = e["with_blackboard"], e["without_blackboard"]
            lines.append(f"| {cov:g} | {label} | {e['n']} | {_pct(w['accuracy'])} ({_pct(b['accuracy'])}) | "
                         f"{_pct(w['hallucination_rate'])} ({_pct(b['hallucination_rate'])}) | "
                         f"{_pct(w['abstention_rate'])} ({_pct(b['abstention_rate'])}) |")
    lines += ["", "Reading it: with the passage in the KB, hallucinations should become correct answers; without it they "
                  "can only become abstentions. Accuracy that rises with coverage while hallucination falls is the "
                  "mitigation working; a hallucination drop that is all abstention is the closed-book effect.", ""]
    return "\n".join(lines)


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"command failed with exit code {r.returncode}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="QA file with a `knowledge` field (e.g. HaluEval qa_data.json); runs stage 1 once")
    ap.add_argument("--answers_file", help="existing stage-1 generations.jsonl (skips stage 1)")
    ap.add_argument("--probe", help="probe .joblib, passed to stage 1")
    ap.add_argument("--coverages", default="0,0.5,1", help="comma-separated coverage levels in [0, 1]")
    ap.add_argument("--kb_seed", type=int, default=None, help="which items are covered (default 0)")
    ap.add_argument("--kb_file", default=None, help="extra domain corpus added to the KB in every run")
    ap.add_argument("--out_root", default="eval_runs/kb_sweep")
    ap.add_argument("--stage1_args", default="", help='extra args for stage 1, e.g. "--limit 300 --shuffle"')
    ap.add_argument("--stage2_args", default="", help='extra args for every stage-2 run, e.g. "--workers 4 --enable_web"')
    ap.add_argument("--share_memory", action="store_true",
                    help="let questions share episodic memory (default: isolated, so a memory hit never replaces retrieval)")
    args = ap.parse_args(argv)

    if bool(args.dataset) == bool(args.answers_file):
        ap.error("give exactly one of --dataset (run stage 1) or --answers_file (reuse answers)")
    covs = [float(x) for x in args.coverages.split(",") if x.strip()]
    if not covs or any(not 0.0 <= c <= 1.0 for c in covs):
        ap.error("--coverages must be numbers between 0 and 1")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    gens = args.answers_file
    if args.dataset:
        gens_path = out_root / "stage1" / "generations.jsonl"
        if gens_path.exists():
            print(f"[sweep] stage 1 already done -> {gens_path}")
        else:
            cmd = [py, str(PIPELINE), "--dataset", args.dataset, "--generate_only", "--out_dir", str(out_root / "stage1")]
            if args.probe:
                cmd += ["--probe", args.probe]
            _run(cmd + shlex.split(args.stage1_args))
        gens = str(gens_path)

    reports: List[Tuple[float, Dict[str, Any]]] = []
    for c in covs:
        out_dir = out_root / f"cov_{c:g}"
        cmd = [py, str(PIPELINE), "--answers_file", gens, "--kb_coverage", str(c), "--out_dir", str(out_dir),
               "--reuse_stage2"]
        if args.kb_seed is not None:
            cmd += ["--kb_seed", str(args.kb_seed)]
        if args.kb_file:
            cmd += ["--kb_file", args.kb_file]
        if not args.share_memory:
            cmd.append("--isolate_memory")
        _run(cmd + shlex.split(args.stage2_args))
        reports.append((c, json.loads((out_dir / "metrics.json").read_text())))

    md = build_summary(reports)
    (out_root / "sweep_summary.md").write_text(md, encoding="utf-8")
    (out_root / "sweep_summary.json").write_text(json.dumps({f"{c:g}": r for c, r in reports}, indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"Saved -> {out_root}/sweep_summary.md")


if __name__ == "__main__":
    main()
