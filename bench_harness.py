"""
HalluciGuard benchmark harness.

Use inside the notebook (after running cells 0-13):

    exec(open("bench_harness.py").read())
    run_benchmark(orchestrator, memory_collection, knowledge_collection, label="after")

For the "before" numbers, run the same two lines in a copy of the notebook
from the baseline commit, with label="before". Then:

    compare_results()

It never calls the Gemini API by itself. Only orchestrator.run() does.
"""
import csv
import json
import re
import time

_DATA_FILE = "benchmark_data.json"
_LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]


def _clear_memory(memory_collection):
    ids = memory_collection.get().get("ids", [])
    if ids:
        memory_collection.delete(ids=ids)


def _numbers(text):
    """All numbers in a text, with thousands commas removed."""
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")}


def _evidence_text(orchestrator):
    try:
        ev = orchestrator.blackboard.read("retrieved_evidence", []) or []
        return " ".join(str(e.get("text", "")) for e in ev)
    except Exception:
        return ""


def _one_run(orchestrator, item):
    return orchestrator.run(
        prompt=item["prompt"],
        response=item["response"],
        claim=item["claim"],
        confidence_score=0.9,
    )


def _pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def run_benchmark(orchestrator, memory_collection, knowledge_collection,
                  label, data_file=_DATA_FILE, pause=0.5):
    data = json.load(open(data_file, encoding="utf-8"))

    # 1) make sure the benchmark documents are in the knowledge base
    knowledge_collection.upsert(
        ids=[d["id"] for d in data["documents"]],
        documents=[d["text"] for d in data["documents"]],
        metadatas=[{"source": "benchmark"} for _ in data["documents"]],
    )
    print(f"[{label}] knowledge base size: {knowledge_collection.count()}")

    # ---------------- Pass 1: verdicts and corrections, fresh memory per claim ----------------
    rows = []
    _consec = 0
    for k, item in enumerate(data["claims"], start=1):
        _clear_memory(memory_collection)
        row = {"id": item["id"], "gold": item["gold"], "claim": item["claim"],
               "predicted": "", "correct": "", "source": "", "n_evidence_ids": "",
               "corrected": "", "new_numbers": "", "final_response": "", "error": ""}
        try:
            out = _one_run(orchestrator, item)
            vr = out.get("verification_result") or {}
            final = out.get("final_response") or ""
            new_nums = _numbers(final) - _numbers(item["response"]) - _numbers(_evidence_text(orchestrator))
            row.update({
                "predicted": vr.get("verdict", ""),
                "correct": vr.get("verdict", "") == item["gold"],
                "source": out.get("verification_source", ""),
                "n_evidence_ids": len(vr.get("supporting_evidence_ids") or []),
                "corrected": out.get("correction_result") is not None,
                "new_numbers": " ".join(sorted(new_nums)),
                "final_response": final,
            })
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
        _consec = (_consec + 1) if row["error"] else 0
        if _consec >= 3:
            raise RuntimeError(
                "Stopping: 3 errors in a row (fix this before re-running). Last error: " + row["error"]
            )
        flag = "ERR" if row["error"] else ("ok " if row["correct"] else "MISS")
        print(f"  {k:02d}/{len(data['claims'])} {flag} gold={item['gold']:<12} pred={row['predicted']:<12} corrected={row['corrected']}")
        time.sleep(pause)

    with open(f"bench_results_{label}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- Pass 2: memory test on INSUFFICIENT claims (run each twice) ----------------
    mem_rows = []
    _consec2 = 0
    for item in [c for c in data["claims"] if c["gold"] == "INSUFFICIENT"]:
        _clear_memory(memory_collection)
        mrow = {"id": item["id"], "claim": item["claim"], "saved_after_run1": "",
                "run2_source": "", "run2_confidence": "", "contaminated": "", "error": ""}
        try:
            _one_run(orchestrator, item)
            mrow["saved_after_run1"] = memory_collection.count()
            out2 = _one_run(orchestrator, item)
            vr2 = out2.get("verification_result") or {}
            mrow["run2_source"] = out2.get("verification_source", "")
            mrow["run2_confidence"] = vr2.get("confidence", "")
            mrow["contaminated"] = out2.get("verification_source") == "episodic_memory"
        except Exception as e:
            mrow["error"] = f"{type(e).__name__}: {e}"[:200]
        mem_rows.append(mrow)
        _consec2 = (_consec2 + 1) if mrow["error"] else 0
        if _consec2 >= 3:
            raise RuntimeError(
                "Stopping: 3 errors in a row in the memory test. Last error: " + mrow["error"]
            )
        print(f"  mem {item['id']} saved={mrow['saved_after_run1']} run2={mrow['run2_source']} contaminated={mrow['contaminated']}")
        time.sleep(pause)

    with open(f"bench_memory_{label}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mem_rows[0].keys()))
        w.writeheader()
        w.writerows(mem_rows)

    _clear_memory(memory_collection)  # leave memory clean

    # ---------------- Summary ----------------
    ok_rows = [r for r in rows if not r["error"]]
    insuff = [r for r in ok_rows if r["gold"] == "INSUFFICIENT"]
    corrected = [r for r in ok_rows if r["corrected"] is True]
    mem_ok = [m for m in mem_rows if not m["error"]]

    summary = {
        "label": label,
        "n_claims": len(rows),
        "n_errors": len(rows) - len(ok_rows) + (len(mem_rows) - len(mem_ok)),
        "accuracy_pct": _pct(sum(1 for r in ok_rows if r["correct"] is True), len(ok_rows)),
        "accuracy_by_class_pct": {
            g: _pct(sum(1 for r in ok_rows if r["gold"] == g and r["correct"] is True),
                    sum(1 for r in ok_rows if r["gold"] == g))
            for g in _LABELS
        },
        # bug #1: claims the KB can't verify that still got a correction
        "unsupported_correction_pct": _pct(sum(1 for r in insuff if r["corrected"] is True), len(insuff)),
        # corrected answers that contain numbers found in neither the claim nor the evidence
        "corrections_with_invented_numbers": sum(1 for r in corrected if r["new_numbers"]),
        "n_corrections": len(corrected),
        # bug #2: unverifiable results saved to memory and replayed
        "ungrounded_saved_to_memory_pct": _pct(sum(1 for m in mem_ok if (m["saved_after_run1"] or 0) > 0), len(mem_ok)),
        "memory_contamination_pct": _pct(sum(1 for m in mem_ok if m["contaminated"] is True), len(mem_ok)),
    }
    json.dump(summary, open(f"bench_summary_{label}.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: bench_results_{label}.csv, bench_memory_{label}.csv, bench_summary_{label}.json")
    return summary


def compare_results(before="before", after="after"):
    b = json.load(open(f"bench_summary_{before}.json"))
    a = json.load(open(f"bench_summary_{after}.json"))
    lines = [
        ("Verdict accuracy (%)", b["accuracy_pct"], a["accuracy_pct"]),
        ("  SUPPORTED accuracy (%)", b["accuracy_by_class_pct"]["SUPPORTED"], a["accuracy_by_class_pct"]["SUPPORTED"]),
        ("  CONTRADICTED accuracy (%)", b["accuracy_by_class_pct"]["CONTRADICTED"], a["accuracy_by_class_pct"]["CONTRADICTED"]),
        ("  INSUFFICIENT accuracy (%)", b["accuracy_by_class_pct"]["INSUFFICIENT"], a["accuracy_by_class_pct"]["INSUFFICIENT"]),
        ("Unsupported-correction rate (%)  [lower is better]", b["unsupported_correction_pct"], a["unsupported_correction_pct"]),
        ("Corrections with invented numbers  [lower is better]", b["corrections_with_invented_numbers"], a["corrections_with_invented_numbers"]),
        ("Ungrounded results saved to memory (%)  [lower is better]", b["ungrounded_saved_to_memory_pct"], a["ungrounded_saved_to_memory_pct"]),
        ("Memory-contamination rate (%)  [lower is better]", b["memory_contamination_pct"], a["memory_contamination_pct"]),
    ]
    print(f"{'Metric':<62}{'Before':>9}{'After':>9}")
    print("-" * 80)
    for name, x, y in lines:
        print(f"{name:<62}{str(x):>9}{str(y):>9}")
    with open("bench_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "before", "after"])
        for name, x, y in lines:
            w.writerow([name.strip(), x, y])
    print("\nSaved: bench_comparison.csv")