"""Reproducible matching scale benchmark; OFFLINE SYNTHETIC fixtures by default.

No model accuracy or GPU throughput claims can be inferred from this benchmark.
Run: python benchmarks/benchmark_matching.py --sizes 100,1000,10000,100000
The loopback HTTP test suite separately exercises the actual HTTP transport.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevp.engine import DecisionEngine
from jevp.match_cache import ScoreCache
from jevp.matching import EnterpriseMatcher, MatchOptions, build_match_questions, scoring_state
from jevp.retrieval import CompanyIndex

PRODUCT = {"name": "Vibration maintenance", "description": "vibration predictive maintenance for steel plants",
           "target_industries": ["steel"]}


def dataset(n):
    companies, labels = [], []
    for i in range(n):
        segment = i % 100
        industry = "steel" if segment < 20 else ("retail" if segment < 60 else "software")
        demand = segment < 5
        supplier = 5 <= segment < 8
        companies.append({"id": str(i), "name": f"Company {i}",
                          "basic": {"industry": industry, "scale": "medium", "location": "test region"},
                          "bidding": [{"date": "2026-10-01", "title": "vibration predictive maintenance procurement"}]
                                     if demand else [],
                          "website": {"strategy": "sells vibration predictive maintenance software" if supplier else "operations"}})
        labels.append(3 if demand else 2 if industry == "steel" and not supplier else 1 if supplier else 0)
    return companies, labels


class SyntheticBackend:
    """Deterministic logprob fixture. No neural network or remote API involved."""
    candidate_scoring = "token_string"
    model_id = "OFFLINE-SYNTHETIC-NOT-A-MODEL"

    def __init__(self, delay=0):
        self.prompts = 0
        self.batches = 0
        self.delay = delay

    def batch_last_logits(self, prompts):
        self.prompts += len(prompts)
        self.batches += 1
        if self.delay:
            time.sleep(self.delay)
        rows = []
        for prompt in prompts:
            state = json.loads(prompt.split("STATE:\n", 1)[1].split("\n\n", 1)[0])
            company = state["company"]
            steel = company["basic"]["industry"] == "steel"
            level = 3 if company.get("bidding") else 1 if "sells" in company["website"]["strategy"] else 2 if steel else 0
            row = {str(i): -0.01 if i == level else -8 for i in range(4)}
            row.update(Yes=-0.01 if steel else -8, No=-8 if steel else -0.01)
            rows.append(row)
        return rows


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)]


def ndcg(ids, labels, k):
    gain = lambda values: sum((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(values))
    ideal = gain(sorted(labels, reverse=True)[:k])
    return gain([labels[i] for i in ids[:k]]) / ideal if ideal else 1.0


def run_size(n, repeat):
    companies, labels = dataset(n)
    backend = SyntheticBackend()
    engine = DecisionEngine(backend, None)
    options = MatchOptions(as_of="2026-10-20", use_cache=False)
    with tempfile.TemporaryDirectory() as directory:
        cache = ScoreCache(Path(directory) / "scores.sqlite3", max_entries=max(n + 1, 1000))
        matcher = EnterpriseMatcher(engine, cache)
        start = time.perf_counter()
        index = CompanyIndex(companies)
        index_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        full = matcher.match(companies, PRODUCT, options)
        full_ms = (time.perf_counter() - start) * 1000
        full_ids = [r["input_index"] for r in full["ranking"]]
        full_meta = full["meta"]
        accuracy = sum(r["level"] == labels[r["input_index"]] for r in full["ranking"]) / n
        # Compare batched scoring against per-company calls on a bounded sample.
        questions = build_match_questions()
        serial = [engine.decide(scoring_state(PRODUCT, company, options.as_of), questions)
                  for company in companies[:100]]
        by_id = {r["input_index"]: r for r in full["ranking"] if r["input_index"] < 100}
        identical = all(by_id[i]["per_question"] == {k: v.to_dict() for k, v in answer.items()}
                        for i, answer in enumerate(serial))
        del full
        top_opts = replace(options, mode="topk", top_k=min(100, n), candidate_limit=min(500, n))
        top_times = []
        for _ in range(repeat):
            start = time.perf_counter()
            top = matcher.match(index.companies, PRODUCT, top_opts, index=index)
            top_times.append((time.perf_counter() - start) * 1000)
        top_ids = [r["input_index"] for r in top["ranking"]]
        k = min(100, n)
        cached_opts = replace(options, use_cache=True)
        cache_fill_meta = matcher.match(companies, PRODUCT, cached_opts)["meta"]
        start = time.perf_counter()
        hot = matcher.match(companies, PRODUCT, cached_opts)
        hot_ms = (time.perf_counter() - start) * 1000
        hot_meta = hot["meta"]
        del hot
        # One changed profile must invalidate exactly that company.
        companies[-1] = {**companies[-1], "version": "updated"}
        delta_meta = matcher.match(companies, PRODUCT, cached_opts)["meta"]
        overlap = len(set(top_ids) & set(full_ids[:k])) / k
        # Quality gate for this controlled fixture only, not production data.
        assert identical and hot_meta["call_est"] == 0
        assert delta_meta["evaluated_companies"] == 1
        assert overlap >= 0.95 and ndcg(top_ids, labels, k) >= 0.95
        cache.close()
        return {"companies": n, "index_build_ms": round(index_ms, 2),
                "full_ms": round(full_ms, 2), "full_companies_per_second": round(n / (full_ms / 1000), 1),
                "full_question_evaluations": full_meta["call_est"],
                "topk_p50_ms": round(percentile(top_times, 0.5), 2),
                "topk_p95_ms": round(percentile(top_times, 0.95), 2), "topk_repeats": repeat,
                "topk_question_evaluations": top["meta"]["call_est"],
                "topk_candidate_count": top["meta"]["candidate_count"],
                "topk_overlap_with_full": round(overlap, 4),
                "topk_ndcg": round(ndcg(top_ids, labels, k), 4),
                "synthetic_level_accuracy": accuracy, "batch_equals_serial": identical,
                "cache_fill_question_evaluations": cache_fill_meta["call_est"],
                "hot_full_ms": round(hot_ms, 2), "hot_question_evaluations": hot_meta["call_est"],
                "incremental_changed_companies": delta_meta["evaluated_companies"],
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                     (1024 ** 2 if sys.platform == "darwin" else 1024), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="100,1000,10000,100000")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", default="results/matching_scale.json")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    report = {"kind": "OFFLINE_SYNTHETIC", "python": platform.python_version(),
              "machine": platform.machine(), "platform": platform.system(),
              "note": "Fixture CPU pipeline only; no real model/GPU throughput or production recall claims.",
              "results": []}
    for n in map(int, args.sizes.split(",")):
        if n < 1:
            parser.error("sizes must be positive")
        result = run_size(n, args.repeat)
        report["results"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
