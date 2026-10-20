"""Product-aware enterprise matching, bounded inference and explicit coverage."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from datetime import date

from .backend import is_chinese_like
from .match_cache import fingerprint
from .protocol import Answer, NoulQuestion, ScoreQuestion
from .retrieval import CompanyIndex

RULES_VERSION = "enterprise-match.v2.1"


def build_match_questions(question_set="core", apply_patch=True):
    questions = {
        "industry_fit": NoulQuestion(
            instructions="Does COMPANY fit PRODUCT's target buyers, use cases and mandatory constraints? "
                         "Distinguish buying needs from selling a similar product. Missing evidence is not a negative fact."),
        "demand_signal": NoulQuestion(
            instructions="Is there evidence of demand for PRODUCT in COMPANY's procurement or news "
                         "within the 12 months ending AS_OF? General buzzwords alone are weak evidence."),
        "budget_scale": ScoreQuestion(
            instructions="Does COMPANY have the purchasing capacity and scale to adopt PRODUCT?",
            criteria=["Limited purchasing capacity", "Possible but uncertain capacity", "Strong purchasing capacity"]),
        "strategic_fit": ScoreQuestion(
            instructions="How well does COMPANY's documented strategy support PRODUCT's value proposition?",
            criteria=["Little supporting evidence", "Some alignment", "Direct strategic alignment"]),
        "urgency": NoulQuestion(
            instructions="As of AS_OF, is there an actionable buying window for PRODUCT at COMPANY? "
                         "Expired procurement is not a current buying window."),
        "match_score": ScoreQuestion(
            instructions="Rate COMPANY as a potential buyer of PRODUCT using target fit, actual needs, "
                         "capacity and timing. Do not equate competitor or supplier similarity with demand.",
            criteria=["Unrelated or unsuitable buyer", "Low match or insufficient evidence",
                      "Plausible buyer with supporting evidence", "Strong buyer fit with concrete demand evidence"]),
    }
    sets = {"core": ("industry_fit", "demand_signal", "match_score"),
            "min": ("match_score",), "all": tuple(questions)}
    if question_set not in sets:
        raise ValueError("questions must be core, all or min")
    names = list(sets[question_set])
    if apply_patch and "industry_fit" not in names:
        names.insert(0, "industry_fit")
    return {name: questions[name] for name in names}


def scoring_state(product, company, as_of):
    # Vectors are retrieval features, not model prompt content.
    return {"product": {k: v for k, v in product.items() if k != "embedding"},
            "as_of": as_of,
            "company": {k: v for k, v in company.items() if k != "embedding"}}


def backend_identity(backend):
    identity = {"class": type(backend).__name__, "model": getattr(backend, "model_id", "unknown")}
    # Include scoring-affecting settings, never secrets. Deployers must bump
    # model_revision when replacing weights behind an unchanged service/model name.
    for key in ("model_revision", "top_logprobs", "chat_template_kwargs", "max_length", "num_ctx"):
        if hasattr(backend, key):
            identity[key] = getattr(backend, key)
    for key in ("zh_backend", "en_backend"):
        child = getattr(backend, key, None)
        if child is not None:
            identity[key] = backend_identity(child)
    return identity


def rank_key(row):
    return (row.get("match_score") if row.get("match_score") is not None else -1,
            row.get("confidence") or 0, -row["input_index"])


def result_row(company, product, answers, questions, apply_patch, idx):
    flat = {key: answer.to_dict() for key, answer in answers.items()}
    ms = flat.get("match_score", {})
    score_raw, conf = ms.get("score"), ms.get("confidence")
    if not isinstance(score_raw, (float, int)) or not 0 <= score_raw <= 3:
        raise ValueError("match_score missing or outside 0..3")
    if conf is not None and (not isinstance(conf, (int, float)) or not 0 <= conf <= 1):
        raise ValueError("confidence must be finite and in 0..1")
    if any(answer.get("raw_mass") == 0 for answer in flat.values()):
        raise ValueError("candidate token evidence is missing or negligible; retry with a compatible scoring backend")
    vals = sorted(ms.get("probabilities", {}).values(), reverse=True)
    if any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in vals):
        raise ValueError("invalid score probabilities")
    margin = vals[0] - vals[1] if len(vals) > 1 else 0.0
    score, note = score_raw, ""
    fit = flat.get("industry_fit", {}).get("noul")
    if apply_patch:
        if fit is None:
            raise ValueError("industry_fit required when apply_patch is enabled")
        if fit < 0.5:
            score = min(score, 1.0)
            note = "不符合该产品的目标客户或必要条件→≤1.0"
        # Non-commercial entities can be legitimate buyers. Only an explicit
        # product constraint excludes them; absent procurement is never a veto.
        basic = company.get("basic", {})
        entity_type = (str(basic.get("scale", "")) + " " + str(basic.get("industry", ""))).lower()
        if product.get("commercial_only") and any(
            s in entity_type for s in ("非企业", "高校", "research", "university", "institute")
        ):
            score = min(score, 0.5)
            note = "该产品仅面向商业企业→≤0.5"
    advice = ("🔥 优先触达" if score >= 2.5 and (conf or 0) >= 0.9 else
              "→ 可跟进" if score >= 2 and (conf or 0) >= 0.7 else
              "◇ 观察待唤醒" if score >= 1.5 else "✋ 暂不触达（记录线索）")
    return {"name": company.get("name", "?"), "company_id": company.get("id"),
            "input_index": idx, "lang": "zh" if is_chinese_like(company, {}) else "en",
            "match_score": round(score, 4), "score_raw": round(score_raw, 4),
            "level": int(round(score)), "score_pct": round(score / 3 * 100),
            "confidence": round(conf, 4) if conf is not None else None,
            "margin": round(margin, 4), "patched": score != score_raw,
            "patch_note": note if score != score_raw else "", "advice": advice,
            "per_question": flat, "score_source": "jev"}


@dataclass(frozen=True)
class MatchOptions:
    mode: str = "full"
    top_k: int = 100
    candidate_limit: int = 500
    recall_limit: int = 5000
    batch_size: int = 32
    max_batch_chars: int = 64000
    question_set: str = "core"
    apply_patch: bool = True
    use_cache: bool = True
    as_of: str = ""
    model_revision: str = "default"

    def validate(self):
        if self.mode not in {"full", "topk"}:
            raise ValueError("mode must be full or topk")
        if not 1 <= self.top_k <= 100000 or not 1 <= self.candidate_limit <= 100000:
            raise ValueError("top_k and candidate_limit must be 1..100000")
        if self.mode == "topk" and self.candidate_limit < self.top_k:
            raise ValueError("candidate_limit must be >= top_k")
        if not 1 <= self.recall_limit <= 200000:
            raise ValueError("recall_limit must be 1..200000")
        if not 1 <= self.batch_size <= 512 or not 1024 <= self.max_batch_chars <= 2000000:
            raise ValueError("invalid batch_size or max_batch_chars")
        if self.as_of:
            date.fromisoformat(self.as_of)


class EnterpriseMatcher:
    def __init__(self, engine, cache=None):
        self.engine = engine
        self.cache = cache

    def match(self, companies, product, options=None, *, index=None, progress=None):
        options = options or MatchOptions()
        options.validate()
        start = time.perf_counter()
        as_of = options.as_of or date.today().isoformat()
        questions = build_match_questions(options.question_set, options.apply_patch)
        retrieval_start = time.perf_counter()
        if options.mode == "topk" and len(companies) > options.candidate_limit:
            index = index or CompanyIndex(companies)
            candidates, retrieval_meta = index.search(product, options.candidate_limit, recall_limit=options.recall_limit)
        else:
            candidates = range(len(companies))
            retrieval_meta = {"method": "all", "exploration_count": 0}
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        namespace = fingerprint({"rules": RULES_VERSION, "product": product,
                                 "questions": {k: q.to_dict() for k, q in questions.items()},
                                 "apply_patch": options.apply_patch, "as_of": as_of,
                                 "backend": backend_identity(self.engine.backend),
                                 "model_revision": options.model_revision})
        cache = self.cache if options.use_cache else None
        ranking = []
        errors = []
        hits = deduplicated = evaluated = failed = batches = 0
        # Keep cache lookups and model state materialization bounded, including full mode.
        chunk_size = 128
        for offset in range(0, len(candidates), chunk_size):
            ids = candidates[offset:offset + chunk_size]
            keyed = []
            states = {}
            for idx in ids:
                state = scoring_state(product, companies[idx], as_of)
                key = fingerprint([namespace, state["company"]])
                keyed.append((idx, key))
                states[key] = state
            cached = cache.get_many(states) if cache is not None else {}
            missing = [key for key in states if key not in cached]
            hits += sum(key in cached for _, key in keyed)
            deduplicated += len(keyed) - len(states)
            evaluated += len(missing)
            fresh = {}
            if missing:
                batches += 1
                answers = self.engine.decide_many([states[key] for key in missing], questions,
                                                 batch_size=options.batch_size,
                                                 max_batch_chars=options.max_batch_chars)
                for key, answer in zip(missing, answers):
                    if isinstance(answer, Exception):
                        fresh[key] = answer
                    else:
                        fresh[key] = {k: v.to_dict() for k, v in answer.items()}
            valid_for_cache = {}
            for idx, key in keyed:
                raw = cached.get(key, fresh.get(key))
                try:
                    if isinstance(raw, Exception):
                        raise raw
                    answers = {k: Answer(v["type"], {a: b for a, b in v.items() if a != "type"})
                               for k, v in raw.items()}
                    row = result_row(companies[idx], product, answers, questions, options.apply_patch, idx)
                    row["cached"] = key in cached
                    ranking.append(row)
                    if key not in cached:
                        valid_for_cache[key] = raw
                except Exception as exc:
                    failed += 1
                    row = {"name": companies[idx].get("name", "?"), "input_index": idx,
                           "company_id": companies[idx].get("id"), "error": str(exc)[:200],
                           "match_score": None, "confidence": None, "score_source": "error"}
                    ranking.append(row)
                    errors.append(row)
            if cache is not None and valid_for_cache:
                cache.put_many(valid_for_cache)
            if progress:
                progress(min(offset + len(ids), len(candidates)), len(candidates))
        if options.mode == "topk":
            ranking = heapq.nlargest(options.top_k, ranking, key=rank_key)
        else:
            ranking.sort(key=rank_key, reverse=True)
        total = len(companies)
        return {"schema": "enterprise-match.v2", "product": product, "questions_used": list(questions),
                "ranking": ranking, "errors": errors, "meta": {
                    "backend": getattr(self.engine.backend, "model_id", "unknown"),
                    "mode": options.mode, "as_of": as_of, "rules_version": RULES_VERSION,
                    "total_companies": total, "candidate_count": len(candidates),
                    "evaluated_companies": evaluated, "failed_companies": failed,
                    "cache_hits": hits, "deduplicated": deduplicated, "inference_chunks": batches,
                    "call_est": evaluated * len(questions),
                    "coverage": len(candidates) / total if total else 1.0,
                    "all_companies_scored": len(candidates) == total and failed == 0,
                    "ranking_scope": "all_companies" if len(candidates) == total else "retrieved_candidates",
                    "retrieval": retrieval_meta, "retrieval_ms": round(retrieval_ms, 2),
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2)}}
