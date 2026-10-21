"""Correctness, cache invalidation, coverage and recall regression tests (offline)."""
from dataclasses import replace
from copy import deepcopy

import pytest

from jevp.engine import DecisionEngine
from jevp.match_cache import ScoreCache
from jevp.matching import EnterpriseMatcher, MatchOptions, build_match_questions, scoring_state
from jevp.protocol import Answer
from jevp.retrieval import CompanyIndex, IndexRegistry


PRODUCT = {"name": "Asset health", "description": "vibration monitoring for steel manufacturing",
           "target_industries": ["steel"]}
COMPANIES = [
    {"id": "a", "name": "Steel buyer", "basic": {"industry": "steel"},
     "bidding": [{"title": "vibration monitoring procurement"}]},
    {"id": "b", "name": "Retail buyer", "basic": {"industry": "retail"}, "bidding": []},
    {"id": "c", "name": "Steel research", "basic": {"industry": "steel research institute"}},
]


class FixtureBackend:
    direct = True
    model_id = "offline-fixture-v1"

    def __init__(self):
        self.calls = []
        self.fail_ids = set()

    def direct_decide(self, state, questions):
        self.calls.append(deepcopy(state))
        company, product = state["company"], state["product"]
        if company.get("id") in self.fail_ids:
            raise RuntimeError("injected model failure")
        fit = any(ind in company.get("basic", {}).get("industry", "")
                  for ind in product.get("target_industries", ["steel"]))
        return {name: Answer("noul", {"noul": 0.95 if fit else 0.05}) if q.type == "noul"
                else Answer("score", {"score": 2.8 if fit else 0.4, "confidence": 0.95,
                                       "probabilities": {"0": 0.01, "1": 0.01, "2": 0.16, "3": 0.82}})
                for name, q in questions.items()}


@pytest.fixture
def matcher():
    backend = FixtureBackend()
    cache = ScoreCache()
    instance = EnterpriseMatcher(DecisionEngine(backend, None), cache)
    yield instance
    cache.close()


def test_product_affects_actual_model_input_and_ranking(matcher):
    first = matcher.match(COMPANIES, PRODUCT)
    second = matcher.match(COMPANIES, {"name": "Retail POS", "target_industries": ["retail"]})
    assert first["ranking"][0]["company_id"] == "a"
    assert second["ranking"][0]["company_id"] == "b"
    assert matcher.engine.backend.calls[-1]["product"]["name"] == "Retail POS"
    assert second["meta"]["cache_hits"] == 0


def test_full_coverage_and_stable_ties_with_duplicate_names(matcher):
    companies = deepcopy(COMPANIES)
    companies[0]["name"] = companies[2]["name"] = "Same display name"
    out = matcher.match(companies, PRODUCT)
    assert [r["input_index"] for r in out["ranking"]] == [0, 2, 1]
    assert out["meta"]["all_companies_scored"]
    assert out["meta"]["coverage"] == 1


def test_min_keeps_industry_correction_and_noncommercial_is_product_specific(matcher):
    assert set(build_match_questions("min", True)) == {"industry_fit", "match_score"}
    assert set(build_match_questions("min", False)) == {"match_score"}
    out = matcher.match([COMPANIES[2]], PRODUCT, MatchOptions(question_set="min"))
    assert out["ranking"][0]["match_score"] == 2.8
    out = matcher.match([COMPANIES[2]], {**PRODUCT, "commercial_only": True})
    assert out["ranking"][0]["match_score"] == 0.5


def test_cache_hit_and_incremental_profile_update(matcher):
    matcher.match(COMPANIES, PRODUCT)
    matcher.engine.backend.calls.clear()
    hot = matcher.match(COMPANIES, PRODUCT)
    assert hot["meta"]["cache_hits"] == 3
    assert hot["meta"]["call_est"] == 0
    assert matcher.engine.backend.calls == []
    changed = deepcopy(COMPANIES)
    changed[1]["website"] = {"news": "New plant procurement"}
    delta = matcher.match(changed, PRODUCT)
    assert delta["meta"]["cache_hits"] == 2
    assert len(matcher.engine.backend.calls) == 1


def test_backend_configuration_is_part_of_cache_identity(matcher):
    matcher.engine.backend.top_logprobs = 20
    matcher.match(COMPANIES, PRODUCT)
    matcher.engine.backend.top_logprobs = 40
    assert matcher.match(COMPANIES, PRODUCT)["meta"]["cache_hits"] == 0


def test_invalid_or_missing_candidate_evidence_is_not_cached(matcher):
    original = matcher.engine.backend.direct_decide

    def invalid(state, questions):
        answers = original(state, questions)
        answers["match_score"].payload["raw_mass"] = 0
        return answers

    matcher.engine.backend.direct_decide = invalid
    out = matcher.match(COMPANIES, PRODUCT)
    assert out["meta"]["failed_companies"] == 3
    matcher.engine.backend.direct_decide = original
    assert matcher.match(COMPANIES, PRODUCT)["meta"]["cache_hits"] == 0


@pytest.mark.parametrize("change", [dict(as_of="2026-10-19"), dict(model_revision="new-weights"),
                                    dict(question_set="all"), dict(apply_patch=False)])
def test_cache_invalidates_on_scoring_contract(matcher, change):
    baseline = MatchOptions(as_of="2026-10-20")
    matcher.match(COMPANIES, PRODUCT, baseline)
    out = matcher.match(COMPANIES, PRODUCT, replace(baseline, **change))
    assert out["meta"]["cache_hits"] == 0


def test_failures_are_isolated_and_not_cached(matcher):
    matcher.engine.backend.fail_ids = {"b"}
    out = matcher.match(COMPANIES, PRODUCT)
    assert len(out["ranking"]) == 3
    assert out["meta"]["failed_companies"] == 1
    assert not out["meta"]["all_companies_scored"]
    assert out["errors"][0]["company_id"] == "b"
    matcher.engine.backend.fail_ids.clear()
    out = matcher.match(COMPANIES, PRODUCT)
    assert out["meta"]["cache_hits"] == 2
    assert out["meta"]["failed_companies"] == 0


def test_topk_reports_limited_scope_and_reuses_index(matcher):
    companies = deepcopy(COMPANIES) + [
        {"id": f"retail-{i}", "name": "Retail", "basic": {"industry": "retail"}}
        for i in range(1000)]
    index = CompanyIndex(companies)
    out = matcher.match(index.companies, PRODUCT, MatchOptions(mode="topk", top_k=2, candidate_limit=20), index=index)
    assert len(out["ranking"]) == 2
    assert out["meta"]["candidate_count"] == 20
    assert out["meta"]["evaluated_companies"] == 20
    assert not out["meta"]["all_companies_scored"]
    assert out["meta"]["ranking_scope"] == "retrieved_candidates"
    assert out["ranking"][0]["company_id"] == "a"


def test_cache_ttl_lru_and_reopen(tmp_path):
    now = [10]
    path = tmp_path / "scores.sqlite3"
    cache = ScoreCache(path, ttl_seconds=5, max_entries=2, clock=lambda: now[0])
    cache.put_many({"a": {"value": 1}, "b": {"value": 2}})
    now[0] += 1
    cache.get_many(["a"])
    cache.put_many({"c": {"value": 3}})
    assert set(cache.get_many(["a", "b", "c"])) == {"a", "c"}
    cache.close()
    cache = ScoreCache(path, ttl_seconds=5, clock=lambda: now[0])
    assert cache.get_many(["c"])["c"]["value"] == 3
    now[0] += 6
    assert cache.get_many(["a", "c"]) == {}
    cache.close()


def test_dense_retrieval_and_input_validation():
    index = CompanyIndex([{"name": "far", "embedding": [0, 1]},
                          {"name": "near", "embedding": [1, 0]},
                          {"name": "opposite", "embedding": [-1, 0]}])
    selected, _ = index.search({"embedding": [1, 0]}, 1, exploration=0)
    assert selected == [1]
    # Also verify the dependency-free vector path.
    if hasattr(index.vectors, "tolist"):
        index.vectors = index.vectors.tolist()
    assert index.search({"embedding": [1, 0]}, 1, exploration=0)[0] == [1]
    with pytest.raises(ValueError):
        index.search({"embedding": [1, 0, 0]}, 1)
    with pytest.raises(ValueError):
        CompanyIndex([{"embedding": [float("nan"), 1]}])
    assert "embedding" not in scoring_state({"embedding": [1], "name": "x"}, {"embedding": [1]}, "2026-10-20")["product"]


def test_chinese_recall_missing_evidence_and_snapshot():
    companies = [{"name": "设备企业", "basic": {"industry": "钢铁制造"},
                  "bidding": [{"title": "设备预测性维护采购"}]}, {"name": "未知企业"}, {"name": "商店"}]
    index = CompanyIndex(companies)
    companies[0]["basic"]["industry"] = "edited externally"
    ids, _ = index.search({"description": "设备预测性维护"}, 1, exploration=0)
    assert ids == [0]
    assert index.companies[0]["basic"]["industry"] == "钢铁制造"
    ids, _ = index.search({"description": "completely missing terms"}, 2)
    assert len(set(ids)) == 2  # unknown profiles remain eligible through exploration


def test_registry_eviction_and_expiry():
    registry = IndexRegistry(max_indexes=1, max_documents=10, ttl_seconds=7200)
    old, _ = registry.add(COMPANIES)
    new, _ = registry.add(COMPANIES)
    with pytest.raises(KeyError):
        registry.get(old)
    assert len(registry.get(new).companies) == 3
    registry.ttl = -1
    with pytest.raises(KeyError):
        registry.get(new)


@pytest.mark.parametrize("options", [MatchOptions(mode="invalid"), MatchOptions(mode="topk", top_k=10, candidate_limit=2),
                                     MatchOptions(batch_size=0), MatchOptions(as_of="not-a-date")])
def test_invalid_options_fail_before_inference(matcher, options):
    with pytest.raises(ValueError):
        matcher.match(COMPANIES, PRODUCT, options)
    assert matcher.engine.backend.calls == []
