"""Reusable local BM25 + optional supplied-vector candidate retrieval.

No embedding service is called. Missing evidence is never a hard exclusion.
Retrieval ranks candidates, not calibrated purchase probabilities or JEV scores.
"""
from __future__ import annotations

import heapq
import math
import random
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy

from .match_cache import fingerprint

_WORDS = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")
_STOP = {"the", "and", "for", "with", "that", "this", "from", "our", "of", "to", "in", "a", "an", "is"}


def text(value):
    if isinstance(value, dict):
        return " ".join(text(v) for k, v in value.items() if k not in {"embedding", "id", "name", "version"})
    if isinstance(value, (list, tuple)):
        return " ".join(text(v) for v in value)
    return "" if value is None else str(value)


def terms(value):
    out = []
    for token in _WORDS.findall(text(value).lower()[:12000]):
        if "\u3400" <= token[0] <= "\u9fff":
            out.extend(token[i:i + 2] for i in range(len(token) - 1))
            if len(token) == 1:
                out.append(token)
        elif token not in _STOP:
            out.append(token)
    return out


class CompanyIndex:
    """Immutable company snapshot. Build once, reuse across products and queries."""

    def __init__(self, companies):
        self.companies = deepcopy(companies)
        self.postings = {"profile": defaultdict(list), "industry": defaultdict(list),
                         "procurement": defaultdict(list), "news": defaultdict(list)}
        self.lengths = {field: [] for field in self.postings}
        self.document_hashes = []
        self.vector_ids = []
        vectors = []
        dimension = None
        for idx, company in enumerate(self.companies):
            self.document_hashes.append(fingerprint(company))
            sections = {"profile": company, "industry": company.get("basic", {}).get("industry", ""),
                        "procurement": company.get("bidding", []),
                        "news": company.get("website", {}).get("news", "")}
            for field, section in sections.items():
                counts = Counter(terms(section))
                self.lengths[field].append(sum(counts.values()))
                for term, count in counts.items():
                    self.postings[field][term].append((idx, count))
            vector = company.get("embedding")
            if vector is not None:
                if not isinstance(vector, list) or not vector:
                    raise ValueError("company embedding must be a nonempty numeric list")
                if dimension is None:
                    dimension = len(vector)
                if len(vector) != dimension or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                    raise ValueError("company embeddings must be finite and have equal dimensions")
                norm = math.hypot(*vector)
                if not norm:
                    raise ValueError("company embedding cannot be a zero vector")
                self.vector_ids.append(idx)
                vectors.append([x / norm for x in vector])
        self.digest = fingerprint(self.document_hashes)
        self.dimension = dimension
        self.vectors = vectors
        if vectors:
            try:
                import numpy as np
            except ImportError:
                pass
            else:
                self.vectors = np.asarray(vectors, dtype="float32")
        n = max(1, len(companies))
        self.avg_lengths = {field: max(1, sum(sizes) / n) for field, sizes in self.lengths.items()}

    def _bm25(self, query, field, limit):
        scores = defaultdict(float)
        n = len(self.companies)
        for token in set(terms(query)):
            posting = self.postings[field].get(token, ())
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for idx, tf in posting:
                length = self.lengths[field][idx] / self.avg_lengths[field]
                scores[idx] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length))
        return heapq.nlargest(limit, scores, key=lambda i: (scores[i], -i))

    def _dense(self, vector, limit):
        if self.dimension is None:
            return []
        if not isinstance(vector, list) or len(vector) != self.dimension or not all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in vector
        ):
            raise ValueError("product embedding must match company embedding dimensions")
        norm = math.hypot(*vector)
        if not norm:
            raise ValueError("product embedding cannot be a zero vector")
        unit = [x / norm for x in vector]
        if hasattr(self.vectors, "shape"):
            scores = self.vectors @ unit
        else:
            scores = [sum(a * b for a, b in zip(row, unit)) for row in self.vectors]
        selected = heapq.nlargest(limit, range(len(scores)), key=lambda i: (float(scores[i]), -i))
        return [self.vector_ids[i] for i in selected]

    def search(self, product, limit, *, recall_limit=5000, exploration=0.1):
        if limit < 1 or not 0 <= exploration <= 0.5:
            raise ValueError("candidate limit must be positive; exploration must be 0..0.5")
        n = len(self.companies)
        if limit >= n:
            return list(range(n)), {"method": "all", "exploration_count": 0}
        recall_limit = max(limit, min(recall_limit, n))
        query = [product.get("name", ""), {k: v for k, v in product.items() if k != "embedding"}]
        # Procurement is a stronger buying signal than a strategy/seller page.
        # Weights are transparent heuristics, not learned/calibrated probabilities.
        channels = [(1.0, self._bm25(query, "profile", recall_limit)),
                    (2.0, self._bm25(query, "procurement", recall_limit)),
                    (1.0, self._bm25(query, "news", recall_limit))]
        if product.get("target_industries"):
            channels.append((1.0, self._bm25(product["target_industries"], "industry", recall_limit)))
        if product.get("embedding") is not None:
            channels.append((1.0, self._dense(product["embedding"], recall_limit)))
        fused = defaultdict(float)
        for weight, channel in channels:
            for rank, idx in enumerate(channel, 1):
                fused[idx] += weight / (60 + rank)
        recall_pool = heapq.nlargest(recall_limit, fused, key=lambda i: (fused[i], -i))
        reserve = min(limit - 1, max(1, int(limit * exploration))) if exploration else 0
        selected = recall_pool[:limit - reserve]
        selected_set = set(selected)
        # Stable exploration protects some missing-evidence cases, without claiming recall guarantees.
        rng = random.Random(fingerprint([self.digest, product]))
        explored = 0
        while len(selected) < limit:
            idx = rng.randrange(n)
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
                explored += 1
        return selected, {"method": "bm25+weighted_rrf" + ("+supplied_cosine" if
                         product.get("embedding") is not None and self.dimension is not None else ""),
                         "recall_pool_count": len(recall_pool), "recall_limit": recall_limit,
                         "exploration_count": explored, "approximate_selection": True}


class IndexRegistry:
    """Bounded process-local prepared indexes. Eviction/expiry invalidates the handle."""

    def __init__(self, max_indexes=2, max_documents=200000, ttl_seconds=7200):
        self.max_indexes = max_indexes
        self.max_documents = max_documents
        self.ttl = ttl_seconds
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def add(self, companies):
        if not companies or len(companies) > self.max_documents:
            raise ValueError(f"index requires 1..{self.max_documents} companies")
        index = CompanyIndex(companies)
        handle = uuid.uuid4().hex
        with self.lock:
            now = time.monotonic()
            self.entries = OrderedDict((k, v) for k, v in self.entries.items() if now - v[0] < self.ttl)
            while self.entries and (len(self.entries) >= self.max_indexes or
                  sum(len(v[1].companies) for v in self.entries.values()) + len(companies) > self.max_documents):
                self.entries.popitem(last=False)
            self.entries[handle] = (now, index)
        return handle, index

    def get(self, handle):
        with self.lock:
            entry = self.entries.get(handle)
            if entry is None or time.monotonic() - entry[0] >= self.ttl:
                self.entries.pop(handle, None)
                raise KeyError("index not found or expired; rebuild it on this worker")
            self.entries.move_to_end(handle)
            return entry[1]
