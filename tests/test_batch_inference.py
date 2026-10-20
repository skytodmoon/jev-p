"""Real loopback HTTP tests: ordering, global concurrency, failures and batching."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from jevp.backend import OpenAICompatibleBackend, RouterBackend, _EstimateTokenizer
from jevp.engine import DecisionEngine
from jevp.protocol import NoulQuestion, ScoreQuestion, build_prompt


@pytest.fixture
def http_backend(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    state = {"active": 0, "peak": 0, "calls": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["calls"] += 1
            try:
                time.sleep(0.005)  # deterministic network/service work, not a performance claim
                prompt = body["messages"][0]["content"]
                if "FAIL_THIS_STATE" in prompt:
                    self.send_response(503)
                    self.end_headers()
                    return
                high = "preferred" in prompt
                row = {"Yes": -0.1, "No": -3, "0": -4, "1": -3,
                       "2": -2 if high else -0.1, "3": -0.1 if high else -2}
                payload = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
                    {"token": token, "logprob": lp} for token, lp in row.items()]}]}}]}
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            finally:
                with lock:
                    state["active"] -= 1

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with patch.object(OpenAICompatibleBackend, "_make_tokenizer", return_value=_EstimateTokenizer()):
        backend = OpenAICompatibleBackend("local-test-fixture", f"http://127.0.0.1:{server.server_port}",
                                          max_concurrency=3, timeout=2)
    yield backend, state
    server.shutdown()
    server.server_close()
    thread.join()


QUESTIONS = {"fit": NoulQuestion(instructions="Is this a buyer?"),
             "score": ScoreQuestion(instructions="Rate the match", criteria=["none", "low", "medium", "high"])}


def test_batched_matches_serial_in_input_order(http_backend):
    backend, state = http_backend
    engine = DecisionEngine(backend, None)
    states = ["preferred company", "ordinary company", "preferred company two"]
    serial = [{k: v.to_dict() for k, v in engine.decide(s, QUESTIONS).items()} for s in states]
    batched = engine.decide_many(states, QUESTIONS, batch_size=4)
    assert [{k: v.to_dict() for k, v in a.items()} for a in batched] == serial
    assert 1 < state["peak"] <= 3


def test_global_concurrency_bound_across_simultaneous_calls(http_backend):
    backend, state = http_backend
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda _: backend.batch_last_logits(["preferred"] * 9), range(2)))
    assert all(len(r) == 9 for r in rows)
    assert state["peak"] == 3
    assert state["calls"] == 18


def test_one_http_failure_does_not_poison_other_states(http_backend):
    backend, _ = http_backend
    out = DecisionEngine(backend, None).decide_many(
        ["preferred", "FAIL_THIS_STATE", "ordinary"], QUESTIONS, batch_size=6)
    assert isinstance(out[1], Exception)
    assert set(out[0]) == set(out[2]) == set(QUESTIONS)


def test_prompt_count_and_character_budget_are_bounded():
    class Tracking:
        candidate_scoring = "token_string"

        def __init__(self):
            self.batches = []

        def batch_last_logits(self, prompts):
            self.batches.append(prompts)
            return [{"Yes": -0.1, "No": -2, "0": -4, "1": -3, "2": -2, "3": -0.1} for _ in prompts]

    backend = Tracking()
    engine = DecisionEngine(backend, None)
    budget = max(len(build_prompt("x", n, q)) for n, q in QUESTIONS.items()) * 2
    out = engine.decide_many(["x"] * 40, QUESTIONS, batch_size=3, max_batch_chars=budget)
    assert all(not isinstance(a, Exception) for a in out)
    assert all(len(batch) <= 3 and sum(map(len, batch)) <= budget for batch in backend.batches)
    out = engine.decide_many(["x" * 1000, "x"], QUESTIONS, max_batch_chars=budget)
    assert isinstance(out[0], ValueError)
    assert not isinstance(out[1], Exception)


def test_bad_batch_shape_fails_explicitly():
    class WrongShape:
        def batch_last_logits(self, prompts):
            return []
    out = DecisionEngine(WrongShape(), None).decide_many(["a", "b"], QUESTIONS)
    assert all(isinstance(a, RuntimeError) for a in out)


def test_local_context_overflow_is_rejected_per_state():
    class Tokenizer:
        def __call__(self, value, **kwargs):
            return {"input_ids": list(range(len(value)))}

    class LocalBackend:
        max_length = 400

        def batch_last_logits(self, prompts):
            return [[0.0] * 10 for _ in prompts]

    out = DecisionEngine(LocalBackend(), Tokenizer()).decide_many(
        ["x" * 1000, "short"], {"fit": QUESTIONS["fit"]})
    assert isinstance(out[0], ValueError)
    assert set(out[1]) == {"fit"}


def test_router_keeps_batch_order_for_both_languages(http_backend):
    backend, _ = http_backend
    engine = DecisionEngine(RouterBackend(zh_backend=backend, en_backend=backend), None)
    out = engine.decide_many(["preferred", "优先购买产品的公司" * 20, "ordinary"], QUESTIONS)
    assert all(set(a) == set(QUESTIONS) for a in out)
    assert out[0]["score"].payload["score"] > out[2]["score"].payload["score"]
