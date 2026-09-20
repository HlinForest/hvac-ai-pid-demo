import json

from hvac_pid.core import Scenario
from hvac_pid.llm import tune
from hvac_pid.tuning import tune as tune_bo


def response(kp, ki):
    # Synthetic transport fixture, never used by demonstration commands.
    return {"choices": [{"message": {"content": None, "tool_calls": [{
        "id": "test-call", "type": "function", "function": {"name": "evaluate_gains",
        "arguments": json.dumps({"kp": kp, "ki": ki, "reason": "synthetic test"})}}]}}],
        "usage": {"total_tokens": 10}}


def test_missing_credentials_produce_unavailable_not_baseline(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    saved = []
    result = tune(Scenario(), record=saved.append)
    assert result.status == "unavailable"
    assert result.best is None
    assert result.trials == []
    assert saved[-1]["cost"]["calls"] == 0


def test_invalid_candidate_consumes_round_without_clipping_then_can_recover():
    replies = iter([response(99, 0.01), response(0.3, 0.02)])
    sent, saved = [], []
    def client(messages):
        sent.append(messages)
        return next(replies)
    result = tune(Scenario(duration=4), seed=2, rounds=2, client=client, record=saved.append)
    assert len(result.trials) == 7
    assert result.trials[5]["status"] == "invalid"
    assert "kp" not in result.trials[5]
    assert result.cost["simulations"] == 6
    assert result.trials[6]["kp"] == 0.3
    assert sent[1][-1]["role"] == "tool"
    assert "outside" in sent[1][-1]["content"]
    assert saved[-1]["status"] == "complete"


def test_provider_failure_preserves_trials_and_reports_incomplete():
    calls = 0
    def client(messages):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("synthetic outage")
        return response(0.3, 0.02)
    saved = []
    result = tune(Scenario(duration=4), rounds=3, client=client, record=saved.append)
    assert result.status == "incomplete"
    assert result.cost["calls"] == 2
    assert len(result.trials) == 6
    assert saved[-1]["exchanges"][-1]["error"].startswith("Model call failed")


def test_bo_llm_use_identical_initial_evaluations():
    s = Scenario(duration=4)
    bo = tune_bo(s, seed=7, rounds=0)
    llm = tune(s, seed=7, rounds=0, client=lambda _: None)
    assert [(t["kp"], t["ki"], t["iae"]) for t in bo.trials] == [(t["kp"], t["ki"], t["iae"]) for t in llm.trials]


def test_in_progress_checkpoint_is_not_marked_complete():
    saved = []
    tune(Scenario(duration=4), rounds=1, client=lambda _: response(0.2, 0.01), record=saved.append)
    assert saved[0]["status"] == "running"
    assert saved[-1]["status"] == "complete"


def test_malformed_response_is_persisted_as_incomplete():
    saved = []
    result = tune(Scenario(duration=4), rounds=1, client=lambda _: {"choices": []}, record=saved.append)
    assert result.status == "incomplete"
    assert saved[-1]["status"] == "incomplete"
    assert result.cost["simulations"] == 5


def test_explicit_reasoning_option_reaches_transport_without_recording_key(monkeypatch):
    from hvac_pid.llm import ChatClient
    sent = []
    class Reply:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps(response(0.3, 0.02)).encode()
    def send(req, timeout):
        sent.append(json.loads(req.data))
        assert req.get_header('Authorization') == 'Bearer synthetic-test-key'
        return Reply()
    monkeypatch.setattr('hvac_pid.llm.request.urlopen', send)
    ChatClient('https://example.invalid', 'test-model', 'synthetic-test-key', reasoning_effort='none')([])
    assert sent[0]['reasoning_effort'] == 'none'
    assert 'synthetic-test-key' not in json.dumps(sent)
