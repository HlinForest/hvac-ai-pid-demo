"""One compatible chat/tool-call connection. No retries or substitute models."""
from dataclasses import asdict
import json
import os
from time import perf_counter
from urllib import request

from .core import Gains
from .tuning import TuningResult, best_gains, in_bounds, initial_samples, run_trial


TOOL = {
    "type": "function", "function": {
        "name": "evaluate_gains",
        "description": "Run a complete cooling simulation with fixed PI gains and return IAE and diagnostics.",
        "parameters": {
            "type": "object", "properties": {
                "kp": {"type": "number"}, "ki": {"type": "number"},
                "reason": {"type": "string"},
            }, "required": ["kp", "ki", "reason"], "additionalProperties": False,
        },
    },
}


def configuration(base_url=None, model=None, key_env="DASHSCOPE_API_KEY"):
    return (base_url or os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model or os.getenv("LLM_MODEL", ""), os.getenv(key_env, ""))


class ChatClient:
    def __init__(self, base_url, model, api_key, timeout=60, reasoning_effort=None):
        self.base_url, self.model, self.api_key, self.timeout = base_url, model, api_key, timeout
        self.reasoning_effort = reasoning_effort

    def __call__(self, messages):
        payload = {"model": self.model, "messages": messages, "tools": [TOOL],
                   "tool_choice": {"type": "function", "function": {"name": "evaluate_gains"}},
                   "temperature": 0}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        req = request.Request(self.base_url.rstrip("/") + "/chat/completions",
                              data=json.dumps(payload).encode(),
                              headers={"Authorization": f"Bearer {self.api_key}",
                                       "Content-Type": "application/json"})
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.load(response)


def tune(scenario, seed=0, rounds=15, *, client=None, model=None, base_url=None,
         key_env="DASHSCOPE_API_KEY", reasoning_effort=None, record=None):
    if rounds < 0:
        raise ValueError("rounds cannot be negative")
    start = perf_counter()
    exchanges = []
    result = TuningResult(None, cost={"model": model, "calls": 0, "proposal_rounds": rounds, "simulations": 0,
                                     "plant_steps": 0, "reasoning_effort": reasoning_effort, "usage": []})

    def persist():
        result.cost["seconds"] = perf_counter() - start
        if record:
            record({**result.to_dict(), "exchanges": exchanges})

    if client is None:
        url, name, key = configuration(base_url, model, key_env)
        result.cost.update({"model": name, "base_url": url})
        if not name or not key:
            result.status = "unavailable"
            result.stop_reason = f"Set LLM_MODEL and {key_env}; no live model was called"
            persist()
            return result
        client = ChatClient(url, name, key, reasoning_effort=reasoning_effort)
    result.status, result.stop_reason = "running", "experiment in progress"
    identified, gains = initial_samples(scenario, seed)
    result.trials = [run_trial(scenario, g, i, source) for i, (g, source) in
                     enumerate(zip(gains, ["ZN", "SIMC", "random", "random", "random"]))]
    result.best = best_gains(result.trials)
    result.cost.update({"simulations": 5, "plant_steps": 5 * scenario.steps,
                        "identification_steps": round(240 / scenario.dt)})
    messages = [
        {"role": "system", "content":
         "You tune cooling PI gains by experiments. Positive error means too hot. "
         "Call evaluate_gains exactly once per turn. Minimize IAE (degree-minutes). "
         "Kp units are command/degree, Ki units are command/(degree*minute). "
         "Kp must be in [0.01,3], Ki in [0.0001,0.5]. Explain the proposed change briefly. "
         "There is no acceptance gate; worse trials remain visible. Propose any gains in bounds."},
        {"role": "user", "content": json.dumps({
            "identified_model": asdict(identified),
            "experiment": {k: v for k, v in asdict(scenario).items() if k not in {"gain", "tau", "delay"}},
            "initial_trials": result.trials, "proposal_budget": rounds,
        }, ensure_ascii=False)},
    ]
    persist()
    for round_index in range(rounds):
        call_start = perf_counter()
        sent_messages = json.loads(json.dumps(messages))
        result.cost["calls"] += 1
        try:
            body = client(sent_messages)
            message = body["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
                raise ValueError("Malformed tool_calls response")
        except Exception as exc:
            # This single catch is the experiment's persistence boundary.
            result.status = "incomplete"
            result.stop_reason = f"Model call failed: {type(exc).__name__}: {exc}"
            exchanges.append({"round": round_index + 1, "request": sent_messages,
                              "error": result.stop_reason, "seconds": perf_counter() - call_start})
            persist()
            return result
        exchanges.append({"round": round_index + 1, "request": sent_messages,
                          "response": body, "seconds": perf_counter() - call_start})
        result.cost["usage"].append(body.get("usage", {}))
        try:
            if len(calls) != 1 or calls[0].get("function", {}).get("name") != "evaluate_gains":
                raise ValueError("Return exactly one evaluate_gains tool call")
            arguments = json.loads(calls[0]["function"]["arguments"])
            if set(arguments) != {"kp", "ki", "reason"}:
                raise ValueError("Expected kp, ki, reason")
            if not isinstance(arguments["reason"], str) or not arguments["reason"].strip():
                raise ValueError("reason must be a non-empty string")
            if any(type(arguments[k]) not in (int, float) for k in ("kp", "ki")):
                raise ValueError("kp and ki must be numbers")
            candidate = Gains(arguments["kp"], arguments["ki"])
            if not in_bounds(candidate):
                raise ValueError("Proposal outside Kp=[0.01,3], Ki=[0.0001,0.5]; no clipping applied")
            trial = run_trial(scenario, candidate, len(result.trials), "LLM",
                              round=round_index + 1, reason=arguments["reason"])
            result.cost["simulations"] += 1
            result.cost["plant_steps"] += scenario.steps
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            trial = {"trial": len(result.trials), "round": round_index + 1,
                     "source": "LLM", "status": "invalid", "error": str(exc)}
        result.trials.append(trial)
        result.best = best_gains(result.trials)
        feedback = json.dumps({"result": trial, "remaining_rounds": rounds - round_index - 1})
        if calls and all(call.get("id") for call in calls):
            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
            for call in calls:
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": feedback})
        else:
            messages.append({"role": "user", "content": feedback})
        persist()
    initial_best = min(t["iae"] for t in result.trials[:5])
    final_best = min(t["iae"] for t in result.trials if t["status"] == "evaluated")
    result.status = "complete"
    result.stop_reason = "budget exhausted; improved initial samples" if final_best < initial_best else "budget exhausted; no improvement over initial samples"
    persist()
    return result
