"""Measured CPU inference cost, separate from training and commissioning."""
from pathlib import Path
from time import perf_counter_ns
import platform

import numpy as np

from hvac_pid.artifacts import metadata, read_json, write_csv, write_json
from hvac_pid.core import PI, Scenario, evaluate
from hvac_pid.fnn import FNN
from hvac_pid.tuning import identify, simc
from .methods import ONLINE, load_policies


def timings(call, samples):
    for i in range(100):
        call(i)
    elapsed = []
    for i in range(samples):
        start = perf_counter_ns()
        call(i)
        elapsed.append((perf_counter_ns() - start) / 1000)
    return {"median_us": float(np.median(elapsed)), "p95_us": float(np.percentile(elapsed, 95)),
            "samples": samples, "warmup_calls": 100}


def benchmark(output, samples=1000, include_dqn=True):
    if samples < 1:
        raise ValueError("samples must be positive")
    if include_dqn:
        import torch
        torch.set_num_threads(1)
    output = Path(output)
    s = Scenario()
    model = identify(s)[0]
    anchor = simc(model)
    errors = evaluate(s, anchor).trace.temperature - s.setpoint
    pi = PI(anchor)
    rows = [{"method": "PI (all methods)", "component": "control update",
             "interval_minutes": s.dt, "learned_parameters": 2, "artifact_bytes": 0,
             **timings(lambda i: pi.update(float(errors[i % len(errors)]), s.dt), samples)}]
    fnn = FNN.load(output / "05-fnn/model.npz")
    rows.append({"method": "FNN", "component": "one-time commissioning prediction",
                 "interval_minutes": None, "learned_parameters": int(fnn.consequents.size),
                 "artifact_bytes": (output / "05-fnn/model.npz").stat().st_size,
                 **timings(lambda i: fnn.predict(model), samples)})
    policies = load_policies(output, include_dqn)
    # Use actual decision observations from frozen nominal rollouts.
    for method, (name, chapter, _, _, filename) in ONLINE.items():
        if name not in policies:
            continue
        policy = policies[name]
        from hvac_pid.core import TuningEnv
        env = TuningEnv(s, anchor)
        observations = []
        done = False
        while not done:
            obs = env.observation()
            observations.append(obs)
            step = env.step if policy.action_mode == "discrete" else env.step_continuous
            _, _, done = step(policy.choose(obs))
        parameters = int(policy.q.size) if method == "qlearning" else 0
        if method != "qlearning":
            tensors = {id(p): p for value in vars(policy).values() if hasattr(value, "parameters")
                       for p in value.parameters()}
            parameters = sum(p.numel() for p in tensors.values())
        rows.append({"method": name, "component": "frozen gain policy",
                     "interval_minutes": 2.0, "learned_parameters": parameters,
                     "artifact_bytes": (output / chapter / filename).stat().st_size,
                     **timings(lambda i: policy.choose(observations[i % len(observations)]), samples)})
    report = {"environment": {**metadata(), "processor": platform.processor(), "torch_threads": 1 if include_dqn else None},
              "protocol": "CPU, batch=1, 100 warm-up calls, wall time includes observation conversion; no gradients or exploration.",
              "fixed_methods": "ZN/SIMC/BO/FNN/LLM/PG4PI-HVAC all execute only the shared PI online. FNN prediction above is one-time.",
              "rows": rows}
    write_json(output / "benchmark/result.json", report)
    write_csv(output / "benchmark/timings.csv", rows)
    return report
