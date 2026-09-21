"""Single calculations from the same functions used by the experiments."""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hvac_pid.artifacts import read_json, write_json
from hvac_pid.core import Gains, Scenario, evaluate
from hvac_pid.fnn import FNN, features
from hvac_pid.tuning import (identify, simc, zn, initial_samples, encode, decode,
                             fit_gp, expected_improvement)
from .methods import ONLINE


def explain(output, method="all", seed=0):
    output = Path(output)
    methods = ["classical", "bo", "fnn", *ONLINE, "pg4pi", "llm"] if method == "all" else [method]
    for name in methods:
        if name == "classical":
            model = identify(Scenario())[0]
            gains = simc(model)
            trace = evaluate(Scenario(), gains).trace
            data = {"identified": asdict(model), "lambda": model.tau / 3,
                    "simc": asdict(gains), "zn": asdict(zn(model)),
                    "samples": [{"step": i, "temperature": float(trace.temperature[i]),
                                 "error": float(trace.temperature[i] - trace.setpoint[i]),
                                 "command": float(trace.command[i]),
                                 "next_temperature": float(trace.next_temperature[i])}
                                for i in [0, 20, 60, 100, 600]]}
        elif name == "bo":
            record = read_json(output / "04-bo/result.json")
            scenario = Scenario(**record["scenario"])
            seed = record["seed"]
            model, initial = initial_samples(scenario, seed)
            x = np.asarray([encode(g) for g in initial])
            y = np.asarray([evaluate(scenario, g).iae for g in initial])
            gp = fit_gp(x, y)
            candidates = np.random.default_rng(seed + 1000).random((1024, 2))
            mean, std = gp.predict(candidates, return_std=True)
            ei = expected_improvement(mean, std, y.min())
            i = int(ei.argmax())
            data = {"seed": seed, "scenario": asdict(scenario), "x": x.tolist(), "y": y.tolist(),
                    "candidate_count": len(candidates), "selected_index": i,
                    "selected_point": candidates[i].tolist(), "gains": asdict(decode(candidates[i])),
                    "mean": float(mean[i]), "std": float(std[i]), "best_observed": float(y.min()),
                    "z": float((y.min() - mean[i]) / std[i]), "ei": float(ei[i])}
        elif name == "fnn":
            model = FNN.load(output / "05-fnn/model.npz")
            obj = identify(Scenario())[0]
            x = [obj.gain, obj.tau, obj.delay]
            weights = features([x])[0]
            data = {"input": x, "weights": weights.tolist(), "weight_sum": float(weights.sum()),
                    "consequents": model.consequents.tolist(),
                    "log_gains": (weights @ model.consequents).tolist(),
                    "gains": asdict(model.predict(obj))}
        elif name in ONLINE:
            chapter = ONLINE[name][1]
            report = read_json(output / chapter / "training.json")
            field = "example_update" if name == "qlearning" else "first_update"
            if not report.get(field):
                raise ValueError(f"Run {name} with enough episodes to produce an actual update first")
            data = {"source": f"{chapter}/training.json", "seed": report["seed"],
                    "update": report[field]}
        elif name == "pg4pi":
            report = read_json(output / "14-pg4pi/result.json")
            data = {"source": "14-pg4pi/result.json", "first_trajectory": report["trials"][0],
                    "cost": report["cost"]}
        else:
            report = read_json(output / "08-llm/result.json")
            data = {"source": "08-llm/result.json", "status": report["status"],
                    "exchanges": report.get("exchanges", [])[:2],
                    "note": "Recorded responses only; this command never calls an API."}
        write_json(output / "explain" / f"{name}.json", data)
