"""Chapter orchestration. Algorithms do not know about files or the website."""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from hvac_pid import plotting
from hvac_pid.artifacts import metadata, read_json, save_trace, write_csv, write_json
from hvac_pid.core import Gains, Scenario, evaluate
from hvac_pid.fnn import FNN
from hvac_pid.rl import QPolicy, validate
from hvac_pid.tuning import Identified, identify, initial_samples, simc, step_experiment, tune, zn
from .data import load_splits
from .methods import ONLINE, implementation, load_policies


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "experiments" / "splits.json"


def directory(output, chapter):
    path = Path(output) / chapter
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_evaluations(path, results):
    for name, result in results.items():
        save_trace(path / f"{name}.csv", result.trace)
    write_json(path / "metrics.json", {name: r.metrics() for name, r in results.items()})


def temperature(output, kp=0.15, ki=0.005, scenario=None):
    s = scenario or Scenario()
    path = directory(output, "01-temperature")
    results = {"Manual": evaluate(s, Gains(kp, ki)), "Aggressive": evaluate(s, Gains(2.0, 0.5))}
    save_evaluations(path, results)
    write_json(path / "config.json", {"scenario": asdict(s), "manual_gains": {"kp": kp, "ki": ki}})
    plotting.responses(results, path / "response.svg")


def plant(output, scenario=None):
    s = scenario or Scenario()
    path = directory(output, "02-plant")
    model, time, temperature = identify(s)
    write_json(path / "identification.json", {"scenario": asdict(s), "identified": asdict(model)})
    write_csv(path / "step.csv", [{"minute": t, "temperature": y} for t, y in zip(time, temperature)])
    curves = []
    for name, obj in [("Nominal", s), ("tau = 30 min", replace(s, tau=30)),
                      ("delay = 5 min", replace(s, delay=5)), ("K = 10", replace(s, gain=10))]:
        t, y = step_experiment(obj)
        curves.append((name, t, y))
    plotting.step_curves(curves, path / "step.svg")


def classical(output, scenario=None):
    s = scenario or Scenario()
    path = directory(output, "03-classical")
    model = identify(s)[0]
    settings = {"ZN": zn(model), "SIMC": simc(model)}
    results = {name: evaluate(s, g) for name, g in settings.items()}
    save_evaluations(path, results)
    write_json(path / "gains.json", {"identified": asdict(model), "gains": {n: asdict(g) for n, g in settings.items()}})
    plotting.responses(results, path / "response.svg")
    sweep = {f"lambda = {lam} min": evaluate(s, simc(model, lam)) for lam in [2, 7, 20]}
    plotting.responses(sweep, path / "lambda.svg")
    write_json(path / "lambda.json", {n: r.metrics() for n, r in sweep.items()})


def bo(output, seed=0, rounds=15, scenario=None):
    s = scenario or Scenario()
    path = directory(output, "04-bo")
    result = tune(s, seed, rounds)
    write_json(path / "result.json", {**result.to_dict(), "scenario": asdict(s), "seed": seed})
    write_csv(path / "trials.csv", result.trials)
    plotting.search(result.trials, path / "search.svg")
    plotting.surrogate(result.trials, path / "surrogate.svg")
    evaluation = evaluate(s, result.best)
    save_evaluations(path, {"BO": evaluation})
    plotting.responses({"BO": evaluation}, path / "response.svg")
    return result


def fnn(output, seed=0, rounds=15, progress=print):
    path = directory(output, "05-fnn")
    split = load_splits(SPLITS)
    write_json(path / "splits.json", read_json(SPLITS))
    start = perf_counter()
    labels = {"train": [], "validation": []}
    for split_name in labels:
        for index, scenario in enumerate(split[split_name]):
            identified = identify(scenario)[0]
            tuned = tune(scenario, seed + index, rounds)
            labels[split_name].append({"index": index, "scenario": asdict(scenario),
                                       "identified": asdict(identified), "gains": asdict(tuned.best),
                                       "cost": tuned.cost, "trials": tuned.trials})
            write_json(path / "labels.json", labels)
            if progress:
                progress(f"FNN teacher {split_name} {index + 1}/{len(split[split_name])}")
    label_seconds = perf_counter() - start
    x = [[r["identified"][k] for k in ("gain", "tau", "delay")] for r in labels["train"]]
    y = np.asarray([[r["gains"][k] for k in ("kp", "ki")] for r in labels["train"]])
    candidates = []
    fit_start = perf_counter()
    for ridge in [0.0001, 0.001, 0.01]:
        model = FNN.fit(x, y, ridge)
        validation_iae = [evaluate(s, model.predict(Identified(**row["identified"]))).iae
                          for s, row in zip(split["validation"], labels["validation"])]
        candidates.append((np.mean(validation_iae), ridge, model))
    _, ridge, model = min(candidates, key=lambda c: c[0])
    model.save(path / "model.npz")
    selected = [{"ridge": r, "mean_validation_iae": float(loss)} for loss, r, _ in candidates]
    prediction = np.asarray([[g.kp, g.ki] for g in
                             (model.predict(Identified(*row)) for row in x)])
    plotting.fnn_predictions(y, prediction, model.consequents, path / "learning.svg")
    nominal = Scenario()
    gains = model.predict(identify(nominal)[0])
    result = evaluate(nominal, gains)
    save_evaluations(path, {"FNN": result})
    plotting.responses({"FNN": result}, path / "response.svg")
    simulation_count = sum(r["cost"]["simulations"] for rows in labels.values() for r in rows)
    write_json(path / "training.json", {
        "seed": seed, "ridge": ridge, "selection": selected, "nominal_gains": asdict(gains),
        "label_source": "BO", "train_objects": 48, "validation_objects": 12,
        "cost": {"label_simulations": simulation_count,
                 "label_plant_steps": sum(r["cost"]["plant_steps"] for rows in labels.values() for r in rows),
                 "identification_steps": 2 * 60 * round(240 / nominal.dt),
                 "selection_simulations": 36, "selection_plant_steps": 36 * nominal.steps,
                 "label_seconds": label_seconds, "fit_and_selection_seconds": perf_counter() - fit_start,
                 "seconds": perf_counter() - start},
    })
    return model


def rl_experiment(output, method, seed=0, episodes=500, progress=print):
    learner = implementation(method)
    name, chapter, _, _, model_file = ONLINE[method]
    path = directory(output, chapter)
    split = load_splits(SPLITS)
    write_json(path / "splits.json", read_json(SPLITS))
    try:
        policy, report = learner.train(split["train"], episodes, seed, progress)
    except Exception as error:
        write_json(path / "failure.json", {"method": name, "seed": seed, "requested_episodes": episodes,
                                          "status": "failed", "error": str(error), "environment": metadata()})
        raise
    snapshots = report.pop("snapshots", None)
    if snapshots:
        np.savez(path / "snapshots.npz", **{f"snapshot_{i}": q for i, q in enumerate(snapshots)})
        plotting.q_snapshots(snapshots, path / "qtable.svg")
    report["seed"] = seed
    if hasattr(policy, "config"):
        report["config"] = policy.config
    report["validation_iae"] = validate(policy, split["validation"])
    report["cost"]["validation_simulations"] = len(split["validation"])
    report["cost"]["validation_plant_steps"] = sum(s.steps for s in split["validation"])
    report["cost"]["validation_identification_steps"] = sum(round(240 / s.dt) for s in split["validation"])
    policy.save(path / model_file)
    write_json(path / "training.json", report)
    write_csv(path / "history.csv", report["history"])
    plotting.learning(report["history"], path / "learning.svg", name)
    s = Scenario()
    anchor = simc(identify(s)[0])
    results = {"SIMC": evaluate(s, anchor), name: evaluate(s, anchor, policy)}
    save_evaluations(path, results)
    plotting.responses(results, path / "response.svg", gains=True)
    return policy


def pg4pi(output, seed=0, episodes=500):
    from hvac_pid.pg4pi import tune as tune_pi
    path = directory(output, "14-pg4pi")
    s = Scenario()
    try:
        result = tune_pi(s, seed=seed, episodes=episodes)
    except Exception as error:
        write_json(path / "failure.json", {"method": "PG4PI-HVAC", "seed": seed, "requested_episodes": episodes,
                                          "status": "failed", "error": str(error), "environment": metadata()})
        raise
    write_json(path / "result.json", {**result.to_dict(), "scenario": asdict(s), "seed": seed})
    write_csv(path / "trials.csv", result.trials)
    if result.best is None:
        raise RuntimeError("PG4PI did not return a fixed PI controller: " + result.stop_reason)
    results = {"SIMC": evaluate(s, simc(identify(s)[0])), "PG4PI-HVAC": evaluate(s, result.best)}
    save_evaluations(path, results)
    plotting.responses(results, path / "response.svg")
    plotting.pg_learning(result.trials, path / "learning.svg")
    return result


def llm(output, seed=0, rounds=15, live=False, **connection):
    from hvac_pid import llm as module
    path = directory(output, "08-llm")
    if not live:
        report = {"best": None, "trials": [], "status": "not_run",
                  "stop_reason": "Live experiment not requested. Run the llm command with --live.",
                  "cost": {"calls": 0, "simulations": 0}, "exchanges": []}
        write_json(path / "result.json", report)
        return report
    result = module.tune(Scenario(), seed, rounds,
                         record=lambda data: write_json(path / "result.json", data), **connection)
    data = read_json(path / "result.json")
    data.update({"seed": seed, "scenario": asdict(Scenario()), "live": True})
    write_json(path / "result.json", data)
    if result.best:
        plotting.search(result.trials, path / "search.svg")
        evaluation = evaluate(Scenario(), result.best)
        save_evaluations(path, {"LLM": evaluation})
        plotting.responses({"LLM": evaluation}, path / "response.svg")
    return data


class TimedPolicy:
    def __init__(self, policy):
        self.policy, self.seconds, self.calls = policy, 0.0, 0
        self.action_mode = policy.action_mode

    def choose(self, observation):
        start = perf_counter()
        result = self.policy.choose(observation)
        self.seconds += perf_counter() - start
        self.calls += 1
        return result


def compare(output, seed=0, rounds=15, include_dqn=True, episodes=500):
    """New-object test: commission each object once; never refit learned policies.

    Frozen-parameter tests reuse the nominal commissioning gains for shifted
    objects. Both protocols are exported separately.
    """
    path = directory(output, "09-compare")
    s = Scenario()
    model = identify(s)[0]
    fnn_model = FNN.load(Path(output) / "05-fnn/model.npz")
    policies = load_policies(Path(output), include_dqn)
    for method, (name, chapter, *_) in ONLINE.items():
        if name in policies:
            training = read_json(Path(output) / chapter / "training.json")
            if training["seed"] != seed or training["cost"]["episodes"] != episodes:
                raise ValueError(f"{name} model must match the requested seed and training budget")
    bo_data = read_json(Path(output) / "04-bo/result.json")
    if bo_data["scenario"] != asdict(s) or bo_data["seed"] != seed or bo_data["cost"]["proposal_rounds"] != rounds:
        raise ValueError("BO record must match nominal scenario, seed and proposal budget")
    fixed = {"ZN": zn(model), "SIMC": simc(model), "BO": Gains(**bo_data["best"]),
             "FNN": fnn_model.predict(model)}
    pg_data = read_json(Path(output) / "14-pg4pi/result.json")
    if pg_data["scenario"] != asdict(s) or pg_data["seed"] != seed or pg_data["cost"]["episodes"] != episodes:
        raise ValueError("PG4PI record must match nominal scenario, seed and trajectory budget")
    fixed["PG4PI-HVAC"] = Gains(**pg_data["best"])
    llm_data = read_json(Path(output) / "08-llm/result.json")
    if llm_data["status"] == "complete" and llm_data["best"] and llm_data["cost"]["calls"]:
        if (llm_data.get("scenario") != asdict(s) or llm_data.get("seed") != seed
                or llm_data["cost"]["proposal_rounds"] != rounds):
            raise ValueError("LLM record must match nominal scenario, seed and proposal budget")
        fixed["LLM"] = Gains(**llm_data["best"])
    cases = {"nominal": s, "slower": replace(s, tau=30), "long_delay": replace(s, delay=5),
             "larger_load": replace(s, disturbance=1.5)}
    rows = []
    for case, scenario in cases.items():
        results = {name: evaluate(scenario, gains) for name, gains in fixed.items()}
        inference = {}
        for name, policy in policies.items():
            timed = TimedPolicy(policy)
            results[name] = evaluate(scenario, fixed["SIMC"], timed)
            inference[name] = {"inference_calls": timed.calls, "inference_seconds": timed.seconds}
        save_evaluations(path / case, results)
        plotting.responses(results, path / case / "response.svg")
        plotting.responses({n: results[n] for n in fixed}, path / case / "fixed.svg")
        plotting.responses({n: results[n] for n in ["SIMC", *policies]}, path / case / "online.svg", gains=True)
        for name, result in results.items():
            rows.append({"case": case, "method": name,
                         "mode": "online" if name in policies else "fixed",
                         **result.metrics(), **inference.get(name, {"inference_calls": 0, "inference_seconds": 0})})
    write_csv(path / "frozen.csv", rows)
    write_json(path / "frozen.json", rows)
    # Test split: fixed hyperparameters, new commissioning data, no model learning.
    holdout = []
    for index, obj in enumerate(load_splits(SPLITS)["test"]):
        from hvac_pid.pg4pi import tune as tune_pi
        identified = identify(obj)[0]
        tuned = tune(obj, seed, rounds)
        pg_tuned = tune_pi(obj, seed=seed, episodes=episodes)
        write_json(path / "pg4pi-holdout" / f"object-{index}.json", {
            **pg_tuned.to_dict(), "scenario": asdict(obj), "seed": seed})
        settings = {"ZN": zn(identified), "SIMC": simc(identified), "BO": tuned.best,
                    "FNN": fnn_model.predict(identified), "PG4PI-HVAC": pg_tuned.best}
        for name in [*settings, *policies]:
            gains = settings["SIMC"] if name in policies else settings[name]
            result = evaluate(obj, gains, policies.get(name))
            holdout.append({"object": index, "scenario": asdict(obj), "method": name,
                            "mode": "online" if name in policies else "fixed", **result.metrics(),
                            "commissioning_steps": round(240 / obj.dt),
                            "tuning_simulations": tuned.cost["simulations"] if name == "BO" else
                                pg_tuned.cost["simulations"] if name == "PG4PI-HVAC" else 0})
        if index == 0:
            write_json(path / "holdout-example-bo.json", tuned.to_dict())
    write_json(path / "holdout.json", holdout)
    plotting.distributions(holdout, path / "holdout.svg")
    costs = {
        "ZN": {"simulations": 0, "shared_classical_identification_steps": round(240 / s.dt)},
        "SIMC": {"simulations": 0, "shared_classical_identification_steps": round(240 / s.dt)},
        "BO": bo_data["cost"], "FNN": read_json(Path(output) / "05-fnn/training.json")["cost"],
        "Q-Learning": read_json(Path(output) / "06-qlearning/training.json")["cost"],
        "LLM": {"status": llm_data["status"], **llm_data["cost"]},
        "PG4PI-HVAC": pg_data["cost"],
    }
    for method, (name, chapter, *_) in ONLINE.items():
        if include_dqn or method == "qlearning":
            costs[name] = read_json(Path(output) / chapter / "training.json")["cost"]
    write_json(path / "costs.json", costs)
    write_json(path / "protocol.json", {
        "seed": seed, "scenario": asdict(s), "frozen_cases": {k: asdict(v) for k, v in cases.items()},
        "frozen": "All initial gains/anchors come from nominal commissioning. No retuning on shifts.",
        "holdout": "Each new object is commissioned. FNN and online policies stay frozen; BO and PG4PI tune the new object. No hyperparameters are selected from test scores.",
        "llm_status": llm_data["status"], "llm_holdout": "Not run; no implicit API calls during comparison.",
        "environment": metadata(),
    })


def all_experiments(output, seed=0, rounds=15, episodes=500, include_dqn=True):
    write_json(Path(output) / "environment.json", metadata())
    temperature(output)
    plant(output)
    classical(output)
    bo(output, seed, rounds)
    fnn(output, seed, rounds)
    for method in ONLINE:
        if include_dqn or method == "qlearning":
            rl_experiment(output, method, seed, episodes)
    pg4pi(output, seed, episodes)
    # Preserve an explicit live experiment; never replace it with a placeholder.
    if not (Path(output) / "08-llm/result.json").exists():
        llm(output)
    compare(output, seed, rounds, include_dqn, episodes)
    from .benchmark import benchmark
    benchmark(output, include_dqn=include_dqn)
    if episodes >= 5:
        from .explain import explain
        for method in ["classical", "bo", "fnn", "qlearning", "pg4pi", "llm", *(["dqn", "ppo", "td3", "sac", "crossq"] if include_dqn else [])]:
            explain(output, method, seed)
