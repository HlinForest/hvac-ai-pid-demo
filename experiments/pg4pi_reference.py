"""Run the pinned upstream PG4PI example in an isolated output directory.

The upstream repository is an independent state-space/controlgym example.  It
is useful as a provenance reference, but it is not imported by the HVAC
package and its REA setup is not presented as an HVAC reproduction.  This
module downloads the exact commit, optionally installs its dependencies under
the requested output directory, runs ``PG4PI.py`` with a seeded NumPy process,
and records the first update and captured result separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("pg4pi-upstream.json")


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _command(command, *, cwd=None, env=None, timeout=300):
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode:
        return completed, (
            f"command exited with {completed.returncode}: {' '.join(map(str, command))}\n"
            f"{completed.stderr[-4000:]}"
        )
    return completed, None


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _is_runtime_bytecode(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.endswith(".pyc") or "/__pycache__/" in normalized


def _tracked_worktree_changes(source: Path, timeout: int):
    status, error = _command(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        timeout=timeout,
    )
    if error:
        return None, error
    changes = []
    for line in status.stdout.splitlines():
        if len(line) < 3:
            continue
        state, path = line[:2], line[3:]
        # The upstream script writes plots, arrays and other untracked
        # artifacts beside the checkout.  Those are expected.  Python may
        # also rewrite tracked bytecode caches while importing the reference;
        # source/data changes remain blocking even when HEAD is unchanged.
        if state == "??" or _is_runtime_bytecode(path):
            continue
        changes.append({"state": state, "path": path})
    return changes, None


def _ensure_checkout(source: Path, config: dict, timeout: int):
    url = config["repository"]
    commit = config["commit"]
    if source.exists():
        verified, error = _command(["git", "-C", str(source), "rev-parse", "HEAD"], timeout=timeout)
        if error is None and verified.stdout.strip() == commit:
            changes, status_error = _tracked_worktree_changes(source, timeout)
            if status_error:
                return {"status": "blocked", "error": status_error, "source": str(source)}
            if changes:
                return {
                    "status": "blocked",
                    "error": "pinned checkout has tracked source/data changes; choose a new output directory",
                    "source": str(source),
                    "tracked_changes": changes,
                }
            return {
                "status": "reused",
                "commit": commit,
                "source": str(source),
                "tracked_changes": [],
            }
        return {
            "status": "blocked",
            "error": "output source exists but is not the pinned upstream commit; choose a new output directory",
            "source": str(source),
        }

    source.parent.mkdir(parents=True, exist_ok=True)
    # The source is deliberately checked out beneath the caller's ignored
    # output directory.  Nothing is installed into the project environment.
    cloned, error = _command(
        ["git", "clone", "--depth", "1", url, str(source)],
        timeout=timeout,
    )
    if error:
        return {"status": "blocked", "error": error, "source": str(source)}
    checked_out, error = _command(
        ["git", "-C", str(source), "fetch", "--depth", "1", "origin", commit],
        timeout=timeout,
    )
    if error:
        return {"status": "blocked", "error": error, "source": str(source)}
    checked_out, error = _command(
        ["git", "-C", str(source), "checkout", "--detach", commit],
        timeout=timeout,
    )
    if error:
        return {"status": "blocked", "error": error, "source": str(source)}
    verified, error = _command(["git", "-C", str(source), "rev-parse", "HEAD"], timeout=timeout)
    if error or verified.stdout.strip() != commit:
        return {
            "status": "blocked",
            "error": error or f"checkout resolved to {verified.stdout.strip()}, expected {commit}",
            "source": str(source),
        }
    return {"status": "downloaded", "commit": commit, "source": str(source)}


def _install_dependencies(source: Path, dependency_dir: Path, timeout: int):
    requirements = source / "requirements.txt"
    if not requirements.exists():
        return {"status": "blocked", "error": "upstream requirements.txt is missing"}
    dependency_dir.mkdir(parents=True, exist_ok=True)
    # The base project already carries NumPy, SciPy, matplotlib and CPU
    # PyTorch.  Only the upstream environment adapters are downloaded into
    # this output-local target; this avoids silently pulling a second (often
    # CUDA) torch or NumPy into the user's base environment.
    packages = [
        "gymnasium==0.29.1",
        "gym==0.26.2",
        "farama-notifications>=0.0.1",
        "gym-notices>=0.0.4",
    ]
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--target",
        str(dependency_dir),
        "--no-deps",
        *packages,
    ]
    completed, error = _command(command, timeout=timeout)
    if error:
        return {"status": "blocked", "error": error, "command": command}
    return {
        "status": "installed",
        "directory": str(dependency_dir),
        "command": command,
        "packages": packages,
        "base_dependency_note": "numpy/scipy/matplotlib/torch are inherited from the active environment; adapter packages are isolated under deps",
        "stdout": completed.stdout[-4000:],
    }


def _driver_code(seed: int) -> str:
    # The marker lets the parent process separate the compact numeric record
    # from plotting/library output while keeping the original script intact.
    return f'''\
import json
import linecache
import importlib.metadata
import platform
import runpy
import sys
import numpy as np
np.random.seed({int(seed)})
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None
_original_default_rng = np.random.default_rng
_seeded_generator = _original_default_rng({int(seed)})
def seeded_default_rng(value=None, *, seed=None):
    if seed is not None:
        value = seed
    return _seeded_generator if value is None else _original_default_rng(value)
np.random.default_rng = seeded_default_rng
updates = []
def trace(frame, event, arg):
    if event == "line" and frame.f_code.co_filename.endswith("PG4PI.py"):
        source_line = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
        if "env.K_p =" in source_line:
            local = frame.f_locals
            updates.append({{
                "episode": local.get("episode"),
                "return": local.get("G"),
                "action": local.get("a"),
                "mean": local.get("mean"),
                "score_p": local.get("score_p"),
                "score_i": local.get("score_i"),
                "K_p_before": getattr(local.get("env"), "K_p", None),
                "K_i_before": getattr(local.get("env"), "K_i", None),
                "alpha": local.get("alpha"),
            }})
    return trace
sys.settrace(trace)
values = runpy.run_path("PG4PI.py", run_name="__main__")
sys.settrace(None)

def serial(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    if isinstance(value, dict):
        return {{str(key): serial(item) for key, item in value.items()}}
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return repr(value)

env = values.get("env")
kp = serial(values.get("k_p_arr", []))
ki = serial(values.get("k_i_arr", []))
returns = serial(values.get("rew_arr", []))
actions = serial(values.get("u_arr", []))
score_p = serial(values.get("score_p", None))
score_i = serial(values.get("score_i", None))
last_action = serial(values.get("a", None))
last_mean = serial(values.get("mean", None))
last_return = serial(values.get("G", None))
update_records = serial(updates)
payload = {{
    "seed": {int(seed)},
    "runtime": {{"python": platform.python_version(), "platform": platform.platform(),
                "packages": {{name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "matplotlib", "torch", "gym", "gymnasium")}}}},
    "entrypoint": "PG4PI.py",
    "episode_count": len(returns),
    "returns": returns,
    "actions": actions,
    "kp_before_update": kp,
    "ki_before_update": ki,
    "environment": {{
        "id": serial(getattr(env, "id", None)),
        "sample_time": serial(getattr(env, "sample_time", None)),
        "noise_cov": serial(getattr(env, "noise_cov", None)),
        "K_p_final": serial(getattr(env, "K_p", None)),
        "K_i_final": serial(getattr(env, "K_i", None)),
    }},
    "last_update": {{
        "episode": max(0, len(returns) - 1),
        "return": last_return,
        "action": last_action,
        "mean": last_mean,
        "score_p": score_p,
        "score_i": score_i,
        "alpha": serial(values.get("alpha", None)),
    }},
    "first_update_trace": update_records[0] if update_records else None,
    "last_update_trace": update_records[-1] if update_records else None,
}}
if returns:
    payload["one_update"] = {{
        "episode": 0,
        "return": returns[0],
        "action": actions[0] if actions else None,
        "K_p_before": kp[0] if kp else None,
        "K_i_before": ki[0] if ki else None,
        "K_p_after": kp[1] if len(kp) > 1 else payload["environment"]["K_p_final"],
        "K_i_after": ki[1] if len(ki) > 1 else payload["environment"]["K_i_final"],
    }}
print("__PG4PI_REFERENCE_JSON__" + json.dumps(payload, allow_nan=False))
'''


def _run_upstream(source: Path, dependency_dir: Path, seed: int, timeout: int):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    path_entries = [str(source), str(dependency_dir)]
    if environment.get("PYTHONPATH"):
        path_entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    completed, error = _command(
        [sys.executable, "-c", _driver_code(seed)],
        cwd=source,
        env=environment,
        timeout=timeout,
    )
    if error:
        return None, error, completed
    marker = "__PG4PI_REFERENCE_JSON__"
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            payload = json.loads(line[len(marker):])
            break
    if payload is None:
        return None, "upstream completed without the numeric result marker", completed
    return payload, None, completed


def run(
    output="outputs/pg4pi-reference",
    *,
    seed: int = 0,
    install_deps: bool = False,
    execute: bool = True,
    timeout: int = 300,
):
    """Run the pinned upstream entrypoint and write an auditable record.

    ``install_deps`` is opt-in.  When enabled, pip receives ``--target`` and
    writes only beneath ``output/deps``; the repository's base environment is
    never modified.  A missing optional dependency or an upstream runtime
    error is returned as ``status='blocked'`` with captured stderr.
    """
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    source = output / config["execution"]["source_dir"]
    dependency_dir = output / config["execution"]["dependency_dir"]
    record = {
        "repository": config["repository"],
        "commit": config["commit"],
        "entrypoint": config["entrypoint"],
        "seed": seed,
        "output": str(output),
        "status": "blocked",
        "install_deps": bool(install_deps),
        "execute": bool(execute),
    }

    checkout = _ensure_checkout(source, config, timeout)
    record["checkout"] = checkout
    if checkout.get("status") == "blocked":
        record["error"] = checkout["error"]
        _write_json(output / "config.json", record)
        _write_json(output / "results.json", record)
        return record

    if install_deps:
        dependencies = _install_dependencies(source, dependency_dir, timeout)
    else:
        dependencies = {"status": "not_requested", "directory": str(dependency_dir)}
    record["dependencies"] = dependencies
    if dependencies.get("status") == "blocked":
        record["error"] = dependencies["error"]
        _write_json(output / "config.json", record)
        _write_json(output / "results.json", record)
        return record

    if not execute:
        record["status"] = "prepared"
        _write_json(output / "config.json", record)
        _write_json(output / "results.json", record)
        return record

    payload, error, completed = _run_upstream(source, dependency_dir, seed, timeout)
    (output / "stdout.txt").write_text(
        completed.stdout if completed is not None else "", encoding="utf-8"
    )
    (output / "stderr.txt").write_text(
        completed.stderr if completed is not None else "", encoding="utf-8"
    )
    if error:
        record["error"] = error
        record["status"] = "blocked"
        _write_json(output / "config.json", record)
        _write_json(output / "results.json", record)
        return record

    record["status"] = "complete"
    record["results"] = payload
    record["artifacts"] = sorted(
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    _write_json(output / "one-update.json", payload.get("one_update", {}))
    _write_json(
        output / "update-trace.json",
        {
            "first_update": payload.get("first_update_trace"),
            "last_update": payload.get("last_update_trace"),
            "note": "Trace records are captured immediately before the upstream K_p assignment; one-update.json contains the first gain update summary.",
        },
    )
    _write_json(output / "results.json", payload)
    _write_json(output / "config.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/pg4pi-reference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    result = run(
        args.output,
        seed=args.seed,
        install_deps=args.install_deps,
        execute=not args.skip_run,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("status") in {"complete", "prepared", "reused", "downloaded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
