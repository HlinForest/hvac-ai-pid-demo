"""A real base-only run must never reach optional torch imports."""
from pathlib import Path
import subprocess
import sys


def test_all_numpy_methods_run_with_torch_import_blocked(tmp_path):
    program = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise RuntimeError('Base experiment attempted to import optional torch')
sys.meta_path.insert(0, NoTorch())
from hvac_pid.__main__ import main
raise SystemExit(main(['all', '--without-torch', '--episodes', '1', '--rounds', '0', '--output', sys.argv[1]]))
"""
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "14-pg4pi/metrics.json").exists()
    assert (tmp_path / "09-compare/holdout.json").exists()
    assert not (tmp_path / "07-dqn").exists()
    assert not (tmp_path / "10-ppo").exists()
