from __future__ import annotations

import os

from hvac_pid.env import load_project_env


def test_load_project_env_reads_dotenv_and_respects_real_env(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "# comment line\n"
        "LOAD_ENV_TEST_KEY=file-value\n"
        "QUOTED_KEY=\"quoted value\"\n"
        "malformed line without equals\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOAD_ENV_TEST_KEY", "shell-value")
    loaded = load_project_env(tmp_path)
    assert os.environ["LOAD_ENV_TEST_KEY"] == "shell-value"  # real env wins
    assert os.environ["QUOTED_KEY"] == "quoted value"
    assert loaded == {"QUOTED_KEY": "quoted value"}


def test_load_project_env_missing_file_is_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("QUOTED_KEY", raising=False)
    assert load_project_env(tmp_path) == {}
