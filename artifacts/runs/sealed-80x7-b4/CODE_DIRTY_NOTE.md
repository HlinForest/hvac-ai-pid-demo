# B4 `code_dirty=true` 说明（事后重建，原始运行时仅保留哈希）

- 运行时记录：`sealed_provenance.json: code_sha=e7a22ae`，`code_dirty=true`，`worktree_status_sha16=2ce4cc8434e8fad7`。
  当时 `tools/run_sealed_evaluation.py:_git_is_dirty` 只保存 `git status --porcelain` 的 SHA-16，不保存修改内容本身（improve.md §3 指出该缺陷）。
- 事后重建：`git diff e7a22ae..1d67d9a --name-only`（`code_sha`..B4 提交）仅触及：
  `artifacts/runs/sealed-80x7-b4/*`（本目录 B4 新证据）、`docs/主报告.md/docx` + `docs/figures/` 两张同轴图、
  `EXPERIMENTS.md`/`RESULTS.md`、`reports/分报告/11_*`、两份 `PRUNED.md`、`tools/check_docx.py`。
  未触及 `hvac_pid/*.py`、任何 `tools/run_*.py`、`tools/make_*.py`、`tools/analyze_failures.py`、`tools/failure_control_tests.py`。
  `tools/check_docx.py` 的差异仅把关键数字审计从 B2/B3 重定向到 B4（审计工具，不参与仿真数值）。
- 结论：当时的脏工作区为“尚未提交的 B4 输出 + 报告编写中修改”，不含仿真源码修改；数值结果不受影响。
  若当时脏区含源码修改，本文件应附代码快照/补丁——经上述核查，不需要。
- 后续：`tools/run_sealed_evaluation.py` 已新增 `_git_worktree_evidence()`，
  新运行会在 `sealed_provenance.json` 中同时记录 `worktree_status_lines` 与 `worktree_diff_stat`（本 B4 文件为旧格式，上述两字段为事后占位说明，以本文件为准）。
- 训练批次原始文件：`artifacts/runs/v4-20260906-bf6bda6/` 部分渲染中间文件已按 `PRUNED.md` 裁剪（未随仓库保存，仅保留删除前哈希）；
  冻结策略、选型历史、数据划分、汇总指标与溯源全部保留；B4 本目录（0.81 MB）全量入库、无裁剪。
  已删文件勿用重跑文件冒充原始文件（`PRUNED.md` 已修正措辞）。
- Release：当前仓库暂无 Release 附件；完整原始证据以本仓库 `artifacts/runs/sealed-80x7-b4/` + `SHA256SUMS`/`evidence_index.csv` 为准。
