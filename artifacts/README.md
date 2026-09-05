# artifacts — v4 唯一产物根

- 所有新实验产物写入 `artifacts/runs/<run_id>/`。
- `<run_id>` 格式：`v4-<yyyymmdd>-<shortsha>`，例如 `v4-20260906-2870262`。
- 每个 run 目录必须包含 `manifest.yaml`（由 `experiments/manifests/v4.yaml`
  复制并填入实际 `run_id`、代码 SHA、依赖版本）与 `environment.json`。
- 禁止脚本隐式读取顶层 `outputs_*`；历史 15 个 `outputs*` 目录为只读归档，
  迁移完成前不得覆盖或删除（见 `experiments/manifests/v4.yaml`）。
- Modelica 原始 CSV 哈希记录在 `provenance.csv`；无法复跑时标记
  “历史证据，当前未复现”。
