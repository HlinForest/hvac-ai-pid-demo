# 证据与复现

本网站承担教学阅读，不替换实验封存目录。当前主结果来自 `artifacts/runs/sealed-80x7-b4`，策略训练冻结来自 `artifacts/runs/v4-20260906-bf6bda6`。主统计量是逐场景配对比均值和 bootstrap 95% CI；验收必须同时通过 bounded、comfort、recovery 和 actuator 四项。

每篇报告的“如何确认复现成功”都给出本篇检查项。完整的哈希、数据隔离、预算台账、失败控制和硬件门状态见仓库根目录的 `EXPERIMENTS.md`、`docs/主报告.md` 和 `embedded/hardware_gates.md`。

网页构建时，`tools/build_site.py` 从 `docs/主报告.md` 和 `reports/分报告/*.md` 生成页面内容，并把现有 `reports/figures` 复制到网站资源目录。不要直接编辑 `site/content`，下一次构建会重新生成。
