"""Add the shared, step-by-step teaching frame to the report source files.

The Markdown files remain the source of truth.  This small idempotent pass only
adds the common reading route; it does not rewrite experiment numbers or
historical evidence.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "分报告"

ROUTES = {
    "00": "先建立问题地图：对象、七种方法、结果口径和仿真边界。",
    "01": "先跑通一条最短路径：安装、运行、找到产物，再判断这次运行是否成功。",
    "02": "先认识虚拟机房：热容、热阻、执行器和传感器，最后改一个参数观察曲线。",
    "03": "先固定比较规则：数据隔离、指标、预算、随机种子和回退门，再看任何排名。",
    "04": "先做阶跃辨识，再把 K、τ、L 代入 Z-N 公式，最后检查激进参数的代价。",
    "05": "先理解 λ 怎样改变快慢，再扫描候选、冻结 IMC 基线并检查稳定性。",
    "06": "先看一个候选怎样被评估，再跟踪 GP、EI 和真实仿真如何循环选点。",
    "07": "先把性能分和风险分分开，再观察安全门如何拒绝不可靠候选。",
    "08": "先生成标签和规则，再把连续状态落到规则表，观察在线增益和回退。",
    "09": "先定义状态、动作、奖励，再走查一次 TD 更新，最后检查覆盖与冻结策略。",
    "10": "先跑零费用 replay，再跟踪建议、工具、验收、回退和最终导出的参数。",
    "11": "先锁定同一张考卷，再读逐场景结果、配对统计、失败控制和成本口径。",
    "12": "先导出同一份策略，再做 Python/C++ 对拍和 PC-SIL，最后列出仍待实体验证的门。",
}


def route_block(prefix: str) -> str:
    run_line = (
        "本篇不运行新的训练批次，重点是读懂证据和口径。"
        if prefix in {"00", "03"}
        else "每一步都先看输入，再看中间产物，最后用表格或断言核对输出。"
    )
    return f"""## 阅读路线

{ROUTES[prefix]}

按下面的顺序阅读：

1. **问题**：先说清这一步想回答什么，避免把指标当成结论。
2. **准备**：列出前置章节、配置、数据来源和版本；{run_line}
3. **运行**：复制命令或读取封存产物，记录输出目录和随机种子。
4. **走查**：用一个具体数值代入公式，再对照实现中的关键几行。
5. **观察**：先读坐标、曲线和表格，再说明“发生了什么”和“这说明什么”。
6. **核对**：用容差、断言、哈希或独立验证判断复现是否成功。
7. **边界**：把已经证明、只作观察和仍待验证的内容分开写。

| 阅读检查 | 完成标准 |
|---|---|
| 输入明确 | 能说出本步使用的状态、参数、数据集或基线 |
| 中间过程可见 | 至少有一段数值演算、过程记录或关键代码 |
| 输出可核对 | 能定位到 CSV、图、日志、断言或封存目录 |
| 结论有边界 | 没有把仿真、replay、候选或稳定性误写成实体验收 |

---

"""


def enrich(path: Path) -> bool:
    prefix = path.name[:2]
    if prefix not in ROUTES:
        return False
    text = path.read_text(encoding="utf-8")
    marker = "## 阅读路线\n"
    if marker in text:
        return False
    needle = "## 这次要解决什么问题\n"
    if needle not in text:
        raise ValueError(f"missing section marker in {path}")
    text = text.replace(needle, route_block(prefix) + needle, 1)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    changed = [path.name for path in sorted(REPORT_DIR.glob("*.md")) if enrich(path)]
    print(f"enriched {len(changed)} reports")
    for name in changed:
        print(name)


if __name__ == "__main__":
    main()
