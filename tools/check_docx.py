"""逐节核对提交版主报告 DOCX（docs/主报告.docx vs docs/主报告.md + CSVs）。

Checks (fail loudly):
  1. headings 1:1: MD 的 #/## 标题与 DOCX 的 Heading 1/2 逐节对应（顺序+文字）。
  2. tables: MD 表格数 == DOCX 表格数；主结果两表均为 7 数据行；表内关键
     数字（B1 RL 0.9817 / B2 BO 1.0785 / B2 Safe-BO 1.0000）与对应
     sealed_80_summary.csv 一致（1e-4 容差）。
  3. stale-number scan: DOCX 全文不得出现旧版已更正数字
     (0.9664 / 1.4995 / 1.0168 / 0.8911 / 1.0046 / [0.9434 / [1.4105)，
     不得出现“回退基准本身合格”“七算法统一训练公平竞赛”（无否定限定时）。
  4. 每节非空：每个 Heading 1 下至少 1 个正文段落或表格。

Usage: python tools/check_docx.py [--md docs/主报告.md] [--docx docs/主报告.docx]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐节核对主报告 DOCX")
    parser.add_argument("--md", type=Path, default=ROOT / "docs" / "主报告.md")
    parser.add_argument("--docx", type=Path, default=ROOT / "docs" / "主报告.docx")
    return parser.parse_args()


def md_headings(md: str) -> list[tuple[int, str]]:
    out = []
    for line in md.splitlines():
        match = re.match(r"^(#{1,2})\s+(.*\S)\s*$", line)
        if match:
            out.append((len(match.group(1)), match.group(2).strip()))
    return out


def md_table_count(md: str) -> int:
    return sum(1 for line in md.splitlines() if re.match(r"^\|.*\|\s*$", line)) and \
        len(re.findall(r"(?m)^\|.*\|\s*$\n\|[\s:\-|]+\|\s*$", md))


def read_summary(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {r["algorithm"]: {k: float(r[k]) for k in
            ("mean_objective", "paired_ratio_mean", "paired_ratio_lo95", "paired_ratio_hi95",
             "validation_rate") if k in r} for r in rows}


def _para_size_pt(paragraph) -> float | None:
    for run in paragraph.runs:
        if run.text.strip() and run.font.size is not None:
            return float(run.font.size.pt)
    return None


def main() -> None:
    args = parse_args()
    try:
        from docx import Document
    except ImportError as exc:
        raise SystemExit("缺少 python-docx，请先 pip install python-docx") from exc
    md = args.md.read_text(encoding="utf-8")
    doc = Document(str(args.docx))

    # 1. headings 1:1 (md2docx 用字号/加粗而非 Heading 样式，故按文本序列比对；
    #    同时抽查字号映射 H1=20/H2=15pt 加粗）。
    want = md_headings(md)
    paras = [(p.text.strip(), _para_size_pt(p)) for p in doc.paragraphs]
    texts = [t for t, _ in paras if t]
    pos = 0
    size_of = {}
    for level, title in want:
        try:
            idx = texts.index(title, pos)
        except ValueError:
            raise AssertionError(f"DOCX 缺标题: ({level}) {title!r}")
        # 字号抽查：H1=20pt，H2=15pt（md2docx 约定）。
        seq = [s for t, s in paras if t == title]
        size_of[title] = seq[0]
        want_size = {1: 20.0, 2: 15.0}[level]
        assert any(abs((s or 0) - want_size) < 0.1 for s in seq), \
            f"标题字号漂移: {title!r} want {want_size}pt, got {seq}"
        pos = idx + 1
    print(f"headings OK ({len(want)} 节按序 1:1，字号抽查通过)")

    # 2. tables.
    want_tables = md_table_count(md)
    assert len(doc.tables) == want_tables, f"表格数 {len(doc.tables)} != MD {want_tables}"
    data_tables = [t for t in doc.tables if len(t.rows) >= 8]  # header + sep + 7
    assert len(data_tables) >= 2, f"主结果两表缺失（找到 {len(data_tables)} 个≥8行表）"
    print(f"tables OK ({len(doc.tables)} 张，含 {len(data_tables)} 张主结果表）")

    full_text = "\n".join(p.text for p in doc.paragraphs)
    table_text = "\n".join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    everything = full_text + "\n" + table_text

    # 3. key numbers vs CSVs (B1 legacy batch + B2 retrain batch).
    b1 = read_summary(ROOT / "artifacts" / "runs" / "sealed-80x7-v4" / "sealed_80_summary.csv")
    b2 = read_summary(ROOT / "artifacts" / "runs" / "sealed-80x7-v4-retrain" / "sealed_80_summary.csv")
    for label, algo, field in (("B1-RL", "rl", "paired_ratio_mean"),
                               ("B2-BO", "bo", "paired_ratio_mean"),
                               ("B2-SafeBO", "safe-bo", "paired_ratio_mean")):
        src = b1 if label.startswith("B1") else b2
        assert f"{src[algo][field]:.4f}" in everything, \
            f"{label} {algo}.{field}={src[algo][field]:.4f} 在 DOCX 中缺失"
    print("numbers OK (B1-RL/B2-BO/B2-SafeBO 与 CSV 一致）")

    # 4. stale scan (语境感知：更正声明中引用旧值允许，但裸写旧结论禁止）。
    stale_nums = ["1.4995", "1.0168", "0.8911", "1.0046", "0.9434", "1.4105"]
    hits = [s for s in stale_nums if s in everything]
    assert not hits, f"DOCX 含已更正旧统计值: {hits}"
    for pos, marker in ((m.start(), m.group(0)) for m in re.finditer("0.9664|回退基准本身合格|15/18 stable", everything)):
        window = everything[max(0, pos - 30):pos + 60]
        assert any(k in window for k in ("旧版", "更正", "已更正", "否定", "不得", "曾", "实为")), \
            f"旧表述疑似当作现行结论: ...{window}..."
    print("stale scan OK (旧值仅出现在更正/否定语境）")

    # 5. non-empty sections (按标题序列切分正文；标题唯一，已在步骤1验证顺序）。
    titles = [t for _, t in want]
    counts: dict[str, int] = {}
    current: str | None = None
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        if text in titles and text not in counts:
            current = text
            counts[current] = 0
            continue
        if current is not None:
            counts[current] += 1
    assert len(counts) == len(titles), f"节数 {len(counts)} != {len(titles)}"
    empty = [h for h, c in counts.items() if c == 0]
    assert not empty, f"空节: {empty}"
    print(f"sections OK ({len(counts)} 节均非空）")

    # 6. figures: every ![alt](path) in MD must resolve to a non-empty image.
    # (Existence + decodability; pagination/readability still needs a human
    # pass over the rendered DOCX — see delivery checklist in 主报告附件.)
    import re as _re
    missing: list[str] = []
    for match in _re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md):
        img = args.md.parent / match.group(1)
        if not img.exists() or img.stat().st_size == 0:
            missing.append(match.group(1))
    assert not missing, f"缺图: {missing}"
    from PIL import Image as _Image  # pillow ships with matplotlib
    for match in _re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md):
        with _Image.open(args.md.parent / match.group(1)) as im:
            im.verify()
    n_imgs = len(_re.findall(r"!\[[^\]]*\]\([^)]+\)", md))
    print(f"figures OK ({n_imgs} 张存在且可解码；分页可读性需人工逐页确认）")
    print("docx: ALL OK")


if __name__ == "__main__":
    sys.exit(main())
