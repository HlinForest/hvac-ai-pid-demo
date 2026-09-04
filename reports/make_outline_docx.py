# -*- coding: utf-8 -*-
"""把 实验报告生成.md（规范化写作大纲）转换为 docx 文档。

用法：python reports/make_outline_docx.py
输出：reports/实验报告生成.docx
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC = HERE / "实验报告生成.md"
DST = HERE / "实验报告生成.docx"

EAST_ASIA = "微软雅黑"
CODE_FONT = "Consolas"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)  # 深蓝
GRAY = RGBColor(0x59, 0x59, 0x59)


def set_run_font(run, size=10.5, bold=False, italic=False, color=None,
                 font= EAST_ASIA, ascii_font=None):
    run.font.name = ascii_font or font
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    # 中文字体必须走 eastAsia
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), font)


INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_inline(par, text, size=10.5, base_bold=False, color=None):
    """解析 **加粗** 与 `行内代码` 两种标记。"""
    for seg in INLINE_RE.split(text):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**") and len(seg) > 4:
            run = par.add_run(seg[2:-2])
            set_run_font(run, size=size, bold=True, color=color)
        elif seg.startswith("`") and seg.endswith("`") and len(seg) > 2:
            run = par.add_run(seg[1:-1])
            set_run_font(run, size=max(size - 1, 8), font=EAST_ASIA,
                         ascii_font=CODE_FONT, color=RGBColor(0xA3, 0x15, 0x15))
        else:
            run = par.add_run(seg)
            set_run_font(run, size=size, bold=base_bold, color=color)


def add_table(doc, header, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Table Grid"
    table.autofit = True
    for j, cell_text in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_inline(cell.paragraphs[0], cell_text, size=9, base_bold=True)
        # 表头底纹
        shd = cell._element.get_or_add_tcPr().makeelement(
            qn("w:shd"), {qn("w:fill"): "DEEAF6", qn("w:val"): "clear"})
        cell._element.get_or_add_tcPr().append(shd)
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            add_inline(cell.paragraphs[0], cell_text, size=9)
    doc.add_paragraph()  # 表后空行


def main():
    doc = Document()
    # 页面与默认段落
    for section in doc.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.4)
        section.right_margin = Cm(2.4)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 表格
        if stripped.startswith("|") and i + 1 < n and re.match(
                r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            add_table(doc, header, rows)
            continue

        if not stripped:
            i += 1
            continue
        if stripped == "---":
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(p, text, size=20, base_bold=True, color=ACCENT)
                doc.add_paragraph()
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(4)
                sizes = {2: 15, 3: 12.5, 4: 11}
                add_inline(p, text, size=sizes.get(level, 11), base_bold=True,
                           color=ACCENT)
                # 标题左侧竖线效果：用底边框代替
            i += 1
            continue

        # 引用块（开头使用说明）
        if stripped.startswith(">"):
            quote = stripped.lstrip("> ").rstrip()
            if not quote:
                i += 1
                continue
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.space_after = Pt(2)
            # 连续引用行合并为一个段落块
            add_inline(p, quote, size=9.5, color=GRAY)
            i += 1
            continue

        # 无序列表（含缩进的子项）
        m = re.match(r"^(\s*)-\s+(.*)$", line)
        if m:
            indent = len(m.group(1))
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75 + (0.6 if indent >= 2 else 0))
            p.paragraph_format.space_after = Pt(2)
            bullet = "•  " if indent < 2 else "–  "
            run = p.add_run(bullet)
            set_run_font(run, size=10.5, color=ACCENT)
            add_inline(p, m.group(2))
            i += 1
            continue

        # 有序列表（含缩进的子项，如 2.3 节辨识流程）
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            indent = len(m.group(1))
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75 + (0.6 if indent >= 2 else 0))
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(f"{m.group(2)}. ")
            set_run_font(run, size=10.5, bold=True, color=ACCENT)
            add_inline(p, m.group(3))
            i += 1
            continue

        # 普通段落
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        add_inline(p, stripped)
        i += 1

    # 页脚页码省略，保持简单
    DST.parent.mkdir(parents=True, exist_ok=True)
    doc.save(DST)
    print(f"已生成: {DST}")


if __name__ == "__main__":
    main()
