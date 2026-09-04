# -*- coding: utf-8 -*-
"""公式预检：每个 $$ 块公式 与 行内 $ 公式 都用 matplotlib mathtext 试渲一遍。

失败即非零退出 + 打印失败的 LaTeX。
运行：
    cd E:/HAVC/hvac-ai-pid-demo
    export PYTHONPATH="C:/Users/M00094113/AppData/Roaming/Python/Python314/site-packages"
    C:/Python314/python.exe reports/check_math.py <md 文件路径>
"""
import re
import sys
import tempfile
from pathlib import Path

REPORTS = Path(r"E:\HAVC\hvac-ai-pid-demo\reports")
sys.path.insert(0, str(REPORTS))
from md2docx import render_math_png  # noqa: E402

tmp = Path(tempfile.gettempdir()) / "_eqcheck.png"
bad = []


def main():
    if len(sys.argv) < 2:
        print("用法: check_math.py <md 文件> [<md 文件> ...]", file=sys.stderr)
        sys.exit(2)
    for src in sys.argv[1:]:
        text = Path(src).read_text(encoding="utf-8")
        name = Path(src).name
        print(f"=== {name} ===")
        for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S):
            tex = m.group(1).strip()
            ok = render_math_png(tex, tmp, 13)
            status = "OK " if ok else "BAD"
            print(f"  [{status}] block: {tex[:60]}...")
            if not ok:
                bad.append((name, "block", tex))
        body = re.sub(r"\$\$.+?\$\$", "", text, flags=re.S)
        for m in re.finditer(r"(?<!\$)\$([^$\n]+)\$(?!\$)", body):
            tex = m.group(1).strip()
            ok = render_math_png(tex, tmp, 11)
            status = "OK " if ok else "BAD"
            print(f"  [{status}] inline: {tex[:60]}...")
            if not ok:
                bad.append((name, "inline", tex))
    print()
    if bad:
        print(f"❌ 共 {len(bad)} 个公式渲染失败：")
        for name, kind, tex in bad:
            print(f"  {name} [{kind}] {tex}")
        sys.exit(1)
    print(f"✅ 全部公式渲染通过")


if __name__ == "__main__":
    main()
