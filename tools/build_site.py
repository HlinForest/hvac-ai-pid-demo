"""Build the VitePress source tree from the report Markdown sources.

The report Markdown remains canonical. Generated content is ignored by the
site build, so a fresh build cannot silently keep an old chapter.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
CHAPTERS = SITE / "chapters"
MAIN = SITE / "main.md"
LEGACY_CONTENT = SITE / "content"
PUBLIC = SITE / "public"


def rewrite_assets(text: str) -> str:
    # Report sources use both figures/foo.png and ../figures/foo.png depending
    # on their original directory. All site pages use the public URL.
    text = re.sub(r"(!\[[^\]]*\]\()((?:\.\./)?figures/)", r"\1/figures/", text)
    text = re.sub(r"(?<![\w/])figures/([\w\-\.\u4e00-\u9fff]+\.png)", r"/figures/\1", text)
    return text


def copy_markdown(source: Path, destination: Path) -> None:
    destination.write_text(rewrite_assets(source.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")


def main() -> None:
    if CHAPTERS.exists():
        shutil.rmtree(CHAPTERS)
    if MAIN.exists():
        MAIN.unlink()
    # Remove the pre-release layout if a developer ran an older version of
    # this builder. Keeping both layouts would create duplicate routes.
    if LEGACY_CONTENT.exists():
        shutil.rmtree(LEGACY_CONTENT)
    CHAPTERS.mkdir(parents=True)
    for source in sorted((ROOT / "reports" / "分报告").glob("*.md")):
        prefix = source.name[:2]
        if prefix.isdigit():
            copy_markdown(source, CHAPTERS / f"{prefix}.md")
    copy_markdown(ROOT / "docs" / "主报告.md", MAIN)

    figures = PUBLIC / "figures"
    if figures.exists():
        shutil.rmtree(figures)
    figures.mkdir(parents=True)
    # The root directory contains the original report figures; the teaching
    # reports add their step-by-step figures in their own single-source folder.
    # Copying both keeps the Markdown links stable while the source files stay
    # in their existing locations.
    source_dirs = (
        ROOT / "reports" / "figures",
        ROOT / "reports" / "分报告" / "figures",
        ROOT / "docs" / "figures",
        ROOT / "artifacts" / "runs" / "sealed-80x7-b4" / "figures",
    )
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for source in source_dir.glob("*.png"):
            shutil.copy2(source, figures / source.name)

    downloads = PUBLIC / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    main_docx = ROOT / "docs" / "主报告.docx"
    if main_docx.exists():
        shutil.copy2(main_docx, downloads / main_docx.name)
    archive = shutil.make_archive(str(downloads / "分报告"), "zip", ROOT / "reports" / "分报告")
    print(f"site source ready: {len(list(CHAPTERS.glob('*.md')))} chapters")
    print(f"figures: {len(list(figures.glob('*.png')))}; downloads: {Path(archive).name}")


if __name__ == "__main__":
    main()
