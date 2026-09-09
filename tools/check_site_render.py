"""Check the built VitePress site for rendered math and resolvable images."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "site" / ".vitepress" / "dist"
BASE_PATH = "/hvac-ai-pid-demo/"


def chapter_html(chapter: str) -> list[Path]:
    return sorted((DIST / "chapters").glob(f"{chapter}*.html")) + sorted(
        (DIST / "chapters" / chapter).glob("**/*.html")
    )


def main() -> int:
    if not DIST.exists():
        print(f"missing build directory: {DIST}", file=sys.stderr)
        return 1

    html_files = sorted(DIST.rglob("*.html"))
    if not html_files:
        print("no HTML files in VitePress output", file=sys.stderr)
        return 1

    math_failures: list[str] = []
    image_failures: list[str] = []
    image_count = 0
    for chapter in ("04", "05", "09"):
        pages = chapter_html(chapter)
        if not pages:
            math_failures.append(f"chapter {chapter}: HTML page not found")
            continue
        chapter_text = "\n".join(page.read_text(encoding="utf-8") for page in pages)
        if 'class="katex' not in chapter_text:
            math_failures.append(f"chapter {chapter}: no KaTeX markup")
        if re.search(r">\s*\$\$", chapter_text):
            math_failures.append(f"chapter {chapter}: raw display formula remains")

    for page in html_files:
        text = page.read_text(encoding="utf-8")
        for source in re.findall(r'<img[^>]+src="([^"]+)', text):
            image_count += 1
            path = unquote(urlsplit(source).path)
            if path.startswith(BASE_PATH):
                relative = path[len(BASE_PATH) :]
            elif path.startswith("/"):
                relative = path[1:]
            else:
                continue
            if not (DIST / relative).is_file():
                image_failures.append(f"{page.relative_to(DIST)} -> {source}")

    if math_failures or image_failures:
        for failure in math_failures + image_failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"site render OK: {len(html_files)} HTML files, {image_count} image references, KaTeX 04/05/09")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
