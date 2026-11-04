"""Render .github/README.md as the single page of the GitHub Pages site.

Usage: python scripts/build_pages.py OUTPUT_DIR   (needs: pip install markdown==3.9)
"""

import html
import sys
from pathlib import Path

import markdown

README = Path(__file__).resolve().parent.parent / ".github" / "README.md"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<style>
  :root {{ color-scheme: light dark; --fg: #1b1f24; --bg: #ffffff; --muted: #59636e; --line: #d1d9e0;
          --code: #f6f8fa; --link: #0969da; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg: #e6edf3; --bg: #0d1117; --muted: #9198a1; --line: #3d444d; --code: #151b23; --link: #4493f8; }}
  }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
         font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  main {{ max-width: 860px; margin: 0 auto; padding: 32px max(16px, env(safe-area-inset-left)) 64px; }}
  h1, h2 {{ border-bottom: 1px solid var(--line); padding-bottom: .3em; }}
  a {{ color: var(--link); }}
  code {{ font: 85% ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--code);
         padding: .2em .4em; border-radius: 6px; }}
  pre {{ background: var(--code); padding: 16px; border-radius: 6px; overflow-x: auto; }}
  pre code {{ padding: 0; background: none; }}
  table {{ border-collapse: collapse; display: block; overflow-x: auto; }}
  th, td {{ border: 1px solid var(--line); padding: 6px 12px; text-align: left; vertical-align: top; }}
  footer {{ color: var(--muted); font-size: 14px; margin-top: 48px; }}
</style>
</head>
<body>
<main>
{body}
<footer>Rendered from <a href="https://github.com/prteek/easy_sm/blob/main/.github/README.md">README.md</a>
· <a href="https://github.com/prteek/easy_sm">GitHub</a> · <a href="https://pypi.org/project/easy-sm/">PyPI</a></footer>
</main>
</body>
</html>
"""


def render(readme: str) -> str:
    body = markdown.markdown(readme, extensions=["tables", "fenced_code", "md_in_html", "toc"])
    first_line = readme.strip().splitlines()[0].lstrip("# ").strip()
    description = next((line for line in readme.splitlines()[1:] if line.strip()), "")
    return PAGE.format(title=html.escape(first_line), description=html.escape(description), body=body)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(README.read_text()))
    print(out / "index.html")


if __name__ == "__main__":
    main()
