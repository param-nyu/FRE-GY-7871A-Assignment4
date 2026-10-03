"""Assemble the 5-6 page PDF report.

Pipeline: run the analysis, build the figures, render an HTML document with
print CSS, then convert with headless Chrome. Chrome is used because no PDF
toolchain (pandoc, weasyprint, wkhtmltopdf) is installed on the target machine
while Chrome is, and its print engine gives better typography than any of the
pure-Python alternatives anyway.

Usage::

    python build_report.py            # full build
    python build_report.py --no-pdf   # HTML only, no Chrome
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src import config, exhibits  # noqa: E402
from src.analysis import run_all  # noqa: E402

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)

CSS = """
@page { size: Letter; margin: 0.55in 0.68in; }
* { box-sizing: border-box; }
body {
  font-family: "Source Serif Pro", Georgia, "Times New Roman", serif;
  font-size: 9.3pt; line-height: 1.33; color: #16181d; margin: 0;
}
h1 { font-size: 15.5pt; line-height: 1.2; margin: 0 0 2pt; letter-spacing: -0.2pt; }
h2 {
  font-size: 11pt; margin: 11pt 0 4pt;
  border-bottom: 0.8pt solid #c9ccd2; padding-bottom: 2pt;
}
h3 { font-size: 9.6pt; margin: 8pt 0 2pt; font-style: italic; font-weight: 600; }
p { margin: 0 0 5.5pt; text-align: justify; hyphens: auto; }
.subtitle { font-size: 10.5pt; color: #4a4f58; margin-bottom: 1pt; }
.byline { font-size: 8.5pt; color: #6a6f78; margin-bottom: 9pt; }
.abstract {
  background: #f5f6f8; border-left: 2.5pt solid #2b6cb0;
  padding: 7pt 10pt; margin: 0 0 10pt; font-size: 9pt;
}
.abstract p { margin: 0 0 5pt; }
.abstract p:last-child { margin: 0; }
table {
  border-collapse: collapse; width: 100%;
  font-size: 7.8pt; margin: 4pt 0 6pt;
  font-family: "Helvetica Neue", Arial, sans-serif;
}
th, td { padding: 2.2pt 4pt; text-align: right; }
th {
  border-bottom: 0.8pt solid #16181d; border-top: 0.8pt solid #16181d;
  font-weight: 600; white-space: nowrap;
}
td:first-child, th:first-child { text-align: left; }
td:nth-child(2), th:nth-child(2) { text-align: left; }
tbody tr:last-child td { border-bottom: 0.8pt solid #16181d; }
tbody tr:nth-child(even) { background: #fafbfc; }
.table-title { font-size: 8.6pt; font-weight: 700; margin: 10pt 0 1pt; }
.notes {
  font-size: 7.2pt; color: #52575f; line-height: 1.35;
  margin: 1pt 0 9pt; text-align: justify;
}
figure { margin: 6pt 0 7pt; page-break-inside: avoid; text-align: center; }
figure img { width: 100%; max-width: 100%; }
figcaption { font-size: 7.4pt; color: #52575f; margin-top: 2pt; text-align: left; }
.callout {
  border-left: 2.5pt solid #c1443c; background: #fdf6f5;
  padding: 6pt 9pt; margin: 6pt 0; font-size: 8.7pt;
}
.callout strong { color: #99322c; }
ul, ol { margin: 0 0 7pt; padding-left: 15pt; }
li { margin-bottom: 1.8pt; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.4pt;
       background: #f0f1f3; padding: 0.5pt 2.5pt; border-radius: 2pt; }
.page-break { page-break-before: always; }
.footer { font-size: 7.2pt; color: #83888f; margin-top: 14pt;
          border-top: 0.5pt solid #d4d7dc; padding-top: 4pt; }
"""


def _img(path: Path) -> str:
    """Embed a PNG as a data URI so the HTML is a single self-contained file."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _table_html(frame: pd.DataFrame, decimals: int = 3) -> str:
    """Render a table for the PDF using the exhibit formatter.

    This deliberately delegates to :func:`table_generator._fmt` rather than
    keeping its own rules. An earlier version duplicated the formatting logic
    and immediately drifted -- the PDF printed "62.000 outlets" while the
    Markdown exhibit printed "62".
    """
    from src.table_generator import _fmt

    formatted = frame.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].map(lambda v: _fmt(v, decimals))
    return formatted.to_html(index=False, border=0, escape=False)


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return shutil.which("chromium") or shutil.which("google-chrome")


def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Render with headless Chrome. Returns False if Chrome is unavailable."""
    chrome = find_chrome()
    if not chrome:
        print("[report] No Chrome/Chromium found; skipping PDF. HTML is at "
              f"{html_path}")
        return False

    result = subprocess.run(
        [
            chrome, "--headless", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if not pdf_path.exists():
        print(f"[report] Chrome failed to write a PDF:\n{result.stderr[:500]}")
        return False
    return True


def build_html(results: dict, figures: dict[str, Path]) -> str:
    """Render the full report body. Imported by the notebook too."""
    from src.report_text import render_sections

    sections = render_sections(results, figures, _img, _table_html)
    today = dt.date.today().strftime("%d %B %Y")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>AI Risk Sentiment and the AI Equity Complex</title>
<style>{CSS}</style></head>
<body>
<h1>Tracking Sentiment Around AI Risk and Its Effect on AI Equities</h1>
<div class="subtitle">An event study around the 8 September 2026
Anthropic existential-risk disclosure</div>
<div class="byline">FRE-GY 7871A &middot; Assignment 4 &middot; {today}</div>
{sections}
<div class="footer">
Data: twitterapi.io advanced search (X), Google News RSS (headlines),
Alpha Vantage (OHLCV), FRED (DGS3MO). Sentiment: VADER with a finance and
AI-risk lexicon patch for social text, Loughran-McDonald for news.
All code, tests and exhibits accompany this report in the repository.
</div>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true", help="HTML only")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[report] running analysis ...")
    results = run_all(output_dir)

    print("[report] building figures ...")
    figures = exhibits.build_all(results, output_dir)

    print("[report] rendering HTML ...")
    html_path = output_dir / "Assignment4_Report.html"
    html_path.write_text(build_html(results, figures))

    if not args.no_pdf:
        pdf_path = output_dir / "Assignment4_Report.pdf"
        if html_to_pdf(html_path, pdf_path):
            size_kb = pdf_path.stat().st_size / 1024
            print(f"[report] wrote {pdf_path} ({size_kb:.0f} KB)")

    print(f"[report] HTML at {html_path}")


if __name__ == "__main__":
    main()
