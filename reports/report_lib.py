"""Helpers for the final report: a print stylesheet, an HTML wrapper, tables, chart colours and rendering with
headless Chrome.

    uv run python reports/final_report.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = Path(__file__).resolve().parent / "final"
SERIES = {"tuned": "#2a78d6", "control": "#eb6834", "other": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

CSS = """
@page { size: Letter; margin: 0.8in 0.85in; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.42; color: #111;
       background: #fff; }
h1 { font-size: 18pt; margin: 0 0 2pt; font-weight: normal; }
.sub { color: #555; font-size: 9.5pt; margin-bottom: 14pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; font-weight: bold; break-after: avoid; page-break-after: avoid; }
p { margin: 4pt 0 7pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #ddd; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { border-bottom: 1.5px solid #888; font-weight: bold; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.box { border-left: 3px solid #2a78d6; background: #f5f7fa; padding: 6pt 10pt; margin: 6pt 0 10pt; }
.warn { border-left-color: #eb6834; background: #fbf5f1; }
.tag { font-size: 8pt; padding: 0 4pt; border: 1px solid #999; border-radius: 3px; color: #333; }
figure { margin: 8pt 0 12pt; page-break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-size: 8.8pt; color: #444; margin-top: 3pt; }
ul { margin: 3pt 0 8pt 16pt; padding: 0; } li { margin: 2pt 0; }
code { font-family: Menlo, monospace; font-size: 8.5pt; }
"""


def page(title: str, subtitle: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head>"
            f"<body><h1>{title}</h1><div class='sub'>{subtitle}</div>{body}</body></html>")


def table(header: list[str], rows: list[list], numeric: set[int] = frozenset()) -> str:
    th = "".join(f"<th class='{'n' if i in numeric else ''}'>{h}</th>" for i, h in enumerate(header))
    trs = "".join("<tr>" + "".join(f"<td class='{'n' if i in numeric else ''}'>{c}</td>" for i, c in enumerate(r))
                  + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{trs}</table>"


def render(html: str, stem: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / f"{stem}.html"
    src.write_text(html)
    pdf = OUT / f"{stem}.pdf"
    subprocess.run([CHROME, "--headless=new", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", str(src)],
                   check=True, capture_output=True)
    return pdf
