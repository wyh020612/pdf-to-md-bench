#!/usr/bin/env python3
"""
Extract real Markdown snippets from each tool's first-PDF output and the
pdftotext baseline. Writes a typed TS module to remotion/src/snippets.ts.

Snippets are short (~600 chars) and trimmed at sentence boundaries when
possible, so they slot cleanly into the side-by-side comparison view.
"""
from __future__ import annotations
import json, re, subprocess, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "assets" / "test_pdfs"
RESULTS_DIR = ROOT / "assets" / "results"
SNIPPETS_TS = ROOT / "remotion" / "src" / "snippets.ts"

SAMPLES = [
    ("arxiv_doublecol",  "sample1_arxiv_doublecol.pdf"),
    ("brk_letter",       "sample2_brk_letter.pdf"),
    ("nasa_scan",        "sample3_nasa_scan.pdf"),
    ("cs224n_slides",    "sample4_cs224n_slides.pdf"),
    ("wiki_long",        "sample5_wiki.pdf"),
]

TOOLS = ["markitdown", "opendataloader-pdf", "docling", "marker", "mineru", "paddleocr"]


def md_path_for(tool: str, sample_file: str) -> Path:
    stem = sample_file.replace(".pdf", "")
    if tool == "mineru":
        return RESULTS_DIR / "mineru" / stem / "auto" / f"{stem}.md"
    if tool == "opendataloader-pdf":
        return RESULTS_DIR / "opendataloader" / stem / f"{stem}.md"
    return RESULTS_DIR / tool / f"{stem}.md"


MAX_CHARS = 520
MAX_LINES = 14


def trim(text: str) -> str:
    """Trim to ~MAX_CHARS at line/sentence boundary, normalize whitespace."""
    # Drop leading blanks
    text = text.lstrip("\n﻿ ").rstrip()
    # If markdown looks like one giant blob, force-wrap
    if "\n" not in text[:200]:
        text = textwrap.fill(text, width=80)
    lines = text.splitlines()[:MAX_LINES]
    out = "\n".join(lines)
    if len(out) > MAX_CHARS:
        out = out[:MAX_CHARS]
        # Snap to last newline / sentence end to avoid mid-word cut
        for cut in (out.rfind("\n"), out.rfind("。"), out.rfind("."), out.rfind(", ")):
            if cut > MAX_CHARS - 120:
                out = out[:cut]
                break
    return out + ("\n…" if len(text) > len(out) else "")


def baseline(pdf: Path) -> str:
    """pdftotext output as the 'truth' to compare against."""
    p = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True, text=True, check=True,
    )
    return trim(p.stdout)


def js_str(s: str) -> str:
    """Encode a string as a JS template literal with backticks safe-escaped."""
    return "`" + s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${") + "`"


def main():
    out: dict[str, dict[str, str]] = {}
    for sample_id, sample_file in SAMPLES:
        pdf = PDF_DIR / sample_file
        out[sample_id] = {"baseline": baseline(pdf)}
        for tool in TOOLS:
            mp = md_path_for(tool, sample_file)
            if mp.exists() and mp.stat().st_size > 0:
                out[sample_id][tool] = trim(mp.read_text(errors="ignore"))
            else:
                out[sample_id][tool] = "(没有输出 / 工具失败)"

    lines = [
        "/**",
        " * Real markdown snippets extracted by bench/extract_snippets.py.",
        " * Each entry is the first ~520 chars of the tool's output for that sample,",
        " * trimmed at line boundary and ending in `…` if more content follows.",
        " * `baseline` is `pdftotext -layout` of the same PDF — used as the truth column.",
        " */",
        "import type { SampleId, ToolId } from './data';",
        "",
        "export type SnippetSet = Record<ToolId | 'baseline', string>;",
        "",
        "export const SNIPPETS: Record<SampleId, SnippetSet> = {",
    ]
    for sample_id, by_tool in out.items():
        lines.append(f"  {sample_id}: {{")
        for k, v in by_tool.items():
            key = f'"{k}"' if "-" in k or k == "baseline" else k
            lines.append(f"    {key}: {js_str(v)},")
        lines.append("  },")
    lines.append("};")

    SNIPPETS_TS.write_text("\n".join(lines))
    print(f"Wrote {SNIPPETS_TS}")
    for sid in out:
        for tool, snip in out[sid].items():
            print(f"  {sid:18s} {tool:22s} {len(snip):4d} chars")


if __name__ == "__main__":
    main()
