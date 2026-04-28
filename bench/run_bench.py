#!/usr/bin/env python3
"""
PDF → Markdown 6-tool benchmark
Runs each tool against each test PDF, records wall time, output size, and writes Markdown.
Outputs raw JSON to bench/results.json for downstream scoring.
"""
from __future__ import annotations
import os, sys, json, time, traceback, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "assets" / "test_pdfs"
RESULTS_DIR = ROOT / "assets" / "results"
RESULTS_JSON = ROOT / "bench" / "results.json"

SAMPLES = [
    ("arxiv_doublecol",  "sample1_arxiv_doublecol.pdf",   8),
    ("brk_letter",       "sample2_brk_letter.pdf",         8),
    ("nasa_scan",        "sample3_nasa_scan.pdf",          6),
    ("cs224n_slides",    "sample4_cs224n_slides.pdf",     12),
    ("wiki_long",        "sample5_wiki.pdf",               8),
]

# ---------------- Per-tool runners ----------------
# Each runner returns (markdown:str)

def run_markitdown(pdf: Path) -> str:
    from markitdown import MarkItDown
    md = MarkItDown()
    return md.convert(str(pdf)).text_content

_docling_converter = None
def run_docling(pdf: Path) -> str:
    global _docling_converter
    from docling.document_converter import DocumentConverter
    if _docling_converter is None:
        _docling_converter = DocumentConverter()
    result = _docling_converter.convert(str(pdf))
    return result.document.export_to_markdown()

_marker_converter = None
def run_marker(pdf: Path) -> str:
    global _marker_converter
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    if _marker_converter is None:
        _marker_converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = _marker_converter(str(pdf))
    text, _, _ = text_from_rendered(rendered)
    return text

def run_mineru(pdf: Path) -> str:
    """Use the mineru CLI; pipeline backend, auto method."""
    out_root = RESULTS_DIR / "mineru"
    out_root.mkdir(parents=True, exist_ok=True)
    # Clean to avoid stale files
    sample_dir = out_root / pdf.stem
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    cmd = [
        "mineru", "-p", str(pdf), "-o", str(out_root),
        "-b", "pipeline", "-m", "auto", "-l", "en",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"mineru exit {proc.returncode}: {proc.stderr[-500:]}")
    md_file = sample_dir / "auto" / f"{pdf.stem}.md"
    if not md_file.exists():
        raise FileNotFoundError(f"mineru output missing: {md_file}")
    return md_file.read_text()

_paddle_ocr = None
def run_paddleocr(pdf: Path) -> str:
    """Light OCR mode - line-by-line text extraction."""
    global _paddle_ocr
    from paddleocr import PaddleOCR
    if _paddle_ocr is None:
        _paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
    results = _paddle_ocr.predict(str(pdf))
    parts: list[str] = []
    for r in results:
        parts.extend(r["rec_texts"])
    return "\n".join(parts)

def run_opendataloader(pdf: Path) -> str:
    """Wrapper around the Java jar; outputs to a folder."""
    import opendataloader_pdf
    out_dir = RESULTS_DIR / "opendataloader" / pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    opendataloader_pdf.convert(
        str(pdf),
        output_dir=str(out_dir),
        format="markdown",
        quiet=True,
    )
    md_file = out_dir / f"{pdf.stem}.md"
    if not md_file.exists():
        # The CLI sometimes drops it as <stem>.pdf.md or similar — fall back
        cands = list(out_dir.glob("*.md"))
        if not cands:
            raise FileNotFoundError(f"opendataloader output missing in {out_dir}")
        md_file = cands[0]
    return md_file.read_text()

TOOLS = [
    ("markitdown",          run_markitdown,         False),
    ("opendataloader-pdf",  run_opendataloader,     False),
    ("docling",             run_docling,            False),
    ("mineru",              run_mineru,             False),
    ("marker",              run_marker,             False),
    ("paddleocr",           run_paddleocr,          False),  # heaviest last
]

# ---------------- Driver ----------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for tool_name, fn, _ in TOOLS:
        out_dir = RESULTS_DIR / tool_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for sample_id, sample_file, pages in SAMPLES:
            pdf_path = PDF_DIR / sample_file
            label = f"{tool_name:22s} | {sample_id:18s}"
            print(f"[run]   {label} ...", flush=True)
            t0 = time.time()
            err = None
            md = ""
            try:
                md = fn(pdf_path)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            dt = time.time() - t0

            # Persist markdown (except mineru/opendataloader which already wrote)
            md_path = None
            if tool_name not in ("mineru", "opendataloader-pdf"):
                md_path = out_dir / f"{sample_file.replace('.pdf', '.md')}"
                if md:
                    md_path.write_text(md)

            char_count = len(md)
            line_count = md.count("\n") + 1 if md else 0
            pages_per_sec = pages / dt if dt > 0 else 0.0
            print(
                f"[done]  {label} | {dt:7.2f}s | {char_count:7d} chars | "
                f"{pages_per_sec:5.2f} pps | err={err}",
                flush=True,
            )

            records.append({
                "tool": tool_name,
                "sample": sample_id,
                "sample_file": sample_file,
                "pages": pages,
                "wall_seconds": round(dt, 3),
                "pages_per_second": round(pages_per_sec, 4),
                "char_count": char_count,
                "line_count": line_count,
                "error": err,
            })

            # Persist after every record so partial results survive crashes
            RESULTS_JSON.write_text(json.dumps({
                "schema_version": 1,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "records": records,
            }, indent=2))

    print(f"\nWrote {len(records)} records to {RESULTS_JSON}")

if __name__ == "__main__":
    main()
