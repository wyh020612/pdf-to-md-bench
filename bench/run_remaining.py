#!/usr/bin/env python3
"""
Continuation bench: re-run only marker (CPU mode, MPS broken) and paddleocr.
Reuses the records list in results.json — appends fresh records and rewrites.
"""
from __future__ import annotations
import json, time, os, traceback
from pathlib import Path

# Force CPU before any torch import — avoids the MPS Surya kernel bug.
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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

_paddle_ocr = None
def run_paddleocr(pdf: Path) -> str:
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
    parts = []
    for r in results:
        parts.extend(r["rec_texts"])
    return "\n".join(parts)

TOOLS = [
    ("marker",     run_marker),
    ("paddleocr",  run_paddleocr),
]

def main():
    # Load existing results, drop any prior records for tools we are about to redo
    data = json.loads(RESULTS_JSON.read_text())
    records = [r for r in data["records"] if r["tool"] not in {t for t, _ in TOOLS}]
    print(f"keeping {len(records)} prior records, redoing {[t for t,_ in TOOLS]}", flush=True)

    for tool_name, fn in TOOLS:
        out_dir = RESULTS_DIR / tool_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, sample_file, pages in SAMPLES:
            pdf_path = PDF_DIR / sample_file
            label = f"{tool_name:14s} | {sample_id:18s}"
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

            md_path = out_dir / f"{sample_file.replace('.pdf', '.md')}"
            if md:
                md_path.write_text(md)
            char_count = len(md)
            line_count = md.count("\n") + 1 if md else 0
            pps = pages / dt if dt > 0 else 0.0
            print(
                f"[done]  {label} | {dt:7.2f}s | {char_count:7d} chars | "
                f"{pps:5.2f} pps | err={err}",
                flush=True,
            )
            records.append({
                "tool": tool_name,
                "sample": sample_id,
                "sample_file": sample_file,
                "pages": pages,
                "wall_seconds": round(dt, 3),
                "pages_per_second": round(pps, 4),
                "char_count": char_count,
                "line_count": line_count,
                "error": err,
            })

            RESULTS_JSON.write_text(json.dumps({
                "schema_version": 1,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "records": records,
            }, indent=2))

    print(f"\nDone. Total records: {len(records)}")

if __name__ == "__main__":
    main()
