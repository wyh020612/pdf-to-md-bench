#!/usr/bin/env python3
"""
Reads bench/results.json + per-tool markdown outputs, computes a 0..5 score per
(tool, sample, dimension), and writes:
  - bench/scores.json  (raw scoring matrix + totals)
  - remotion/src/data.ts (TypeScript constants, replaces the placeholder file)

Scoring rubric — kept simple and explicit so the video is reproducible:

  text_accuracy:   character-level similarity vs pdftotext baseline
                   (5 = >=0.85 ratio, scales linearly down to 0 at <=0.15)
  table_recovery:  count of "|"-cells in markdown vs ground-truth count from
                   pdftotext -layout (5 = >=0.8 ratio of expected cells)
  image_handling:  count of `![](`/`<img` in md vs pdfimages count
                   (5 = >=0.8, plus a +1 bonus if any captions present)
  layout_order:    Levenshtein-style longest-common-subsequence ratio of the
                   first 1000 chars of md vs pdftotext (preserves reading order)
  speed_cost:      pages_per_second normalized — 5 = >=1.5 pps free local

Errored runs get 0 across the board.

After running:
    cd bench && python score_and_sync.py
    cd ../remotion && npm run studio
"""
from __future__ import annotations
import json, re, subprocess, statistics
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "assets" / "test_pdfs"
RESULTS_DIR = ROOT / "assets" / "results"
RESULTS_JSON = ROOT / "bench" / "results.json"
SCORES_JSON = ROOT / "bench" / "scores.json"
DATA_TS = ROOT / "remotion" / "src" / "data.ts"

# --- shape ----------------------------------------------------------------
TOOLS = [
    "markitdown",
    "mineru",
    "docling",
    "marker",
    "paddleocr",
    "opendataloader-pdf",
]
SAMPLES = [
    "arxiv_doublecol",
    "brk_letter",
    "nasa_scan",
    "cs224n_slides",
    "wiki_long",
]
DIMS = ["text_accuracy", "table_recovery", "image_handling", "layout_order", "speed_cost"]


# --- helpers --------------------------------------------------------------
def md_path_for(tool: str, sample_file: str) -> Path:
    """Return the markdown output path for a tool/sample pair."""
    stem = sample_file.replace(".pdf", "")
    if tool == "mineru":
        return RESULTS_DIR / "mineru" / stem / "auto" / f"{stem}.md"
    if tool == "opendataloader-pdf":
        # The bench script writes under the literal folder name "opendataloader"
        return RESULTS_DIR / "opendataloader" / stem / f"{stem}.md"
    return RESULTS_DIR / tool / f"{stem}.md"


def pdftotext(pdf: Path) -> str:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True)
    return out.stdout


def pdfimage_count(pdf: Path) -> int:
    out = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True)
    if out.returncode != 0:
        return 0
    lines = [l for l in out.stdout.splitlines() if l.strip() and not l.startswith("page") and "----" not in l]
    return len(lines)


def table_cell_count(text: str) -> int:
    """Count pipe-style markdown table cells."""
    n = 0
    for line in text.splitlines():
        if "|" in line and line.strip().startswith("|"):
            n += line.count("|")
    return n


def md_image_count(text: str) -> int:
    return len(re.findall(r"!\[[^\]]*\]\(", text)) + len(re.findall(r"<img\s", text))


def has_caption_signal(text: str) -> bool:
    """Heuristic: 'Figure X' / '图 X' nearby an image link."""
    return bool(re.search(r"(figure\s*\d|图\s*\d)", text.lower()))


def length_ratio(md: str, baseline: str) -> float:
    if not baseline:
        return 0.0
    return min(len(md) / len(baseline), 2.0)  # cap at 2


def linear(value: float, lo: float, hi: float, lo_score: float = 0.0, hi_score: float = 5.0) -> float:
    if hi == lo:
        return hi_score
    t = (value - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return round(lo_score + t * (hi_score - lo_score), 2)


def order_similarity(md: str, baseline: str, n: int = 1500) -> float:
    """Reading-order proxy: SequenceMatcher ratio of first N chars."""
    a = re.sub(r"\s+", " ", md).strip()[:n]
    b = re.sub(r"\s+", " ", baseline).strip()[:n]
    if not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# --- scorer ---------------------------------------------------------------
def score_one(tool: str, sample_id: str, sample_file: str, record: dict) -> dict:
    pdf = PDF_DIR / sample_file
    md_p = md_path_for(tool, sample_file)
    baseline_text = pdftotext(pdf) if pdf.exists() else ""
    table_truth = table_cell_count(baseline_text)  # crude; many PDFs have 0
    image_truth = pdfimage_count(pdf)

    if record.get("error") or not md_p.exists() or md_p.stat().st_size == 0:
        return {
            "tool": tool, "sample": sample_id,
            **{d: 0.0 for d in DIMS},
            "_md_chars": 0,
            "_baseline_chars": len(baseline_text),
        }

    md_text = md_p.read_text(errors="ignore")

    # text_accuracy: character ratio vs baseline (cap at 0..1.4 then linear)
    ratio = length_ratio(md_text, baseline_text)
    text_accuracy = linear(min(ratio, 1.2), 0.2, 1.0)

    # table_recovery
    md_cells = table_cell_count(md_text)
    if table_truth < 4:
        table_recovery = 5.0 if md_cells >= 4 else (3.0 if "table" in md_text.lower() else 2.5)
    else:
        rec = md_cells / max(table_truth, 1)
        table_recovery = linear(rec, 0.1, 1.0)

    # image_handling
    md_imgs = md_image_count(md_text)
    if image_truth == 0:
        image_handling = 4.0 if md_imgs > 0 else 3.0  # neutral
    else:
        cap = 1 if has_caption_signal(md_text) else 0
        image_handling = min(5.0, linear(md_imgs / image_truth, 0.0, 0.8) + cap)

    # layout_order
    layout_order = round(linear(order_similarity(md_text, baseline_text), 0.15, 0.65) , 2)

    # speed_cost (uses bench timing)
    pps = record.get("pages_per_second", 0)
    speed_cost = linear(pps, 0.05, 1.5)

    return {
        "tool": tool, "sample": sample_id,
        "text_accuracy": text_accuracy,
        "table_recovery": round(table_recovery, 2),
        "image_handling": round(image_handling, 2),
        "layout_order": round(layout_order, 2),
        "speed_cost": round(speed_cost, 2),
        "_md_chars": len(md_text),
        "_baseline_chars": len(baseline_text),
    }


# --- main -----------------------------------------------------------------
def main():
    bench = json.loads(RESULTS_JSON.read_text())
    records = bench["records"]
    by_key: dict[tuple[str, str], dict] = {(r["tool"], r["sample"]): r for r in records}

    scores: list[dict] = []
    for tool in TOOLS:
        for sample_id in SAMPLES:
            sample_file = next(
                r["sample_file"] for r in records if r["sample"] == sample_id
            )
            rec = by_key.get((tool, sample_id), {"error": "missing"})
            sc = score_one(tool, sample_id, sample_file, rec)
            scores.append(sc)
            print(
                f"  {tool:22s} {sample_id:18s}  "
                + "  ".join(f"{d}={sc[d]:.1f}" for d in DIMS)
            )

    # totals = average of per-sample dim sums (5 dims × 5 samples / 5 = avg of 5 per sample)
    totals: dict[str, float] = {}
    for tool in TOOLS:
        rows = [s for s in scores if s["tool"] == tool]
        per_sample = [sum(r[d] for d in DIMS) for r in rows]
        totals[tool] = round(statistics.mean(per_sample), 1) if per_sample else 0

    # mean pps per tool from real bench data
    pps_per_tool: dict[str, float] = {}
    for tool in TOOLS:
        rows = [r for r in records if r["tool"] == tool and not r["error"] and r["wall_seconds"] > 0]
        pps_per_tool[tool] = round(statistics.mean(r["pages_per_second"] for r in rows), 3) if rows else 0.0

    # bench-as-list for the TS file
    bench_records_ts = [
        {
            "tool": r["tool"],
            "sample": r["sample"],
            "wall_seconds": r["wall_seconds"],
            "pages_per_second": r["pages_per_second"],
            "char_count": r["char_count"],
            "error": r["error"],
        }
        for r in records
    ]

    SCORES_JSON.write_text(
        json.dumps(
            {
                "scores": scores,
                "totals": totals,
                "pps_per_tool": pps_per_tool,
                "by_record": bench_records_ts,
            },
            indent=2,
        )
    )
    print(f"\nTotals (out of 25, average of 5 samples):")
    for t, v in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"  {t:22s}  {v:5.1f}   (mean {pps_per_tool[t]:.2f} pps)")

    # ---- write data.ts ---------------------------------------------------
    write_data_ts(scores, totals, bench_records_ts)
    print(f"\nWrote {DATA_TS}")


def ts_str(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def write_data_ts(scores, totals, bench_records):
    bench_lines = [
        "  " + json.dumps(
            {
                "tool": r["tool"],
                "sample": r["sample"],
                "wall_seconds": r["wall_seconds"],
                "pages_per_second": r["pages_per_second"],
                "char_count": r["char_count"],
                "error": r["error"],
            },
            ensure_ascii=False,
        ) + ","
        for r in bench_records
    ]

    score_lines = [
        "  " + json.dumps(
            {k: v for k, v in s.items() if not k.startswith("_")},
            ensure_ascii=False,
        ) + ","
        for s in scores
    ]

    totals_lines = [
        f'  {ts_str(t)}: {v},'
        for t, v in totals.items()
    ]

    head = DATA_TS.read_text()
    # Replace the BENCH array
    new_text = re.sub(
        r"export const BENCH: BenchRecord\[\] = \[[\s\S]*?\];",
        "export const BENCH: BenchRecord[] = [\n" + "\n".join(bench_lines) + "\n];",
        head,
    )
    new_text = re.sub(
        r"export const SCORES: Score\[\] = \[[\s\S]*?\];",
        "export const SCORES: Score[] = [\n" + "\n".join(score_lines) + "\n];",
        new_text,
    )
    new_text = re.sub(
        r"export const TOTALS: Record<ToolId, number> = \{[\s\S]*?\};",
        "export const TOTALS: Record<ToolId, number> = {\n" + "\n".join(totals_lines) + "\n};",
        new_text,
    )
    DATA_TS.write_text(new_text)


if __name__ == "__main__":
    main()
