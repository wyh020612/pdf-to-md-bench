# pdf-to-md-bench

> 6 个 PDF→Markdown 工具 × 5 份真实 PDF = 30 组实测,**评分细则全部公开**。
> 视频:**[https://www.bilibili.com/video/BV17j96BGEZy](https://www.bilibili.com/video/BV17j96BGEZy)**

## 工具阵容

| 工具 | 类型 | 安装 |
|------|------|------|
| [MarkItDown](https://github.com/microsoft/markitdown) | 微软 · 通用转换器 | `pip install 'markitdown[pdf]'` |
| [MinerU](https://github.com/opendatalab/MinerU) | 上海 AI Lab · pipeline + VLM | `pip install mineru` |
| [Docling](https://github.com/docling-project/docling) | IBM · 全能解析 | `pip install docling` |
| [Marker](https://github.com/datalab-to/marker) | Surya 模型 · 表格强 | `pip install marker-pdf` |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 百度 · OCR 引擎 | `pip install paddleocr 'paddlex[ocr]' paddlepaddle` |
| [opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf) | Java · AI-ready 解析 | `pip install opendataloader-pdf` (需 Java 21+) |

## 测试集 (5 份真实 PDF)

| # | 类型 | 来源 | 页数 |
|---|------|------|------|
| ① 双栏论文 | arXiv 论文 | "Attention Is All You Need" (1706.03762) | 8 |
| ② 财报 / 长文 | Berkshire Hathaway 2023 年信 | brk 官网 | 8 |
| ③ 扫描件 | NASA 1965 工程报告 | NTRS 公开档案 | 6 |
| ④ 幻灯片 | CS224N Lecture 8 · Transformers | Stanford 公开课 | 12 |
| ⑤ 图文长文 | Wikipedia: Photography | wiki REST PDF | 8 |

URL + sha256 全在 [`assets/test_pdfs/manifest.json`](assets/test_pdfs/manifest.json)。
注意 5 份**都有文本层**(NASA 是 NTRS 自动 OCR 过的),所以纯扫描赛道没充分覆盖 — 见下文 caveat。

## 最终评分(满分 25)

| 工具 | 总分 | 平均吞吐 | 备注 |
|------|------|----------|------|
| **opendataloader-pdf** | **21.4** | 6.7 pps | 5/5 类场景拿冠军(text-layer PDFs 前提下) |
| MarkItDown | 17.9 | **30.4 pps** | 速度全场最快,准确度也很高 |
| Docling | 17.8 | 3.4 pps | 全能稳健 · 二号位 |
| Marker | 15.3 | 0.14 pps | 准确度高 · MPS bug 强制 CPU,慢 |
| MinerU | 15.2 | 0.28 pps | pipeline 后端 5/5 全过 · 模型重 |
| PaddleOCR | 12.1 | 0.05 pps | 纯 OCR 模式 · 字流够准但版式无视 |

## 评分维度(0–5 分,满分 25)

| 维度 | 方法 |
|------|------|
| 文本准确 | 与 `pdftotext -layout` 字符总量比 |
| 表格还原 | markdown 中 `|` 单元格数 vs 基线 |
| 图片处理 | `![]()` 数量 vs `pdfimages -list` 计数 |
| 版式顺序 | 前 1500 字与基线的 `SequenceMatcher.ratio()` |
| 速度 / 成本 | pages-per-second 线性归一 (5 = ≥1.5 pps free local) |

完整规则在 [`bench/score_and_sync.py`](bench/score_and_sync.py)。

## 快速复现

```bash
# 0. Python 3.12 + uv
uv venv .venv && source .venv/bin/activate

# 1. 装工具(部分会拉模型,首次需 ~5 GB)
uv pip install 'markitdown[pdf]' opendataloader-pdf docling marker-pdf mineru paddleocr 'paddlex[ocr]' paddlepaddle albumentations

# 2. 拿 PDF(URL 在 manifest.json)
mkdir -p assets/test_pdfs && cd assets/test_pdfs
# … 下 5 份 PDF,确保 sha256 与 manifest 对得上

# 3. 跑 bench(Marker 必须 CPU 模式 - MPS Surya 有 bug)
python bench/run_bench.py
TORCH_DEVICE=cpu PYTORCH_ENABLE_MPS_FALLBACK=1 python bench/run_remaining.py

# 4. 打分 + 出表
python bench/score_and_sync.py
cat bench/scores.json | jq '.totals'
```

## 已知 caveat

1. **所有样本都有文本层** — opendataloader-pdf 这个赛道占便宜,真扫描场景 (`pdftotext` 出空白) 还没测。下一期 VLM 直读会拿真扫描重测。
2. **Marker 在 Apple Silicon MPS 上崩** — Surya 0.17 的 `unpack_qkv_with_mask` 在 MPS 上索引越界 (`torch.AcceleratorError`)。本仓库强制 CPU 跑出真实数据,但生产环境上想用 Marker 建议跑在 NVIDIA GPU。
3. **PaddleOCR 用纯 OCR 模式** — `PPStructureV3` 在 16 GB 内存下被 OOM 杀,所以本期是 `PaddleOCR(...).predict()` 的纯 OCR 文本提取,不是它的结构化模式。结构化模式留给后续测。
4. **打分维度有简化** — `image_handling` 用 `pdfimages` 计数 vs markdown 图片标记数,这对扫描件(图就是页本身)不公平,所以 image score 不要单独看。

## 数据来源 / 复现性

- **bench 输出**: [`bench/results.json`](bench/results.json) — 30 条 wall_seconds + char_count
- **打分矩阵**: [`bench/scores.json`](bench/scores.json) — 30 条 5-维度分数 + totals
- **每工具实际 markdown**: [`assets/results/<tool>/`](assets/results/) — 5 份 PDF 的真实输出全在这,自己跑出来的不一样欢迎开 issue
- **测试 PDF 指纹**: [`assets/test_pdfs/manifest.json`](assets/test_pdfs/manifest.json) — URL + sha256

## 反馈 / 提名

- 想看下一期测某个工具? 开 [issue](https://github.com/wyh020612/pdf-to-md-bench/issues),贴上工具链接 + 一句你为什么觉得它值得测。
- 觉得评分规则不公平? 改 `bench/score_and_sync.py` 提 PR,聊聊更合适的算法。

## License

代码 MIT。测试 PDF 各自来源版权归原作者(我们只放下载链接 + sha256,不重发文件)。
