# Identifying LLM Performance Gaps Across Languages

A comprehensive multilingual benchmark evaluating **OLMo-2-7B-Instruct** across ~100 languages and 10 datasets, spanning Math, Commonsense, NLI, Healthcare QA, and General Knowledge. This project quantifies the relationship between pre-training token exposure and downstream performance, and tests whether English Chain-of-Thought prompting can bridge multilingual gaps.

---

## Key Findings

- English consistently outperforms all other languages across every domain and metric
- Pre-training token exposure has a **Spearman correlation of 0.85** with downstream accuracy
- English CoT improves high/mid-resource languages (French, Spanish, Chinese) but provides little benefit to low-resource ones (Hindi, Korean, Tibetan)
- Translation noise is **not** the primary cause of performance drops — model limitations are
- Hunyuan-MT produces more fluent translations (higher METEOR); NLLB is more literal (higher BLEU/ROUGE-L)

---

## Project Structure

```
llm-multilingual-benchmark/
│
├── notebooks/
│   ├── 01_translate_nllb.ipynb              # Translate datasets to 100 langs via NLLB-600M
│   ├── 02_translate_hunyuan.ipynb           # Translate 10 langs via Hunyuan-MT-7B
│   ├── 03_backtranslate_nllb.ipynb          # Back-translation for quality eval (NLLB)
│   ├── 04_translation_accuracy.ipynb        # BLEU, METEOR, ROUGE-L, BERTScore analysis
│   ├── 05_eval_translations.ipynb           # Downstream task eval on translated datasets
│   ├── 06_eval_mgsm_100lang.ipynb           # MGSM baseline eval across 100 languages
│   ├── 07_eval_svamp.ipynb                  # SVAMP dataset eval with self-consistency
│   ├── 08_english_cot.ipynb                 # English Chain-of-Thought experiments
│   └── 09_token_exposure_analysis.ipynb     # Pre/post-training token distribution analysis
│
├── data/
│   ├── mappings/
│   │   └── fasttext_nllb_mapping_98.csv     # FastText ↔ NLLB language code mapping
│   ├── results/
│   │   ├── EVALUATION_RESULTS.txt           # Raw accuracy scores (OLMo, Mistral) across 100 langs
│   │   ├── average_translation_scores_mgsm.csv   # Per-language translation quality (MGSM)
│   │   └── average_translation_scores_svamp.csv  # Per-language translation quality (SVAMP)
│   └── translated/                          # Large JSONL files — see note below
│       ├── mgsm_multicolumn_multilang_final.jsonl
│       ├── msvamp_multicolumn_multilang_final.jsonl
│       └── xnli_multicolumn_multilang.jsonl
│
├── prompts/
│   ├── translations_prompt_mgsm.json        # Per-language prompts for MGSM eval
│   ├── translations_prompt_msvamp.json      # Per-language prompts for SVAMP eval
│   └── translations.json                    # General multilingual instruction translations
│
├── requirements.txt
└── .gitignore
```

> **Note on large files:** `.jsonl` translation files (some >100MB) and `lid.176.bin` (FastText model, ~125MB) are excluded from git tracking. See the Data section below for how to reproduce them.

---

## Datasets Evaluated

| Domain | Dataset |
|---|---|
| Math | MGSM, MSVAMP |
| Commonsense | XCOPA, XCSQA, XCODAH |
| NLI | XNLI |
| Healthcare QA | HealthQA, LiveQA, MedicationQA |
| General Knowledge | MKQA, MMLU, M-ARC |

---

## Models Used

| Role | Model |
|---|---|
| Evaluation | `allenai/OLMo-2-1124-7B-Instruct` |
| Translation (100 langs) | `facebook/nllb-200-distilled-600M` |
| Translation (10 langs, quality) | `Hunyuan-MT-7B` (WMT25 winner) |
| Language Detection | `fastText lid.176.bin` |

---

## Methodology

### 1. Translation Pipeline
Datasets were translated from English to ~100 languages using NLLB-200-Distilled-600M. For 10 key languages, Hunyuan-MT-7B was used to benchmark translation quality via back-translation (BLEU, METEOR, ROUGE-L, BERTScore, COMET).

### 2. Model Evaluation
All inference was done via **vLLM** with self-consistency sampling:
- `temperature=0.7`, `top_k=40`, `n=20` independent runs per sample
- Final answer selected via **majority voting**

### 3. English Chain-of-Thought (CoT)
The model was prompted to reason internally in English while keeping input/output in the native language. Results were compared against standard native-language prompting.

### 4. Token Exposure Analysis
Language distribution was measured in OLMo's pre-training corpus (`dolmino-mix-1124`, 0.1% sample ≈ 843B tokens) and post-training data (TULU SFT + preference-mix + RLVR-math) using FastText with a 5-token sliding window. Spearman rank correlations were computed between token counts and downstream accuracy.

---

## Results Summary

| Language | Pre-train Tokens | Token % | Avg Accuracy |
|---|---|---|---|
| English | 174,624,114 | 94.57% | 64.10% |
| French | 1,381,649 | 0.75% | 44.40% |
| Spanish | 1,085,026 | 0.59% | 43.80% |
| Chinese | 245,423 | 0.13% | 36.50% |
| Hindi | 49,483 | 0.03% | 27.40% |
| Korean | 36,619 | 0.02% | 27.60% |

**Pre-training Spearman correlation: 0.942**  
**Post-training Spearman correlation: 0.771**

---

## Reproducing the Data

### Download FastText language ID model
```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

### Translate datasets (NLLB)
Run `notebooks/01_translate_nllb.ipynb` — requires a CUDA GPU and ~6GB VRAM.

### Run evaluation
Run `notebooks/05_eval_translations.ipynb` or `notebooks/06_eval_mgsm_100lang.ipynb` — requires vLLM and ~16GB VRAM for OLMo-2-7B.

---

## Installation

```bash
git clone https://github.com/your-org/llm-multilingual-benchmark.git
cd llm-multilingual-benchmark
pip install -r requirements.txt
```

---

## Contributors

| Contributor | Responsibilities |
|---|---|
| Amrit | Math eval, token distribution, self-consistency (MGSM/SVAMP), back-translation |
| Bhavan | Reasoning eval, self-consistency (MMLU/M-ARC), back-translation |
| Karthik | Commonsense eval, self-consistency (XCOPA/XCSQA/XCODAH), back-translation |
| Omkar | Healthcare eval, self-consistency (HealthcareQA) |

---

## Citation / References

- OLMo 2: [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)
- NLLB: [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)
- Hunyuan-MT: [arXiv:2509.05209](https://arxiv.org/abs/2509.05209)
- Self-Consistency CoT: [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- XCOPA: [EMNLP 2020](https://doi.org/10.18653/v1/2020.emnlp-main.185)
- Global MMLU: [arXiv:2412.03304](https://arxiv.org/abs/2412.03304)
