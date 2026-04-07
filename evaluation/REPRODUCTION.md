# OCR Benchmark Reproduction Guide

This guide documents the process of benchmarking local OCR models using the `ai-smart-folders` evaluation framework.

## 1. Setup

### Local Models
The benchmarks require the following models installed via Ollama:
- `deepseek-ocr:3b`
- `Maternion/NuMarkdown-Thinking:8b`
- Support models (LLMs for classification/understanding): `qwen3.5:4b`, `granite4:1b-h`

```bash
ollama pull deepseek-ocr:3b
ollama pull Maternion/NuMarkdown-Thinking:8b
# Ensure support models are also available
```

### Configuration Files
A dedicated YAML configuration file should exist for each OCR model being tested. These files should be located in the root directory:
- `test-deepseek.yaml`
- `test-numarkdown.yaml`

Example configuration (`test-numarkdown.yaml`):
```yaml
inbox_dir: ./tmp_inbox
organized_dir: ./tmp_output
data_dir: ./.ai-smart-folders-data
max_workers: 2
prompt_version: v3

models:
  router_model: granite4:1b-h
  understanding_model: qwen3.5:4b
  classification_model: qwen3.5:4b
  fallback_model: qwen3.5:9b-q4_K_M
  ocr_model: Maternion/NuMarkdown-Thinking:8b

taxonomy:
  level1_default: General
  level2_default: Unsorted
```

## 2. Dataset
The benchmark uses a JSONL dataset containing paths to sample files and their expected classification results.
- **Location:** `evaluation/ocr_benchmark_folders.jsonl`
- **Samples:** Located in `tmp_eval_samples/`

## 3. Running the Benchmark

A specialized runner script is used to execute the benchmark and persist results into timestamped files.

```bash
# Run for DeepSeek
python evaluation/run_benchmark.py --configs test-deepseek.yaml

# Run for NuMarkdown
python evaluation/run_benchmark.py --configs test-numarkdown.yaml

# Run both for comparison (generates a comparison JSON)
python evaluation/run_benchmark.py --configs test-deepseek.yaml test-numarkdown.yaml
```

**Important:** Before running a fresh benchmark comparison, it is recommended to clear the classification cache to ensure models extract text from scratch:
```bash
rm .ai-smart-folders-data/file_cache.pkl
```

## 4. Results
Results are saved as JSON files in the `evaluation/results/` directory with the naming convention:
`{TIMESTAMP}_{MODEL_NAME}.json`

If multiple configs are passed, a `{TIMESTAMP}_comparison.json` is also generated.

## 5. Metrics Tracked
- **Full Matches:** Number of cases where Level 1, Level 2, and Needs Review flags matched perfectly.
- **Matched Level 1 / Level 2:** Accuracy of classification at each level.
- **Needs Review:** Accuracy of the confidence thresholding.
- **Execution Time:** Wall clock time for processing.

---
*Created on: 2026-04-06*
