# AI Smart Folders

AI Smart Folders is my first attempt to play with local AI models and LLMs functions. 
It's a Python-based assistant that scans an inbox directory, summarizes files with optional AI-assisted metadata, and safely routes them into an organized destination structure.

## Features
- Loads configuration overrides from `data_dir/config.yaml` inside your runtime directory (default `~/.ai-smart-folders-data/config.yaml`), or from the path set in `AI_SMART_CONFIG_PATH`, falling back to sane defaults for the inbox, output directory, model name, ignored folders, and worker count.
- Normalizes logging across console, agent, and API outputs so automated runs stay silent while still capturing warnings and errors.
- Integrates with optional libraries such as `ollama`, `pypdf`, `python-docx`, `pytesseract`, `Pillow`, and `python-pptx` to harvest metadata from a wide variety of document types.
- Handles file hashing, caching, threading, and SQLite bookkeeping to avoid reprocessing files and to provide consistent move semantics.

## Requirements
- Python 3.10 or newer.
- Optional libraries surfaced in the script:
  - `ollama`
  - `pypdf`
  - `python-docx`
  - `pytesseract`
  - `Pillow`
  - `python-pptx`
- `tesseract` binary on the system if `pytesseract` is used.

Install dependencies with:

```bash
python -m pip install ollama pypdf python-docx pytesseract Pillow python-pptx
```

## Setup
1. Clone the repository and enter the directory.
2. (Optional) Create and activate a virtual environment.
3. Copy `config.example.yaml` into your data directory (default `~/.ai-smart-folders-data/config.yaml`) or another location you intend to use for runtime files, and edit it there. The sample already lists `inbox_dir`, `organized_dir`, `ollama_model`, `ignore_dirs`, `max_content_length`, `max_workers`, and the `data_dir` value that controls where runtime artifacts land. Make sure the `data_dir` entry matches the directory that now holds this config file.
4. Optionally set `AI_SMART_CONFIG_PATH` to the explicit config file location (if not using the default `data_dir/config.yaml`) and `AI_SMART_DATA_DIR` to the secured data folder before running the script.
5. Make sure `OLLAMA_API_URL` (if using Ollama) and any credentials for optional services are configured in your environment.

## Usage

Run the production-ready agent with logging, caching, and SQLite bookkeeping:

```bash
python smart-folders_v2.py
```

There is also a lightweight, single-threaded helper in `smart-folders_v1.py` that can be used for experimentation:

```bash
python smart-folders_v1.py
```

## Data & Logging
- Runtime data (agent logs, API logs, the cache, and the SQLite DB) lives inside the configured `data_dir` (default `~/.ai-smart-folders-data`). That directory is excluded from the repository, so public access never exposes processed files or personal metadata.
- Set `data_dir` via the config or override it with `AI_SMART_DATA_DIR`. Existing artifacts such as `agent.log`, `api.log`, `file_cache.pkl`, and `file_organization.db` can be moved into that directory (`mkdir -p ~/.ai-smart-folders-data && mv agent.log api.log file_cache.pkl file_organization.db ~/.ai-smart-folders-data/`) before running the script.
- `agent.log`: timestamped events from the FileAgent logger.
- `api.log`: captures 1:1 request/response pairs when AI APIs are used.

## Contribution
Contributions are welcome. Please open issues for bugs or feature ideas, and feel free to submit pull requests following existing styling and logging patterns.

## License
This project is open-source under the MIT License. See `LICENSE` for details.
