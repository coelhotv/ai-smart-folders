# AI Smart Folders

AI Smart Folders is a Python-based assistant that scans an inbox directory, summarizes files with optional AI-assisted metadata, and safely routes them into an organized destination structure.

## Features
- Loads configuration overrides from `config.yaml`, falling back to sane defaults for the inbox, output directory, model name, ignored folders, and worker count.
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
3. Populate `config.yaml` to override defaults, for example:

```yaml
inbox_dir: ~/Dropbox/_courses
organized_dir: ~/OrganizedFiles
ollama_model: gemma3:270m
ignore_dirs:
  - _Unprocessed
  - _Ignored
max_workers: 1
```

4. Make sure `OLLAMA_API_URL` (if using Ollama) and any credentials for optional services are configured in your environment.

## Usage

```bash
python revised_new.py
```

The script writes agent and API logs to `agent.log` and `api.log`.

## Logging
- `agent.log`: timestamped events from the FileAgent logger.
- `api.log`: captures 1:1 request/response pairs when AI APIs are used.

## Contribution
Contributions are welcome. Please open issues for bugs or feature ideas, and feel free to submit pull requests following existing styling and logging patterns.

## License
This project is open-source under the MIT License. See `LICENSE` for details.
