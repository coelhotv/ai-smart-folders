from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple


CONSOLE_INFO_PREFIXES = (
    "=",
    "Inbox:",
    "Output:",
    "Models:",
    "Success:",
    "Move failed",
    "Total files:",
    "Total time:",
    "Post-run stats:",
    "No files found",
    "Run",
)


class TerminalFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return any(msg.startswith(prefix) for prefix in CONSOLE_INFO_PREFIXES)


def setup_logging(data_root: Path) -> Tuple[logging.Logger, logging.Logger]:
    data_root.mkdir(parents=True, exist_ok=True)

    agent_logger = logging.getLogger("FileAgent")
    agent_logger.setLevel(logging.INFO)
    agent_logger.propagate = False

    api_logger = logging.getLogger("OllamaAPI")
    api_logger.setLevel(logging.INFO)
    api_logger.propagate = False

    for logger_obj in (agent_logger, api_logger):
        logger_obj.handlers = []

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s", "%Y-%m-%d %H:%M:%S")
    api_fmt = logging.Formatter("%(asctime)s --- %(message)s", "%Y-%m-%d %H:%M:%S")

    agent_file = logging.FileHandler(data_root / "agent.log", encoding="utf-8")
    agent_file.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(TerminalFilter())

    api_file = logging.FileHandler(data_root / "api.log", encoding="utf-8")
    api_file.setFormatter(api_fmt)

    agent_logger.addHandler(agent_file)
    agent_logger.addHandler(console)
    api_logger.addHandler(api_file)
    return agent_logger, api_logger
