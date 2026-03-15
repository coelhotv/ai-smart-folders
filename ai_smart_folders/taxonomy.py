from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from .models import AppConfig


def sanitize_path_component(name: str, fallback: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", (name or "")).strip()
    safe = re.sub(r"\s+", " ", safe)
    return safe[:100] or fallback


def scan_existing_taxonomy(config: AppConfig) -> Dict[str, List[str]]:
    taxonomy: Dict[str, List[str]] = {}
    root = config.organized_dir
    if not root.exists():
        return taxonomy

    for level1_dir in root.iterdir():
        if not level1_dir.is_dir():
            continue
        if level1_dir.name in config.ignore_dirs or level1_dir.name in config.taxonomy.technical_folders:
            continue
        taxonomy[level1_dir.name] = []
        for level2_dir in level1_dir.iterdir():
            if level2_dir.is_dir():
                taxonomy[level1_dir.name].append(level2_dir.name)
    return taxonomy


def normalize_categories(config: AppConfig, category_l1: str | None, category_l2: str | None) -> Tuple[str, str]:
    global_aliases = {key.lower(): value for key, value in config.taxonomy.aliases.items()}
    level1_aliases = {key.lower(): value for key, value in config.taxonomy.level1_aliases.items()}
    level2_aliases = {key.lower(): value for key, value in config.taxonomy.level2_aliases.items()}

    level1 = sanitize_path_component(category_l1 or config.taxonomy.level1_default, config.taxonomy.level1_default)
    level2 = sanitize_path_component(category_l2 or config.taxonomy.level2_default, config.taxonomy.level2_default)

    level1 = global_aliases.get(level1.lower(), level1_aliases.get(level1.lower(), level1))
    level2 = global_aliases.get(level2.lower(), level2_aliases.get(level2.lower(), level2))
    return level1, level2


def align_with_existing_taxonomy(existing_taxonomy: Dict[str, List[str]], level1: str, level2: str) -> Tuple[str, str]:
    for existing_level1, existing_level2_items in existing_taxonomy.items():
        if existing_level1.lower() == level1.lower():
            level1 = existing_level1
            for existing_level2 in existing_level2_items:
                if existing_level2.lower() == level2.lower():
                    level2 = existing_level2
                    break
            break
    return level1, level2


def technical_destination(config: AppConfig, folder_name: str, filename: str) -> Path:
    safe_folder = sanitize_path_component(folder_name, "_NeedsReview")
    return config.organized_dir / safe_folder / filename
