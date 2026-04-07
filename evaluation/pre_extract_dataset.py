#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import argparse
import logging

# Add the parent directory to sys.path to import ai_smart_folders
sys.path.append(str(Path(__file__).parent.parent))

from ai_smart_folders.config import load_config
from ai_smart_folders.pipeline import SmartFoldersPipeline
from ai_smart_folders.models import DocumentEnvelope

def main():
    parser = argparse.ArgumentParser(description="Pre-calculate OCR and PDF extraction for a dataset.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--dataset", required=True, help="Path to evaluation JSONL")
    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s: %(message)s')
    logger = logging.getLogger("pre-extract")

    # Load config and pipeline
    config = load_config(Path(args.config))
    pipeline = SmartFoldersPipeline(config, logger=logger, api_logger=logger)

    # Load dataset
    dataset_path = Path(args.dataset)
    envelopes = []
    with open(dataset_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            # Find the file path
            file_path = Path(item["source_path"])
            if not file_path.exists() and not file_path.is_absolute():
                file_path = (dataset_path.parent / file_path).resolve()
            
            if not file_path.exists():
                logger.error(f"File not found: {item['source_path']}")
                continue

            envelope = DocumentEnvelope(
                document_id=file_path.name,
                run_id="pre-extract",
                source_path=file_path,
                filename=file_path.name, extension=file_path.suffix.lower()
            )
            envelopes.append(envelope)

    logger.info(f"Loaded {len(envelopes)} items from dataset.")

    # Phase 1: Batch PDFs (ODL)
    logger.info("Starting ODL pre-processing...")
    pipeline._preprocess_pdfs_batch(envelopes)
    
    # Phase 2: Sequential OCR/Extraction
    logger.info("Starting sequential OCR/Extraction...")
    processed = 0
    for env in envelopes:
        # If it was already handled by ODL, it's in the odl_cache and will be cached properly later
        # Actually, let's just run _extract_stage for all
        try:
            pipeline._extract_stage(env)
            processed += 1
            if processed % 5 == 0:
                logger.info(f"Progress: {processed}/{len(envelopes)}")
        except Exception as e:
            logger.error(f"Failed to extract {env.filename}: {e}")

    logger.info("Pre-extraction complete. Cache is now warmed up.")

if __name__ == "__main__":
    main()
