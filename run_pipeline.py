"""
CLI entry point to execute the end-to-end processing pipeline.
Usage:
    python run_pipeline.py
"""

import logging
from src.pipeline import run_end_to_end_pipeline

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Executing Disaster Tweet Intelligence Pipeline...")

    df = run_end_to_end_pipeline()
    logger.info(f"Pipeline executed successfully on {len(df)} tweets!")

if __name__ == "__main__":
    main()
