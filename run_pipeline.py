"""
CLI entry point to execute the end-to-end processing pipeline.
Usage:
    python run_pipeline.py
    python run_pipeline.py --demo    # use synthetic demo data when the HumAID TSVs are not present
"""

import argparse
import logging
from src.pipeline import run_end_to_end_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the CrisisLens processing pipeline.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate and use a small synthetic dataset if no HumAID TSV files are found."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Executing Disaster Tweet Intelligence Pipeline...")

    df = run_end_to_end_pipeline(allow_synthetic=True if args.demo else None)
    logger.info(f"Pipeline executed successfully on {len(df)} tweets!")

if __name__ == "__main__":
    main()
