"""
Unit tests for pipeline module.
"""

import tempfile
from pathlib import Path
import pytest
from src.pipeline import run_end_to_end_pipeline

def test_run_end_to_end_pipeline(temp_raw_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        df_out = run_end_to_end_pipeline(raw_dir=temp_raw_dir, save_outputs=False)
        
        assert not df_out.empty
        assert "category" in df_out.columns
        assert "confidence" in df_out.columns
        assert "locations" in df_out.columns
        assert "urgency_score" in df_out.columns
        assert "urgency_level" in df_out.columns
        assert "matched_terms" in df_out.columns
