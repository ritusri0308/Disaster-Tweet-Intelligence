"""
Guards that the test suite can never overwrite real production artifacts.

A previous version of the suite trained models and wrote reports straight into ``models/`` and
``reports/``, silently replacing the production classifier with a 48-term fixture. conftest.py now
redirects every output directory to a temp location; these tests fail loudly if that ever regresses.
"""

from pathlib import Path

import src.config as config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def test_output_paths_are_redirected_out_of_the_project():
    for name in ("DATA_DIR", "MODELS_DIR", "REPORTS_DIR", "MODEL_PATH", "DB_PATH",
                 "PROCESSED_PARQUET_PATH", "PROCESSED_CSV_PATH"):
        path = getattr(config, name)
        assert not _is_inside(path, PROJECT_ROOT), f"{name}={path} points inside the project during tests"

def test_modules_that_write_artifacts_use_the_isolated_paths():
    import src.classifier as classifier
    import src.needs as needs
    import src.geocoding as geocoding
    import src.pipeline as pipeline

    assert classifier.MODEL_PATH == config.MODEL_PATH
    assert classifier.REPORTS_DIR == config.REPORTS_DIR
    assert needs.REPORTS_DIR == config.REPORTS_DIR
    assert geocoding.REPORTS_DIR == config.REPORTS_DIR
    assert pipeline.MODEL_PATH == config.MODEL_PATH

def test_env_override_helper_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("CRISISLENS_TEST_DIR", raising=False)
    default = Path("/some/default")
    assert config._env_path("CRISISLENS_TEST_DIR", default) == default
    monkeypatch.setenv("CRISISLENS_TEST_DIR", str(PROJECT_ROOT))
    assert config._env_path("CRISISLENS_TEST_DIR", default) == PROJECT_ROOT.resolve()
