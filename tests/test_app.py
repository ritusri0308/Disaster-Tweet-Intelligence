"""
Smoke tests for Streamlit application using AppTest.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

def test_app_load_smoke():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

def test_app_pages_smoke():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Verify radio buttons exist on sidebar with CrisisLens navigation options
    if len(at.sidebar.radio) > 0:
        radio = at.sidebar.radio[0]
        assert any("Situation Room" in opt for opt in radio.options)

        # Test navigation to About & Data page
        about_opt = next((opt for opt in radio.options if "About & Data" in opt), None)
        if about_opt:
            radio.set_value(about_opt).run()
            assert not at.exception
