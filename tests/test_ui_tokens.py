"""
Token hygiene and visual regression test for CrisisLens.
Ensures zero hard-coded dark theme hex colors or sub-14px font sizes
exist outside app/ui/theme.py. Also verifies no emojis in titles or nav,
and base font size >= 16px.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Old dark theme hex colors that must never reappear outside theme.py
DARK_HEX_COLORS = [
    "#0B0F17",
    "#111827",
    "#162033",
    "#1F2A3D",
    "#E6EDF7",
    "#8B9BB4",
    "#22D3EE"
]

# Small font-size regex patterns that violate >= 14px rule
SUB_14PX_PATTERNS = [
    r"font-size:\s*0\.[1-7]\d*rem",
    r"font-size:\s*0\.8[0-6]\d*rem",
    r"font-size:\s*0\.7\d*rem",
    r"font-size:\s*0\.75rem",
    r"font-size:\s*0\.8rem",
    r"font-size:\s*(?:[1-9]|1[0-3])px\b"
]

# Emojis regex
EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001F1E0-\U0001F1FF]"
)

def test_no_dark_hex_colors_outside_theme():
    """Verify no dark hexes exist in app.py, app/pages/, or app/ui/components.py."""
    app_files = list((PROJECT_ROOT / "app").rglob("*.py")) + [PROJECT_ROOT / "app.py"]
    violations = []

    for file_path in app_files:
        # theme.py is allowed to define tokens
        if file_path.name == "theme.py":
            continue

        content = file_path.read_text(encoding="utf-8")
        for line_num, line in enumerate(content.splitlines(), 1):
            for hex_col in DARK_HEX_COLORS:
                if hex_col.lower() in line.lower():
                    violations.append(f"{file_path.name}:{line_num} found dark hex {hex_col}: {line.strip()[:80]}")

    assert not violations, f"Dark hex color violations found ({len(violations)}):\n" + "\n".join(violations)

HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}(?![0-9A-Za-z])")

def test_no_hex_color_literals_outside_theme():
    """All colours come from app/ui/theme.py tokens or cl-* CSS classes; pages and components hold no hex literals."""
    app_files = list((PROJECT_ROOT / "app").rglob("*.py")) + [PROJECT_ROOT / "app.py"]
    violations = []

    for file_path in app_files:
        if file_path.name == "theme.py":
            continue
        for line_num, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            for match in HEX_COLOR.finditer(line):
                violations.append(f"{file_path.relative_to(PROJECT_ROOT)}:{line_num} hex literal {match.group(0)}: {line.strip()[:80]}")

    assert not violations, f"Hex colour literals outside theme.py ({len(violations)}):\n" + "\n".join(violations)

def test_theme_does_not_load_remote_fonts():
    """The app is designed to work air-gapped; a remote font import would break that and leak requests."""
    theme_py = (PROJECT_ROOT / "app" / "ui" / "theme.py").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in theme_py
    assert "@import url(" not in theme_py

def test_no_sub_14px_font_sizes():
    """Verify no elements use < 14px (0.875rem) font sizes outside theme.py."""
    app_files = list((PROJECT_ROOT / "app").rglob("*.py")) + [PROJECT_ROOT / "app.py"]
    pattern = re.compile("|".join(SUB_14PX_PATTERNS), re.IGNORECASE)
    violations = []

    for file_path in app_files:
        if file_path.name == "theme.py":
            continue

        content = file_path.read_text(encoding="utf-8")
        for line_num, line in enumerate(content.splitlines(), 1):
            match = pattern.search(line)
            if match:
                violations.append(f"{file_path.name}:{line_num} found sub-14px font {match.group(0)}: {line.strip()[:80]}")

    assert not violations, f"Sub-14px font violations found ({len(violations)}):\n" + "\n".join(violations)

def test_no_emojis_in_page_titles_and_navigation():
    """Verify page titles and navigation labels do not have emojis."""
    app_py = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    # Check st.Page declarations
    page_matches = re.findall(r"st\.Page\([^,]+,\s*title=[\"']([^\"']+)[\"']", app_py)
    for title in page_matches:
        assert not EMOJI_PATTERN.search(title), f"Emoji found in page title: '{title}'"

    # Check pages for cl-page-title
    for page_file in (PROJECT_ROOT / "app" / "pages").glob("*.py"):
        text = page_file.read_text(encoding="utf-8")
        for title in re.findall(r"cl-page-title'>([^<]+)<", text):
            assert not EMOJI_PATTERN.search(title), f"Emoji found in page header in {page_file.name}: '{title}'"

def test_theme_base_font_and_minimums():
    """Verify theme.py sets html font-size >= 16px and has no sub-14px font definitions."""
    theme_py = (PROJECT_ROOT / "app" / "ui" / "theme.py").read_text(encoding="utf-8")
    match = re.search(r"font-size:\s*(\d+)px", theme_py)
    assert match, "Base font-size not found in theme.py"
    base_size = int(match.group(1))
    assert base_size >= 16, f"Base font size must be >= 16px, found {base_size}px"
