"""Documentation-consistency checks for the DP-HONEY README (U7).

These tests keep the README's compatibility matrix and safety boundaries locked
to the live registry. Example tokens in the README are intentionally NOT asserted
against secret-looking regexes -- they are illustrative, synthetic, and reviewed
by hand (the plan's safety-path note).
"""

from __future__ import annotations

import re
from pathlib import Path

from detect.dp_honey import list_format_slugs

README = Path(__file__).resolve().parents[2] / "src" / "detect" / "dp_honey" / "README.md"

CLI_COMMANDS = [
    "list-formats",
    "preview-corpus",
    "train",
    "generate",
    "inspect-model",
    "validate",
    "report",
    "scan",
    "auto-decoy",
]

OUT_OF_SCOPE_KEYWORDS = ["calibration", "accounting", "gateway", "tool-call"]

# Matches a compatibility-matrix row whose first cell is a backtick-wrapped slug.
_ROW = re.compile(r"(?m)^\|\s*`([a-z0-9-]+)`\s*\|.*$")


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _matrix_rows() -> dict:
    return {m.group(1): m.group(0) for m in _ROW.finditer(_readme_text())}


def test_every_registered_format_appears_in_matrix():
    assert set(_matrix_rows()) == set(list_format_slugs())


def test_matrix_rows_carry_shape_only_wording():
    rows = _matrix_rows()
    assert rows, "expected a populated compatibility matrix"
    for slug, line in rows.items():
        assert "shape only" in line.lower(), slug


def test_readme_mentions_every_cli_command():
    text = _readme_text()
    for command in CLI_COMMANDS:
        assert command in text, command


def test_readme_states_out_of_scope_components():
    lowered = _readme_text().lower()
    assert "out of scope" in lowered
    for keyword in OUT_OF_SCOPE_KEYWORDS:
        assert keyword in lowered, keyword


def test_readme_carries_synthetic_safety_language():
    lowered = _readme_text().lower()
    assert "synthetic" in lowered
    assert "shape-only" in lowered or "shape only" in lowered
