"""Smoke tests: verify CLI --help works for all entry points."""

import subprocess
import sys

import pytest

MODULES = [
    "nnlc_tools.sync_rlogs",
    "nnlc_tools.extract_lateral_data",
    "nnlc_tools.score_routes",
    "nnlc_tools.visualize_coverage",
    "nnlc_tools.visualize_model",
    "nnlc_tools.analyze_interventions",
]


@pytest.mark.parametrize("module", MODULES)
def test_cli_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{module} --help failed: {result.stderr}"
    assert "usage:" in result.stdout.lower() or "optional arguments" in result.stdout.lower()
