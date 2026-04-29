"""Run each carbon_emissions/ script against the bundled CSV and confirm exit 0.

`verify_report.py` is itself a self-test (it asserts every headline number
in the README matches what falls out of the CSV), so the meaningful check
is "does it exit with status 0".
"""

import subprocess
import sys
from pathlib import Path

import pytest

CARBON_DIR = Path(__file__).resolve().parent.parent / "carbon_emissions"
CSV = CARBON_DIR / "dotted_wsd_hardware_duration.csv"


@pytest.mark.parametrize(
    "script", ["calculate_carbon.py", "verify_report.py", "model_breakdown.py"]
)
def test_carbon_script_runs(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(CARBON_DIR / script), str(CSV)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{script} exited with {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_verify_report_passes_all_checks() -> None:
    """verify_report.py prints PASS/FAIL per check; require ALL CHECKS PASSED in stdout."""
    result = subprocess.run(
        [sys.executable, str(CARBON_DIR / "verify_report.py"), str(CSV)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ALL CHECKS PASSED" in result.stdout
    assert "FAIL" not in result.stdout
