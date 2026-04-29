"""Every typer entry-point should respond to `--help` with exit code 0.

This is the cheapest possible smoke test that catches:

- Module-level import errors (e.g. the `assert SAVE_DIR.exists()` regression
  that used to crash on a fresh clone).
- `typer.Option` / `typer.Argument` configuration errors.
- Missing or renamed dependencies that only manifest on import.
"""

import subprocess
import sys

import pytest

ENTRY_POINTS = [
    "dotted_wsd.train.hf_trainer",
    "dotted_wsd.dwsd_eval",
    "dotted_wsd.asbc_eval.process_into_instances",
    "dotted_wsd.asbc_eval.deduplicate_instances",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_help_exits_zero(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`python -m {module} --help` exited with {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Usage:" in result.stdout
