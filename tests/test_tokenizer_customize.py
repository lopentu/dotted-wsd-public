"""Confirm `tokenizers/customize.py` reproduces every shipped customized JSON.

The customized JSONs in `tokenizers/customized/` are vendored alongside the
repo. `tokenizers/customize.py` should derive each of them from its
corresponding `tokenizers/default/` JSON. Two checks here:

1. Run the script in `--check` mode end-to-end as a CLI: exit code 0 means
   every customized JSON is equal to what the script would produce.
2. Re-derive each customized JSON in-process and assert dict equality. This
   catches drift in either direction (an edit to the vendored JSON or a
   regression in the patch function).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tokenizers" / "customize.py"
CUSTOMIZED_DIR = REPO_ROOT / "tokenizers" / "customized"


def test_customize_check_mode_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"customize.py --check exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


sys.path.insert(0, str(REPO_ROOT / "tokenizers"))
from customize import PATCHES, customize_one  # noqa: E402


@pytest.mark.parametrize("name", sorted(PATCHES.keys()))
def test_customize_one_matches_vendored(name: str) -> None:
    produced = customize_one(name, PATCHES[name])
    with (CUSTOMIZED_DIR / f"{name}.json").open() as f:
        existing = json.load(f)
    assert produced == existing, (
        f"customize_one({name!r}) does not match tokenizers/customized/{name}.json"
    )
