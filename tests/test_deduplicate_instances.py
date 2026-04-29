"""End-to-end test for `deduplicate_instances` against a synthetic fixture.

Confirms:

1. The CLI exits 0 when given a directory of `process_into_instances`-shaped
   CSVs.
2. All three expected outputs land in `--save-dir`: the canonical CSV, the
   zstd-compressed feather, and the JSON sidecar.
3. The CSV and the feather contain the same rows (same shape, same column
   set, same values).
4. Duplicate `(test_sentence, test_word)` pairs get collapsed: the input has
   two rows with identical `test_sentence` and the output has one.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Schema written by process_into_instances.gather_instances. The per-file
# CSVs use a *string* `example_id` (file/line/token coordinate). The dedupe
# step renames it to `example_src` and assigns a fresh integer `example_id`,
# which is why this fixture deliberately leaves `example_src` out.
COLUMNS = [
    "example_id",
    "test_word",
    "test_pos",
    "test_sense_id",
    "test_definition",
    "test_sentence",
    "cwn_sense_id",
    "cwn_definition",
    "cwn_sentence",
    "label",
    "source",
]


def _row(example_src: str, test_sentence: str, cwn_sense_id: str, label: int) -> dict:
    return {
        "example_id": example_src,
        "test_word": "一",
        "test_pos": "Neu",
        "test_sense_id": "5224202",
        "test_definition": "序數。順序排在第一。",
        "test_sentence": test_sentence,
        "cwn_sense_id": cwn_sense_id,
        "cwn_definition": "序數。順序排在第一。",
        "cwn_sentence": "舉例句。",
        "label": label,
        "source": "dt-asbc",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_deduplicate_writes_csv_feather_and_json(tmp_path: Path) -> None:
    input_dir = tmp_path / "dt_asbc_dataset"
    input_dir.mkdir()
    save_dir = tmp_path / "out"

    # Two distinct test_sentences, then a duplicate of the first.
    # Each sentence has 2 candidate senses (one labelled 1, one 0).
    _write_csv(
        input_dir / "asbc_dotted_tagged_000-of-140-eval-prepared.csv",
        [
            _row("file=000,line=1,token=0", "<一>、上午議程：", "5224201", 0),
            _row("file=000,line=1,token=0", "<一>、上午議程：", "5224202", 1),
            _row("file=000,line=2,token=0", "他清秀的面龐有<一>雙黑白分明的眼睛。", "5224201", 1),
            _row("file=000,line=2,token=0", "他清秀的面龐有<一>雙黑白分明的眼睛。", "5224202", 0),
        ],
    )
    # Same test_sentence as example=000,line=1 — should be collapsed.
    _write_csv(
        input_dir / "asbc_dotted_tagged_001-of-140-eval-prepared.csv",
        [
            _row("file=001,line=1,token=0", "<一>、上午議程：", "5224201", 0),
            _row("file=001,line=1,token=0", "<一>、上午議程：", "5224202", 1),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dotted_wsd.asbc_eval.deduplicate_instances",
            str(input_dir),
            "--save-dir",
            str(save_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"deduplicate_instances exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    csv_path = save_dir / "asbc-deduplicated-instances.csv"
    feather_path = save_dir / "asbc-deduplicated-instances.feather"
    json_path = save_dir / "test_sentence_to_example_ids.json"

    assert csv_path.exists(), "expected canonical CSV not written"
    assert feather_path.exists(), "expected feather not written"
    assert json_path.exists(), "expected JSON sidecar not written"

    csv_df = pd.read_csv(
        csv_path,
        dtype={"test_sense_id": str, "cwn_sense_id": str, "test_definition": str},
    )
    feather_df = pd.read_feather(feather_path)

    # Same shape, same columns.
    assert csv_df.shape == feather_df.shape, (
        f"CSV {csv_df.shape} and feather {feather_df.shape} disagree on shape"
    )
    assert sorted(csv_df.columns) == sorted(feather_df.columns)

    # Each row in CSV must round-trip through the feather. Cast feather columns
    # to CSV dtypes since CSV has no native dtype info.
    fea_aligned = feather_df[csv_df.columns.tolist()].copy()
    for col in csv_df.columns:
        fea_aligned[col] = fea_aligned[col].astype(csv_df[col].dtype)
    pd.testing.assert_frame_equal(
        csv_df.reset_index(drop=True),
        fea_aligned.reset_index(drop=True),
    )

    # Duplicate test_sentence from example_id=2 should have been collapsed:
    # the deduplicated dataset retains 2 unique test_sentences, each with 2
    # candidate-sense rows = 4 rows total.
    assert csv_df["test_sentence"].nunique() == 2
    assert len(csv_df) == 4
