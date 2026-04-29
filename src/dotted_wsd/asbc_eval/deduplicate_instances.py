import json
from collections.abc import Hashable
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.progress import track

from dotted_wsd.settings import DATA_DIR

YELLOW = "\033[93m"
END = "\033[0m"


def groupby_example_id_and_select_first_group(
    group: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Hashable]]:
    """
    Group a DataFrame by "example_id" and select the first group.
    We just want the test sentence and test word to be unique. We don't want repeats across `example_id`s.
    """
    grouped = group.groupby("example_id")
    groups = list(grouped.groups.keys())

    _, subgroup = next(iter(grouped))

    return subgroup, groups


def main(
    path: Annotated[
        Path,
        typer.Argument(help="Path to the output of `process_into_instances.py`", exists=True),
    ] = DATA_DIR / "dt_asbc_dataset",
    save_dir: Annotated[
        Path,
        typer.Option(
            help=(
                "Directory to write the deduplicated CSV/feather and the JSON "
                "sidecar to. Defaults to the top-level data dir so downstream "
                "consumers (run_asbc_eval.sh) find them."
            )
        ),
    ] = DATA_DIR,
    debug: Annotated[bool, typer.Option(help="Process only the first two files.")] = False,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path.glob("*.csv"))
    if debug:
        files = files[:2]
    seen_test_word_test_sentences = set()
    test_sentence_to_example_ids: dict[
        str, list[Hashable]
    ] = {}  # maps test_sentence (with <test_word>) to all example_ids that share that test_sentence

    dfs = []  # this should only contain one example_id group for each test_sentence with a given <test_word>

    for f in track(files):
        df = pd.read_csv(
            f, dtype={"test_sense_id": str, "cwn_sense_id": str, "test_definition": str}
        )
        grouped_by_original_test_sentence_and_test_word = df.groupby("test_sentence")

        # groups of (test_sentence)
        for (
            name,
            group,
        ) in grouped_by_original_test_sentence_and_test_word:
            subgroup, groups = groupby_example_id_and_select_first_group(group)
            test_sentence_to_example_ids.setdefault(name, [])  # type: ignore
            test_sentence_to_example_ids[name].extend(groups)  # type: ignore
            if (
                name in seen_test_word_test_sentences
            ):  # don't add to final DF but keep track of how many there are
                continue

            dfs.append(subgroup)
            seen_test_word_test_sentences.add(name)

    total = sum(len(v) for v in test_sentence_to_example_ids.values())
    unique = len(test_sentence_to_example_ids)
    duplicates = total - unique
    print(f"Total number of test word/test sentence combinations: {YELLOW}{total}{END}")
    print(f"Number of unique test word/test sentence combinations: {YELLOW}{unique}{END}")
    print(f"Number of duplicate test word/test sentence combinations: {YELLOW}{duplicates}{END}")

    df = pd.concat(dfs, ignore_index=True)

    # rename example_id to example_src and then map example_src to example_id which is an integer, so that we can add it to a PyTorch tensor
    example_ids = df["example_id"].unique()
    ex_id_to_int = {example_id: i for i, example_id in enumerate(example_ids)}
    df = df.rename(columns={"example_id": "example_src"})
    df["example_id"] = df["example_src"].map(ex_id_to_int)

    df.to_csv(save_dir / "asbc-deduplicated-instances.csv", index=False)
    df.reset_index(drop=True).to_feather(
        save_dir / "asbc-deduplicated-instances.feather", compression="zstd"
    )
    with open(save_dir / "test_sentence_to_example_ids.json", "w") as f:
        json.dump(
            {
                "total": total,
                "unique": unique,
                "duplicates": duplicates,
                "res": test_sentence_to_example_ids,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Done!")


if __name__ == "__main__":
    typer.run(main)
