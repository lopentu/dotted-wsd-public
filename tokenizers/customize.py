#!/usr/bin/env python3
"""Reproduce tokenizers/customized/*.json from tokenizers/default/*.json.

Each base model has a small per-base patch function that mutates the default
tokenizer JSON in place. The customizations register one or two new special
tokens with the Rust-level `post_processor` so the fine-tuned model sees a
context-candidate input wrapped in those tokens, and promote those tokens'
`added_tokens` entries to `special=True`.

Run:

    uv run python tokenizers/customize.py            # write tokenizers/customized/
    uv run python tokenizers/customize.py --check    # diff against existing files; exit 1 on drift

No third-party dependencies: read JSON, mutate dict, write JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = THIS_DIR / "default"
CUSTOMIZED_DIR = THIS_DIR / "customized"


def _promote_special(added_tokens: list[dict[str, Any]], token_id: int) -> None:
    """Set `special=True` on the added_tokens entry with the given id."""
    for t in added_tokens:
        if t["id"] == token_id:
            t["special"] = True
            return
    raise ValueError(f"token id {token_id} not in added_tokens")


def _normalize_merges_to_list_pairs(model: dict[str, Any]) -> None:
    """Convert `model.merges` from "<a> <b>" strings to ["<a>", "<b>"] pairs.

    The default JSONs ship with the older space-joined string format; the
    paper's customized JSONs were saved by a newer `tokenizers` library that
    serializes merges as 2-element lists. The two formats are equivalent at
    runtime but byte-comparing them requires us to canonicalize first.
    """
    if not model.get("merges"):
        return
    if isinstance(model["merges"][0], str):
        model["merges"] = [m.split(" ", 1) for m in model["merges"]]


def patch_smollm(j: dict[str, Any]) -> None:
    """Mxode/SmolLM-Chinese-180M.

    Wrap each sequence in `<|startoftext|>` (id=1) and append `<|unused159|>`
    (id=304) as a sentinel marking the end of the candidate. Promote
    `<|unused159|>` to `special=True` in `added_tokens`.
    """
    j["post_processor"]["pair"] = [
        {"SpecialToken": {"id": "<|startoftext|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|startoftext|>", "type_id": 1}},
        {"Sequence": {"id": "B", "type_id": 1}},
        {"SpecialToken": {"id": "<|unused159|>", "type_id": 1}},
    ]
    j["post_processor"]["special_tokens"] = {
        "<|startoftext|>": {
            "id": "<|startoftext|>",
            "ids": [1],
            "tokens": ["<|startoftext|>"],
        },
        "<|unused159|>": {
            "id": "<|unused159|>",
            "ids": [304],
            "tokens": ["<|unused159|>"],
        },
    }
    _promote_special(j["added_tokens"], 304)


def patch_gemma(j: dict[str, Any]) -> None:
    """google/gemma-2-2b.

    Gemma's default already wraps inputs with `<bos>`. We append `<unused99>`
    (id=255999) as the candidate-end sentinel and promote it to
    `special=True` in `added_tokens`.

    Note: the existing customized JSON has `"tokens": ["<unused99"]` (missing
    the trailing `>`). Reproduced byte-for-byte; the typo is harmless because
    tokenization does not consult the `tokens` field.
    """
    j["post_processor"]["pair"] = [
        {"SpecialToken": {"id": "<bos>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<bos>", "type_id": 1}},
        {"Sequence": {"id": "B", "type_id": 1}},
        {"SpecialToken": {"id": "<unused99>", "type_id": 1}},
    ]
    j["post_processor"]["special_tokens"] = {
        "<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]},
        "<unused99>": {
            "id": "<unused99>",
            "ids": [255999],
            "tokens": ["<unused99"],
        },
    }
    _promote_special(j["added_tokens"], 255999)


def patch_llama(j: dict[str, Any]) -> None:
    """meta-llama/Llama-3.2-3B.

    Llama's default `post_processor` is a `Sequence` of two processors; the
    second is a `TemplateProcessing`. Append `<|eot_id|>` (id=128009) to its
    `pair` template as the candidate-end sentinel.

    Also normalizes `model.merges` from the older space-joined string format
    to the newer 2-element-list format, since the existing customized JSON
    uses the latter (purely a serialization quirk; tokenization is identical).
    """
    inner = j["post_processor"]["processors"][1]
    inner["pair"] = [
        {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 1}},
        {"Sequence": {"id": "B", "type_id": 1}},
        {"SpecialToken": {"id": "<|eot_id|>", "type_id": 1}},
    ]
    inner["special_tokens"] = {
        "<|begin_of_text|>": {
            "id": "<|begin_of_text|>",
            "ids": [128000],
            "tokens": ["<|begin_of_text|>"],
        },
        "<|eot_id|>": {
            "id": "<|eot_id|>",
            "ids": [128009],
            "tokens": ["<|eot_id|>"],
        },
    }
    _normalize_merges_to_list_pairs(j["model"])


PATCHES: dict[str, Callable[[dict[str, Any]], None]] = {
    "Mxode-SmolLM-Chinese-180M": patch_smollm,
    "google-gemma-2-2b": patch_gemma,
    "meta-llama-Llama-3.2-3B": patch_llama,
}


def customize_one(name: str, patch: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    default_path = DEFAULT_DIR / f"{name}.json"
    with default_path.open() as f:
        default_json = json.load(f)
    result = copy.deepcopy(default_json)
    patch(result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help=(
            "Don't write; instead, diff the freshly produced JSON against the "
            "existing tokenizers/customized/*.json and exit 1 on any mismatch."
        ),
    )
    args = ap.parse_args()

    drift = 0
    for name, patch in PATCHES.items():
        produced = customize_one(name, patch)
        target_path = CUSTOMIZED_DIR / f"{name}.json"

        if args.check:
            with target_path.open() as f:
                existing = json.load(f)
            if produced == existing:
                print(f"  [ok]   {name}")
            else:
                print(f"  [DIFF] {name}")
                drift += 1
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w") as f:
                json.dump(produced, f, ensure_ascii=False)
            print(f"  wrote {target_path}")

    if args.check and drift:
        print(
            f"\n{drift} of {len(PATCHES)} customized JSONs differ from what this script produces."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
