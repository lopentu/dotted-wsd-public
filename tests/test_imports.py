"""Every importable module under `src/dotted_wsd/` should import cleanly on a fresh sync."""

import importlib

import pytest

MODULES = [
    "dotted_wsd",
    "dotted_wsd.dwsd_datasets",
    "dotted_wsd.dwsd_eval",
    "dotted_wsd.dwsd_types",
    "dotted_wsd.settings",
    "dotted_wsd.utils",
    "dotted_wsd.asbc_eval.asbc_types",
    "dotted_wsd.asbc_eval.deduplicate_instances",
    "dotted_wsd.asbc_eval.process_into_instances",
    "dotted_wsd.tagger.cwn_pos_map",
    "dotted_wsd.tagger.model",
    "dotted_wsd.tagger.prediction",
    "dotted_wsd.tagger.preprocessing",
    "dotted_wsd.tagger.tagger_types",
    "dotted_wsd.tagger.utils",
    "dotted_wsd.train.hf_trainer",
    "dotted_wsd.train.lora",
    "dotted_wsd.train.utils",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
