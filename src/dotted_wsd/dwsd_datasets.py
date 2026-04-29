import json
from itertools import chain, repeat
from multiprocessing import Manager, cpu_count
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from CwnGraph import CwnBase
from datasets import Dataset as HfDataset
from loguru import logger
from rich.console import Console
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .dwsd_types import (
    BatchedRpInstance,
    BatchedWsdInstance,
    RpInstance,
    WsdInstance,
    WsdInstanceForTesting,
)
from .settings import DATA_DIR
from .utils import multiprocess_with_progress

# ref: CWNSensetagger-dev

RPTAG = "rp"
WSDTAG = "wsd"

console = Console()


class RpDatasetForTraining(Dataset):
    def __init__(self, datapath, gdpath, is_debug=False):
        self.data = pd.read_csv(datapath)
        self.glossdict_path = gdpath
        self.is_debug = is_debug

        with open(self.glossdict_path, encoding="UTF-8") as f:
            self.gd = json.load(f)

        self.flattened = self.preprocess(self.data, self.gd)

    def __len__(self):
        if self.is_debug:
            return 100
        else:
            return len(self.flattened)

    def __getitem__(self, idx):
        return self.flattened[idx]

    def preprocess(self, data, gloss_dict) -> list[RpInstance]:
        flattened: list[RpInstance] = []

        for row_id, row in data.iterrows():
            row_sentence = row["Sentence"]
            row_label = row["RP Class"]
            row_word = row["Word"]
            row_dot_obj = row["dot_obj"]

            # skip row_label that is 'dot' or 'X'
            if row_label not in row_dot_obj:
                continue

            type_classes = row_dot_obj.split("*")
            for type_class_x in type_classes:
                # create a context
                contexts = row_sentence

                # create a candidate
                gloss_entry = gloss_dict[type_class_x]
                zh_cand = gloss_entry["zh_trans"]
                zh_cand_gloss = gloss_entry["zh_gloss"]
                candidates = f"{row_word},{zh_cand},{zh_cand_gloss}"

                index_label = 1 if type_class_x == row_label else 0

                # print(len(contexts), len(candidates))
                instance: RpInstance = {
                    "context": contexts,
                    "candidate": candidates,
                    "data_source": RPTAG,
                    "label": index_label,
                    "example_id": row_id,
                }

                flattened.append(instance)

        return flattened


class WsdDatasetForTraining(Dataset):
    def __init__(self, datapath, is_debug=False):
        self.data = pd.read_csv(
            datapath,
            dtype={"test_sense_id": str, "cwn_sense_id": str, "test_definition": str},
        )
        self.instances = self.preprocess(self.data)
        self.is_debug = is_debug

    def __len__(self):
        if self.is_debug:
            return 100
        else:
            return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def preprocess(self, data) -> list[WsdInstance]:
        # [CLS] <instance> [SEP] <word>,<candidate_sense>,<candidate_sense例句>
        # [CLS] s['sentence_id'] [SEP] s['target_word_id'] [COMMA] ['cwn_definition_id'] [COMMA] s['cwn_sentence_id'] [SEP]

        # drop those examples not having a correct answer (happens ~0.1% in annotation data)
        exid_sum = data.groupby("example_id").apply(lambda x: x.label.sum())
        excl_ids = exid_sum[exid_sum == 0].index.values
        data = data.loc[~data.example_id.isin(excl_ids)]

        instances = []
        for _, row in data.iterrows():
            word = row["test_word"]
            sentence = row["test_sentence"]
            candidate_sense = row["cwn_definition"]
            cand_ex = row["cwn_sentence"]
            example_id = row["example_id"]
            context = sentence
            candidate = f"{word},{candidate_sense},{cand_ex}"
            label = row["label"]

            # example_id is offset by 10000 to leave room for RP dataset
            instance: WsdInstance = {
                "context": context,
                "candidate": candidate,
                "example_id": 10000 + example_id,
                "data_source": WSDTAG,
                "label": int(label),
            }

            instances.append(instance)

        return instances


class WsdDatasetForInstanceTesting(Dataset):
    def __init__(
        self,
        datapath: Path,
        debug: bool = False,
        preprocess_workers: int = 8,
        debug_dataset_size: int = 1000,
    ):
        self.debug = debug
        self.debug_dataset_size = debug_dataset_size
        self.data = self._load_data(datapath)
        self.cwn = CwnBase()
        self.id_map = {}
        self.instances: list[WsdInstanceForTesting] = self.preprocess(
            max_workers=preprocess_workers
        )

    def _load_data(self, datapath: Path) -> pd.DataFrame:
        with console.status("[bold green]Loading data...", spinner="dots"):
            if datapath.suffix == ".csv":
                df = pd.read_csv(
                    datapath,
                    dtype={
                        "test_sense_id": str,
                        "cwn_sense_id": str,
                        "test_definition": str,
                    },
                    nrows=self.debug_dataset_size if self.debug else None,
                )
            elif datapath.suffix == ".feather":
                df = pd.read_feather(datapath)
                if self.debug:
                    df = df.head(self.debug_dataset_size)
            else:
                raise ValueError("Unsupported file format")

            # drop those examples not having a correct answer (happens ~0.1% in annotation data)
            exid_sum = df.groupby("example_id").apply(lambda x: x.label.sum())
            excl_ids = exid_sum[exid_sum == 0].index.values
            return df.loc[~df.example_id.isin(excl_ids)]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return {k: self.instances[idx][k] for k in ("candidate", "context", "label", "example_id")}

    def preprocess(self, max_workers: int | None = None) -> list[WsdInstanceForTesting]:
        if max_workers is None:
            max_workers = cpu_count() or 1
        with Manager() as manager:
            shared_dict = manager.dict()

            df_chunks = np.array_split(self.data, indices_or_sections=max_workers)
            res = multiprocess_with_progress(
                self._preprocess_worker,
                list(zip(df_chunks, repeat(shared_dict))),
                description="Preprocessing...",
            )
            assert res
            flattened = list(chain.from_iterable(res))
            self.id_map = dict(shared_dict)

        return flattened

    @staticmethod
    def _preprocess_worker(
        df: pd.DataFrame, shared_id_map: DictProxy
    ) -> list[WsdInstanceForTesting]:
        # [CLS] <instance> [SEP] <word>,<candidate_sense>,<candidate_sense例句>
        # [CLS] s['sentence_id'] [SEP] s['target_word_id'] [COMMA] ['cwn_definition_id'] [COMMA] s['cwn_sentence_id'] [SEP]

        cwn = CwnBase()

        instances = []
        for row in df.itertuples(index=False):
            word = cast(str, row.test_word)
            sentence = cast(str, row.test_sentence)
            candidate_sense = cast(str, row.cwn_definition)
            sense_def = cast(str, row.test_definition)
            cand_ex = cast(str, row.cwn_sentence)
            example_id = cast(int, row.example_id)
            context = sentence
            candidate = f"{word},{candidate_sense},{cand_ex}"
            label = cast(int, row.label)
            if not isinstance(row.test_sense_id, str):
                label_key = f"{word}:{sense_def}"
                if label_key not in shared_id_map:
                    senses = cwn.find_all_senses(word)
                    test_sense_id = next(x.id for x in senses if x.definition == sense_def)
                    shared_id_map[label_key] = test_sense_id
                test_sense_id = shared_id_map[label_key]
            else:
                test_sense_id = row.test_sense_id
            # example_id is offset by 10000 to leave room for RP dataset
            instance: WsdInstanceForTesting = {
                "word": word,
                "sense_id": test_sense_id,
                "pos": str(row.test_pos),
                "context": context,
                "candidate": candidate,
                "example_id": int(10000 + example_id),
                "data_source": WSDTAG,
                "label": int(label),  # convert bool to int
            }

            instances.append(instance)

        return instances

    # def preprocess(self, data):
    #     # [CLS] <instance> [SEP] <word>,<candidate_sense>,<candidate_sense例句>
    #     # [CLS] s['sentence_id'] [SEP] s['target_word_id'] [COMMA] ['cwn_definition_id'] [COMMA] s['cwn_sentence_id'] [SEP]

    #     # drop those examples not having a correct answer (happens ~0.1% in annotation data)

    #     instances = []
    #     for _, row in track(data.iterrows(), total=data.shape[0]):
    #         word = row["test_word"]
    #         sentence = row["test_sentence"]
    #         candidate_sense = row["cwn_definition"]
    #         sense_def = row["test_definition"]
    #         cand_ex = row["cwn_sentence"]
    #         example_id = row["example_id"]
    #         context = sentence
    #         candidate = f"{word},{candidate_sense},{cand_ex}"
    #         label = row["label"]
    #         if not isinstance(row["test_sense_id"], str):
    #             label_key = f"{word}:{sense_def}"
    #             if label_key not in self.id_map:
    #                 senses = self.cwn.find_all_senses(word)
    #                 test_sense_id = [x.id for x in senses if x.definition == sense_def][
    #                     0
    #                 ]
    #                 self.id_map[label_key] = test_sense_id
    #             test_sense_id = self.id_map[label_key]
    #         else:
    #             test_sense_id = row["test_sense_id"]
    #         # example_id is offset by 10000 to leave room for RP dataset
    #         instance: WsdInstanceForTesting = {
    #             "word": word,
    #             "sense_id": test_sense_id,
    #             "pos": row["test_pos"],
    #             "context": context,
    #             "candidate": candidate,
    #             "example_id": 10000 + example_id,
    #             "data_source": WSDTAG,
    #             "label": int(label),  # convert bool to int
    #         }

    #         instances.append(instance)

    #     return instances


class DataCollatorForDottedWsd:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        remove_token_type_ids: bool = False,
        padding: bool = True,
        max_length: int = 320,
        pad_to_multiple_of: int = 8,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.remove_token_type_ids = remove_token_type_ids

    def __call__(self, examples):
        labels = [ex["label"] for ex in examples]
        contexts = [ex["context"] for ex in examples]
        candidates = [ex["candidate"] for ex in examples]
        batch = self.tokenizer(
            contexts,
            candidates,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        batch["example_ids"] = torch.tensor(
            [ex["example_id"] for ex in examples], dtype=torch.int32
        )
        if labels[0] is not None:
            batch["labels"] = torch.tensor(labels)  # BCE
        if self.remove_token_type_ids:
            batch.pop("token_type_ids", None)

        return batch


class DottedWsdDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst_x = self.instances[idx]
        out = {
            "example_id": inst_x["example_id"],
            "context": inst_x["probe"],
            "label": None,
        }

        if inst_x["example_type"] == "wsd":
            out["candidate"] = "{},{},{}".format(
                inst_x["target_word"], inst_x["sense_def"], inst_x["sense_refex"]
            )

        elif inst_x["example_type"] == "rp":
            out["candidate"] = "{},{},{}".format(
                inst_x["target_word"],
                inst_x["typeclass_zh"],
                inst_x["typeclass_gloss_zh"],
            )
        else:
            raise ValueError("Unknown ex_type: " + inst_x["ex_type"])

        return out


def _prepare_batched_examples_for_model(
    examples: BatchedWsdInstance | BatchedRpInstance,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int = 320,
    padding: bool = True,
    pad_to_multiple_of: int = 8,
) -> dict:
    contexts = examples["context"]
    candidates = examples["candidate"]

    tokenized = tokenizer(
        contexts,
        candidates,
        max_length=max_length,
        truncation=True,
        padding=padding,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    token_len = [len(tok) for tok in tokenized.input_ids]
    return {**tokenized, "length": token_len}


def load_dotted_datasets_for_training(
    rp_train: Path | str = DATA_DIR / "RP_train.csv",
    rp_test: Path | str = DATA_DIR / "RP_valid.csv",
    wsd_train: Path | str = DATA_DIR / "WSD_merge_train_v2.csv",
    wsd_test: Path | str = DATA_DIR / "WSD_merge_test_v2.csv",
    gdpath: Path | str = DATA_DIR / "glossdict.json",
    is_debug: bool = False,
) -> tuple[Dataset, Dataset]:
    rp_train_ds = RpDatasetForTraining(rp_train, gdpath=gdpath, is_debug=is_debug)
    rp_test_ds = RpDatasetForTraining(rp_test, gdpath=gdpath, is_debug=is_debug)
    wsd_train_ds = WsdDatasetForTraining(wsd_train, is_debug=is_debug)
    wsd_test_ds = WsdDatasetForTraining(wsd_test, is_debug=is_debug)

    mix_train = ConcatDataset([rp_train_ds, wsd_train_ds])
    mix_test = ConcatDataset([rp_test_ds, wsd_test_ds])

    logger.info(f"RP Train: {len(rp_train_ds)}")
    logger.info(f"RP Test: {len(rp_test_ds)}")
    logger.info(f"WSD Train: {len(wsd_train_ds)}")
    logger.info(f"WSD Test: {len(wsd_test_ds)}")

    return mix_train, mix_test


def load_dotted_datasets_for_hf_trainer(
    rp_train: Path | str = DATA_DIR / "RP_train.csv",
    rp_test: Path | str = DATA_DIR / "RP_valid.csv",
    wsd_train: Path | str = DATA_DIR / "WSD_merge_train_v2.csv",
    wsd_test: Path | str = DATA_DIR / "WSD_merge_test_v2.csv",
    gdpath: Path | str = DATA_DIR / "glossdict.json",
    is_debug: bool = False,
    preprocess: bool = True,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    max_length: int | None = 320,
) -> tuple[HfDataset, HfDataset] | tuple[Dataset, Dataset]:
    mix_train, mix_test = load_dotted_datasets_for_training(
        rp_train=rp_train,
        rp_test=rp_test,
        wsd_train=wsd_train,
        wsd_test=wsd_test,
        gdpath=gdpath,
        is_debug=is_debug,
    )

    mix_train = HfDataset.from_list(mix_train)  # type: ignore
    mix_test = HfDataset.from_list(mix_test)  # type: ignore

    if not preprocess:
        return mix_train, mix_test

    if not tokenizer:
        raise ValueError("Tokenizer must be provided for preprocessing")

    if not tokenizer.sep_token:
        logger.warning(
            "Tokenizer does not have a sep_token. Make sure that the post_processor is set correctly."
        )

    mix_train = mix_train.map(
        _prepare_batched_examples_for_model,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
        },
        num_proc=cpu_count(),
        batched=True,
    )
    mix_test = mix_test.map(
        _prepare_batched_examples_for_model,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
        },
        num_proc=cpu_count(),
        batched=True,
    )

    return mix_train, mix_test
