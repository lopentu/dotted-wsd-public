from dataclasses import dataclass
from enum import Enum

import numpy.typing as npt
import pandas as pd
import torch
from pydantic import BaseModel
from typing_extensions import TypedDict


@dataclass
class DottedWsdModelOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    example_ids: npt.NDArray | torch.Tensor | None
    labels: npt.NDArray | torch.Tensor | None


class WsdInstanceForTesting(TypedDict):
    word: str
    sense_id: str
    pos: str
    context: str
    candidate: str
    example_id: int
    data_source: str
    label: int


class WsdInstanceFromCsv(TypedDict):
    """Each row from dt_asbc_dataset or WSD_merge_test_v2.csv or WSD_merge_train_v2.csv contains these values. Should be preprocessed to return `WsdInstanceForTesting`."""

    example_id: str
    test_word: str
    test_pos: str
    test_sense_id: str
    test_definition: str
    test_sentence: str
    cwn_sense_id: str
    cwn_definition: str
    cwn_sentence: str
    label: int
    source: str


class WsdInstance(TypedDict):
    example_id: int
    data_source: str
    context: str
    candidate: str
    label: int


class BatchedWsdInstance(TypedDict):
    example_ids: list[int]
    data_source: list[str]
    context: list[str]
    candidate: list[str]
    label: list[int]


class RpInstance(TypedDict):
    example_id: int
    data_source: str
    context: str
    candidate: str
    label: int


class BatchedRpInstance(TypedDict):
    example_ids: list[int]
    data_source: list[str]
    context: list[str]
    candidate: list[str]
    label: list[int]


class EvalCategory(str, Enum):
    WSD = "WSD"  # Word Sense Disambiguation
    RP = "RP"  # Regular Polysemy
    ALL = "ALL"  # Both WSD and RP


class PredictionVsGround(TypedDict):
    """
    A class to represent the comparison between predicted index and the ground truth index.

    Attributes:
    ----------
    prediction : int
        The index of the choice predicted to be correct.
    ground : Optional[int]
        The index of the correct choice, if available. If there are multiple correct choices, then there is no correct answer.
    total : int
        The total number of possible choices.
    """

    prediction: int
    ground: int | None
    total: int


class ScoreWsdInstanceOutput(TypedDict):
    accuracy: float | None
    examples_prediction_idx_vs_ground_idx: dict[int, PredictionVsGround]


class EvaluateWsdByExampleOutput(TypedDict):
    accuracy: float
    predictions: list[str]
    probabilities: list[float]
    num_candid: list[int]


class WsdByInstancePayload(TypedDict):
    accuracy: float | None
    df: pd.DataFrame


class WsdByExamplePayload(TypedDict):
    poshint_accuracy: float
    noposhint_accuracy: float
    df: pd.DataFrame


class CompleteEvaluateWsdOutput(TypedDict):
    metadata: dict | None
    by_example: WsdByExamplePayload | None
    by_instance: WsdByInstancePayload | None


class SingleEvaluateRpOutput(TypedDict):
    classification_report: pd.DataFrame
    predictions: list[tuple[str, float]]


class CompleteEvaluateRpOutput(TypedDict):
    hint: SingleEvaluateRpOutput
    nohint: SingleEvaluateRpOutput


class WsdExample(BaseModel):
    """Used for WSD evaluation from wsd_examples.csv"""

    test_word: str
    test_pos: str
    test_sense_id: str
    test_definition: str
    test_sentence: str
    cwn_sense_id: str
    cwn_definition: str
    cwn_sentence: str
    label: bool
    source: str
    example_id: int
