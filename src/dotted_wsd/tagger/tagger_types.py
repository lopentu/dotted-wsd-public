from typing import TypeAlias

from typing_extensions import TypedDict

TokenId: TypeAlias = int
ExampleId: TypeAlias = int


class WsdInstanceForInference(TypedDict):
    example_id: int | str
    example_type: str
    target_word: str
    probe: str
    sense_id: str
    target_pos: str | None
    cwn_pos: str
    simplified_pos: str | None
    sense_def: str
    sense_refex: str  # reference example


class RpInsanceForInference(TypedDict):
    example_id: int
    example_type: str
    target_word: str
    probe: str
    typeclass_en: str
    typeclass_zh: str
    typeclass_gloss_zh: str
