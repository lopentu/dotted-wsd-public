import json
from collections.abc import Callable
from pathlib import Path

import evaluate
import torch
import torch.nn.functional as F
import yaml
from transformers import EvalPrediction, PreTrainedTokenizerFast

YELLOW = "\033[93m"
END = "\033[0m"


def compute_accuracy() -> Callable[[EvalPrediction], dict]:
    # https://huggingface.co/spaces/evaluate-metric/accuracy
    accuracy = evaluate.load("accuracy")

    def inner(eval_pred: EvalPrediction) -> dict:
        predictions, labels = eval_pred
        # 2 labels
        predictions = F.softmax(torch.from_numpy(predictions), dim=1)
        predictions = torch.argmax(predictions, dim=1)

        # 1 label
        # predictions = F.sigmoid(predictions)
        # predictions = (predictions > threshold).int().flatten()
        res = accuracy.compute(
            predictions=predictions, references=labels
        )  # returns {"accuracy": accuracy} or None if not main process
        return res if res is not None else {}

    return inner


def build_suffix(
    use_bf16: bool,
    # handle_sep_token_strategy: Optional[HandleSepTokenStrategy]
) -> str:
    # no suffix is {context} [SEP] {candidate}
    # v1.1 is {context} [SEP] {candidate} [SEP] with last few tokens replaced with [SEP] to stay within max_length
    # v1.2 is {context} [SEP] {candidate} [SEP] with [SEP] directly appended to the end
    suffix = ""
    # if (
    #     handle_sep_token_strategy
    #     == HandleSepTokenStrategies.replace_last_tokens_with_sep_token
    # ):
    #     suffix += "-R"
    # elif (
    #     handle_sep_token_strategy
    #     == HandleSepTokenStrategies.append_custom_sep_token_to_end
    # ):
    #     suffix += "-A"

    suffix += "" if use_bf16 else "-fp32"

    return suffix


def check_special_tokens(added_tokens_list: list[dict], special_tokens: dict) -> None:
    added_tokens = [token["content"] for token in added_tokens_list]
    for token in special_tokens.values():
        if token not in added_tokens:
            raise ValueError(f"Special token {token} not in added tokens")
        idx = added_tokens.index(token)
        attributes = added_tokens_list[idx]
        if not attributes["special"]:
            raise ValueError(f"Special token {token} is not marked as special")


def check_template_proccessing_pair_format(
    pair_format: list[dict], special_tokens_mapping: dict
) -> None:
    if len(pair_format) != 5:
        raise ValueError(
            "Template processing pair format must have 5 elements in the following format: [SpecialToken, Sequence, SpecialToken, Sequence, SpecialToken]"
        )
    # Check format has the following structure: [SPECIAL_1] SEQ_A [SPECIAL_1] SEQ_B [SPECIAL_2]
    structure_to_check = [
        "SpecialToken",
        "Sequence",
        "SpecialToken",
        "Sequence",
        "SpecialToken",
    ]

    pair_keys = [next(iter(element.keys())) for element in pair_format]
    pair_values = [next(iter(element.values())) for element in pair_format]

    if pair_keys != structure_to_check:
        raise ValueError(
            f"Template processing pair format must have the following structure: {structure_to_check}"
        )

    # Make sure first and second special tokens are the same but different from the third special token
    first_special_token = pair_values[0]
    second_special_token = pair_values[2]
    third_special_token = pair_values[4]
    if first_special_token["id"] != second_special_token["id"]:
        raise ValueError("First and second special tokens must be the same")
    if (
        first_special_token["id"] == third_special_token["id"]
        or second_special_token["id"] == third_special_token["id"]
    ):
        raise ValueError(
            "First and second special tokens must be different from the third special token"
        )

    # Make sure special tokens are in the special tokens mapping
    special_tokens_ids = [
        special_token_map["id"] for special_token_map in special_tokens_mapping.values()
    ]
    for special_token in [
        first_special_token,
        second_special_token,
        third_special_token,
    ]:
        if special_token["id"] not in special_tokens_ids:
            raise ValueError(
                f"Special token {special_token['id']} not in special tokens mapping: {special_tokens_mapping}"
            )


def check_tokenizer_has_template_processing(tokenizer_json: dict) -> None:
    post_processor = tokenizer_json.get("post_processor")
    template_to_check = "TemplateProcessing"

    if not post_processor:
        raise ValueError("Tokenizer does not have a post_processor")

    if post_processor.get("type") == "Sequence":  # TemplateProcessing may be inside Sequence
        processors = post_processor.get("processors")
        for processor in processors:
            if processor.get("type") == template_to_check:
                special_tokens_mapping = processor["special_tokens"]
                pair = processor.get("pair")
                print(
                    f"Will use the following {template_to_check} pair post_processor:\n{YELLOW}{json.dumps(pair, indent=2, ensure_ascii=False)}{END}"
                )
                break
        else:
            raise ValueError(f"Tokenizer does not have a {template_to_check} post_processor")
    elif post_processor.get("type") == template_to_check:
        special_tokens_mapping = post_processor["special_tokens"]
        pair = post_processor.get("pair")
        print(
            f"Will use the following {template_to_check} pair post_processor:\n{YELLOW}{json.dumps(pair, indent=2, ensure_ascii=False)}{END}"
        )
    else:
        raise ValueError(f"Tokenizer does not have a {template_to_check} post_processor")

    check_template_proccessing_pair_format(pair, special_tokens_mapping)


def add_special_tokens(
    tokenizer: PreTrainedTokenizerFast, special_tokens: dict
) -> PreTrainedTokenizerFast:
    before_add_vocab_size = len(tokenizer.get_added_vocab())
    tokenizer.add_special_tokens(special_tokens)
    after_add_vocab_size = len(tokenizer.get_added_vocab())
    assert after_add_vocab_size == before_add_vocab_size, (
        f"Special tokens should already be in tokenizer, but added {after_add_vocab_size - before_add_vocab_size} token(s)"
    )

    return tokenizer


def prepare_tokenizer(tokenizer_config_path: str) -> PreTrainedTokenizerFast:
    tokens_to_include = ["eos_token", "pad_token", "cls_token"]
    with open(tokenizer_config_path) as f:
        tokenizer_config = yaml.safe_load(f)
    for token in tokens_to_include:
        if token not in tokenizer_config:
            raise ValueError(f"Tokenizer config does not have {token}")

    tokenizer_path = tokenizer_config.pop("tokenizer_path")
    yaml_path = Path(tokenizer_config_path).parent.resolve()
    tokenizer_path = str((yaml_path / tokenizer_path).resolve())

    with open(tokenizer_path) as f:
        tokenizer_json = json.load(f)

    added_tokens = tokenizer_json.get("added_tokens", [])
    check_special_tokens(added_tokens, tokenizer_config)
    print(
        f"Will use the following special tokens:\n{YELLOW}{json.dumps(tokenizer_config, indent=2, ensure_ascii=False)}{END}"
    )

    check_tokenizer_has_template_processing(tokenizer_json)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer = add_special_tokens(tokenizer, tokenizer_config)

    test_a = ["This is a test sentence", "This is another test sentence"]
    test_b = [
        "The quick brown fox jumps over the lazy dog",
        "Sandy sells seashells by the seashore",
    ]
    enc = tokenizer(test_a, test_b, return_tensors="pt", padding=True, pad_to_multiple_of=8)
    dec = tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=False)
    print(f"Example encoding with special tokens:\n{YELLOW}{dec}{END}")

    return tokenizer
