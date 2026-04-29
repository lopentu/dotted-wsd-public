# https://huggingface.co/docs/transformers/en/tasks/sequence_classification
import os
from typing import Annotated

import typer
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from dotted_wsd.dwsd_datasets import load_dotted_datasets_for_hf_trainer
from dotted_wsd.settings import BASE_DIR
from dotted_wsd.train.lora import get_lora_model
from dotted_wsd.train.utils import build_suffix, compute_accuracy, prepare_tokenizer

# IS_DEBUG = False

ID2LABEL = {0: "NO", 1: "YES"}
LABEL2ID = {val: key for key, val in ID2LABEL.items()}
CLASSIFICATION_CONFIG = {
    "num_labels": 2,
    "id2label": ID2LABEL,
    "label2id": LABEL2ID,
    "ignore_mismatched_sizes": True,
}


def main(
    model_id: Annotated[str, typer.Argument(help="The model ID of a model from the 🤗 Model Hub")],
    per_device_batch_size: Annotated[
        int,
        typer.Argument(
            help="The batch size per device. Both training and evaluation will use this batch size."
        ),
    ],
    report_to: Annotated[
        str,
        typer.Option(
            help="Where to report training metrics. Use 'none' for no reporting, 'wandb' for Weights & Biases, etc. When 'wandb' is selected, set the WANDB_ENTITY and WANDB_PROJECT env vars to control the destination."
        ),
    ] = "none",
    torch_dtype: Annotated[
        str,
        typer.Option(
            help="The torch dtype to load the model. This can be 'fp32' or 'bf16' or 'auto.",
            case_sensitive=True,
        ),
    ] = "auto",
    tokenizer_config_path: Annotated[
        str | None,
        typer.Option(
            help="The path to a yaml file containing the tokenizer config. If not provided, the tokenizer will be loaded from the model_id."
        ),
    ] = None,
    bf16: Annotated[
        bool, typer.Option(help="Train and evaluate using BF16 mixed-precision.")
    ] = True,
    use_lora: Annotated[
        bool,
        typer.Option(help="Train the model using Low Rank Adaptation. Useful for large models."),
    ] = False,
    lora_rank: Annotated[
        int | None,
        typer.Option(
            help="The rank of the Low Rank Adaptation. This is only used if use_lora is set to True."
        ),
    ] = None,
    lora_alpha: Annotated[
        int | None,
        typer.Option(
            help="The alpha value of the Low Rank Adaptation. This is only used if use_lora is set to True."
        ),
    ] = None,
    total_batch_size: Annotated[
        int,
        typer.Option(
            help="The total number of samples seen between weight updates. This will be divided by per_device_batch_size and the resulting value will be the gradient_accumulation_steps."
        ),
    ] = 512,
    suffix: Annotated[
        str,
        typer.Option(
            help="The suffix to append to the model ID. This is used to differentiate between different training runs and models."
        ),
    ] = "",
    push_to_hub: Annotated[
        bool,
        typer.Option(
            help="Push the trained model to the Hugging Face Hub. Requires write access to the namespace specified by --hub-model-id (defaults to the lopentu org)."
        ),
    ] = False,
    hub_model_id: Annotated[
        str | None,
        typer.Option(
            help="Repo ID on the Hugging Face Hub to push to. Only used when --push-to-hub is set. Defaults to 'lopentu/{model_id}-DottedWSD{suffix}'."
        ),
    ] = None,
    hub_private_repo: Annotated[
        bool,
        typer.Option(help="Create the hub repo as private. Only used when --push-to-hub is set."),
    ] = True,
    debug: Annotated[
        bool,
        typer.Option(
            help=(
                "Cap each of the four constituent splits "
                "(rp_train, rp_test, wsd_train, wsd_test) at 100 rows. "
                "Useful for end-to-end smoke tests; trains on 200 samples, "
                "evaluates on 200 samples."
            )
        ),
    ] = False,
):
    suffix = build_suffix(bf16) + suffix
    output_dir = str((BASE_DIR / "output" / model_id.replace("/", "-")).resolve())
    if hub_model_id is None:
        hub_model_id = f"lopentu/{model_id.replace('/', '-')}-DottedWSD{suffix}"
    wandb_run_name = f"{model_id.replace('/', '-')}{suffix}"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_ENTITY", "lopentu")
        os.environ.setdefault("WANDB_PROJECT", "dotted-wsd")
    gradient_accumulation_steps = total_batch_size // per_device_batch_size
    if tokenizer_config_path:
        tokenizer = prepare_tokenizer(tokenizer_config_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if use_lora:
        if not lora_rank or not lora_alpha:
            raise ValueError("lora_rank and lora_alpha must be provided if use_lora is True")
        model = get_lora_model(
            model_id,
            torch_dtype=torch_dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **CLASSIFICATION_CONFIG,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **CLASSIFICATION_CONFIG,
            # attn_implementation="flash_attention_2",
        )

    if not tokenizer.pad_token:
        logger.warning(
            f"Setting tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})"
        )
        tokenizer.pad_token = tokenizer.eos_token

    if not model.config.pad_token_id:
        logger.warning("Setting model.config.pad_token_id to tokenizer.pad_token_id")
        model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )  # 8 for bf16, 16 for fp32

    train_ds, eval_ds = load_dotted_datasets_for_hf_trainer(
        tokenizer=tokenizer,
        is_debug=debug,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=report_to,
        num_train_epochs=3,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=5,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        hub_model_id=hub_model_id,
        hub_private_repo=hub_private_repo,
        push_to_hub=push_to_hub,
        warmup_ratio=0.1,
        save_strategy="epoch",
        eval_strategy="epoch",
        # save_steps=
        # eval_steps=250,
        seed=42,
        run_name=wandb_run_name,
        optim="adamw_8bit",
        # torch_compile=True,
        bf16=bf16,
        bf16_full_eval=bf16,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_accuracy(),
    )

    trainer.train()
    if push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    typer.run(main)
