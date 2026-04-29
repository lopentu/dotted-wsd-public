# https://huggingface.co/docs/peft/en/package_reference/lora
# https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora#load-and-prepare-a-model
# https://huggingface.co/docs/peft/main/en/developer_guides/quantization#quantize-a-model
# https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
import torch
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def get_lora_model(
    model_id: str, torch_dtype: str, lora_rank: int, lora_alpha: int, **kwargs
) -> PeftModel | PeftMixedModel:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        target_modules="all-linear",
        lora_alpha=lora_alpha,
        use_rslora=True,
        init_lora_weights="gaussian",
        lora_dropout=0.1,
        inference_mode=False,
        modules_to_save=["score"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, **kwargs, torch_dtype=torch_dtype
    )
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_id, quantization_config=quantization_config, **kwargs, torch_dtype="auto"
    # )
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config=peft_config)
    model.print_trainable_parameters()
    return model
