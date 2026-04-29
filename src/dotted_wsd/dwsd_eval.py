import os
import pickle
from os import cpu_count

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import typer
from loguru import logger
from rich.progress import track
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .dwsd_datasets import DataCollatorForDottedWsd, WsdDatasetForInstanceTesting
from .dwsd_types import (
    CompleteEvaluateRpOutput,
    CompleteEvaluateWsdOutput,
    EvalCategory,
    EvaluateWsdByExampleOutput,
    PredictionVsGround,
    ScoreWsdInstanceOutput,
    SingleEvaluateRpOutput,
    WsdByExamplePayload,
    WsdByInstancePayload,
    WsdExample,
)
from .settings import DATA_DIR
from .tagger.model import (
    DataParallelModelWrapper,
    DottedWsd,
    DottedWsdHf,
    DottedWsdTagger,
)

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _evaluate_wsd_by_instance(
    model: PreTrainedModel | DataParallel[PreTrainedModel],
    dataloader: DataLoader,
    device: str,
) -> ScoreWsdInstanceOutput:
    model.eval()
    all_logits = []
    all_labels = []
    all_exids = []
    eval_loss_vec = []
    for batch in track(dataloader, description="Evaluating WSD by Instance"):
        example_ids = batch.pop("example_ids").cpu().tolist()
        batch.to(device)
        with torch.no_grad():
            out = model(**batch)
        if isinstance(model, DataParallel):
            eval_loss = out["eval_loss"].detach().cpu()
            logits = out["logits"].detach().tolist()
        else:
            eval_loss = out.loss.detach().item()
            logits = out.logits.detach().cpu()
            if logits.shape[1] == 2:  # binary classification
                logits = logits[:, 1].tolist()  # get the positive class

        eval_loss_vec.append(eval_loss)
        labels = batch["labels"].cpu().tolist()
        all_logits.extend(logits)
        all_labels.extend(labels)
        all_exids.extend(example_ids)

    eval_out = _score_wsd_by_instance(
        np.array(all_logits), np.array(all_exids), np.array(all_labels)
    )
    print(f"Evaluation acc: {GREEN}{eval_out['accuracy']:.4f}{RESET}")
    return eval_out


def _score_wsd_by_instance(
    logits: npt.NDArray,
    example_ids: npt.NDArray,
    labels: npt.NDArray | None = None,
) -> ScoreWsdInstanceOutput:
    assert logits.ndim == example_ids.ndim == 1
    assert logits.shape == example_ids.shape
    if labels is not None:
        assert labels.shape == logits.shape
    else:
        labels = np.full(logits.shape[0], None)

    # groupby example_ids
    by_examples = {}
    for logit_x, exid, label_x in zip(logits, example_ids, labels, strict=False):
        ex_item = by_examples.setdefault(exid, {})
        ex_item.setdefault("logits", []).append(logit_x)
        ex_item.setdefault("labels", []).append(label_x)

    # return by_examples

    ##  compute by-example metric
    n_example = 0
    n_example_correct = 0
    example_preds = {}
    for ex_id, ex_item in by_examples.items():
        ex_logits = ex_item["logits"]
        ex_labels = ex_item["labels"]
        total_options = len(ex_logits)

        pred_idx = int(np.argmax(ex_logits))

        if all(x is not None for x in ex_labels):
            if sum(ex_labels) > 1:
                logger.warning("[WARN] more than one correct answer, skip example ", ex_id)
                continue
            if sum(ex_labels) == 0:
                logger.warning("[WARN] there is no correct answer, skip example ", ex_id)
                continue
            ex_labels = [int(x) for x in ex_labels]
            real_idx = ex_labels.index(1)
            n_example_correct += int(pred_idx == real_idx)
            n_example += 1
            example_preds[ex_id] = PredictionVsGround(
                prediction=pred_idx, ground=real_idx, total=total_options
            )  # (pred, real)
        else:
            example_preds[ex_id] = PredictionVsGround(
                prediction=pred_idx, ground=None, total=total_options
            )

    example_acc = n_example_correct / n_example if n_example > 0 else None

    return ScoreWsdInstanceOutput(
        accuracy=example_acc, examples_prediction_idx_vs_ground_idx=example_preds
    )


def _evaluate_wsd_by_example(
    model: DottedWsdTagger, wsd_examples_df: pd.DataFrame, use_pos_hint: bool
) -> EvaluateWsdByExampleOutput:
    predictions = []
    wsd_examples = [
        WsdExample(**ex)  # type: ignore
        for ex in wsd_examples_df.to_dict(orient="records")
    ]
    ref_labels = [ex.test_sense_id for ex in wsd_examples]

    for ex in track(
        wsd_examples,
        description=f"Evaluating WSD by Example {'With' if use_pos_hint else 'Without'} POS Hints",
    ):
        # By design: if `wsd_tag` fails on one example (e.g. CWN lookup miss,
        # tokenization edge case) we substitute a sentinel and continue, so a
        # single bad input doesn't abort the whole eval pass.
        try:
            input_text = ex.test_sentence
            pos_hint = ex.test_pos if use_pos_hint else None
            ex_pred, inst_preds = model.wsd_tag(input_text=input_text, hint=pos_hint)
            predictions.append((ex_pred.pred_class, ex_pred.prob, len(inst_preds)))
        except Exception:  # noqa: BLE001, PERF203
            predictions.append(("----", 0.0, 0))

    preds = [x[0] for x in predictions]
    probs = [x[1] for x in predictions]
    ncandid = [x[2] for x in predictions]

    acc = accuracy_score(ref_labels, preds)
    print(f"Accuracy {'With' if use_pos_hint else 'Without'} POS Hints: {GREEN}{acc:.4f}{RESET}")

    return EvaluateWsdByExampleOutput(
        accuracy=float(acc),
        predictions=preds,
        probabilities=probs,
        num_candid=ncandid,
    )


def expand_indices(pred_idx_vs_ground_idx: list[PredictionVsGround]) -> list[int]:
    """
    Expands a list of PredictionVsGround objects into a list of integers.

    Each PredictionVsGround object contains a prediction index and a total count.
    This function creates a list of zeros with the length of the total count for each object,
    sets the element at the prediction index to 1, and then concatenates these lists together.

    Args:
        pred_idx_vs_ground_idx (list[PredictionVsGround]): A list of PredictionVsGround objects.

    Returns:
        list[int]: A list of integers where each prediction index is expanded into a list of zeros
                   with a single one at the prediction index.
    """
    preds_expanded = []
    for item in pred_idx_vs_ground_idx:
        preds = [0] * item["total"]
        preds[item["prediction"]] = 1
        preds_expanded.extend(preds)

    return preds_expanded


def evaluate_wsd(
    dotted_model: DottedWsd | DottedWsdHf | DataParallel[DottedWsdHf] | DataParallel[DottedWsd],
    model: PreTrainedModel | DataParallel[DataParallelModelWrapper],
    model_name: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model_id: str,
    save_dir: Path,
    use_gpu: bool,
    preprocess_workers: int,
    remove_token_type_ids: bool = False,
    by_example_test_set_path: Path | None = None,
    by_instance_test_set_path: Path | None = None,
    evaluate_by_instance: bool = True,
    evaluate_by_example: bool = True,
    batch_size: int = 16,
    save_results: bool = True,
    debug: bool = False,
) -> CompleteEvaluateWsdOutput | None:
    logger.info("Evaluating WSD")
    save_dir = save_dir / "wsd"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{model_name}_wsd_eval.pkl"
    if not isinstance(dotted_model, (DottedWsd, DottedWsdHf, DataParallel)):
        raise TypeError(
            "Model must be an instance of DottedWsd, DottedWsdHf, or wrapped in DataParallel"
        )

    metadata = {
        "model_id": model_id,
        "by_instance_test_set_path": by_instance_test_set_path,
        "by_example_test_set_path": by_example_test_set_path,
        "test_category": "wsd",
    }

    tagger = DottedWsdTagger(
        model_id=model_id, model=dotted_model, tokenizer=tokenizer, use_gpu=use_gpu
    )

    # Test by Instances
    if evaluate_by_instance and by_instance_test_set_path is not None:
        test_set = WsdDatasetForInstanceTesting(
            by_instance_test_set_path,
            debug=debug,
            preprocess_workers=preprocess_workers,
        )
        collate_fn = DataCollatorForDottedWsd(
            tokenizer=tokenizer, remove_token_type_ids=remove_token_type_ids
        )
        dataloader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)
        by_instance_res = _evaluate_wsd_by_instance(
            model=model,
            dataloader=dataloader,
            device="cuda" if use_gpu else "cpu",
        )
        indices = list(by_instance_res["examples_prediction_idx_vs_ground_idx"].values())
        expanded_indices = expand_indices(indices)
        by_instance_df = pd.DataFrame(test_set.instances)
        by_instance_df["prediction"] = expanded_indices

        by_instance_payload = WsdByInstancePayload(
            accuracy=by_instance_res["accuracy"],
            df=by_instance_df,
        )
    else:
        by_instance_payload = None

    # Test by Examples
    if evaluate_by_example and by_example_test_set_path is not None:
        wsd_examples = pd.read_csv(by_example_test_set_path, index_col=0)
        if debug:
            wsd_examples = wsd_examples.head(500)

        with_pos_hints = _evaluate_wsd_by_example(
            model=tagger, wsd_examples_df=wsd_examples, use_pos_hint=True
        )
        without_pos_hints = _evaluate_wsd_by_example(
            model=tagger, wsd_examples_df=wsd_examples, use_pos_hint=False
        )

        wsd_examples_with_results = wsd_examples.assign(
            poshint_pred=with_pos_hints["predictions"],
            poshint_prob=with_pos_hints["probabilities"],
            poshint_ncandid=with_pos_hints["num_candid"],
            noposhint_pred=without_pos_hints["predictions"],
            noposhint_prob=without_pos_hints["probabilities"],
            noposhint_ncandid=without_pos_hints["num_candid"],
        )

        by_example_payload = WsdByExamplePayload(
            poshint_accuracy=with_pos_hints["accuracy"],
            noposhint_accuracy=without_pos_hints["accuracy"],
            df=wsd_examples_with_results,
        )

    else:
        by_example_payload = None

    complete_output = CompleteEvaluateWsdOutput(
        metadata=metadata,
        by_example=by_example_payload,
        by_instance=by_instance_payload,
    )

    if save_results:
        with open(save_path, "wb") as f:
            pickle.dump(complete_output, f)

    return complete_output


def _evaluate_rp(
    test_data: pd.DataFrame,
    ref_labels: pd.Series,
    save_dir: Path,
    model: DottedWsdTagger,
    model_name: str,
    use_dot_obj_hint: bool,
) -> SingleEvaluateRpOutput:
    predictions: list[tuple[str, float]] = []  # (pred_class, prob)
    for _, row in track(
        test_data.iterrows(),
        description=(f"Evaluating RP {'With' if use_dot_obj_hint else 'Without'} Hints"),
    ):
        input_text = row.Sentence
        dot_obj = row.dot_obj
        if use_dot_obj_hint:
            ex_pred, _ = model.dotted_tag(input_text=input_text, hint=dot_obj)
        else:
            ex_pred, _ = model.rp_tag(input_text=input_text)
        predictions.append((ex_pred.pred_class, ex_pred.prob))

    rp_labels = sorted(ref_labels.unique().tolist())
    pred_dotted = [p[0] for p in predictions]  # get the predicted class

    print(classification_report(ref_labels, pred_dotted, zero_division=0))

    class_report = classification_report(ref_labels, pred_dotted, output_dict=True, zero_division=0)
    class_report = pd.DataFrame(class_report).transpose()
    conf_matrix = confusion_matrix(ref_labels, pred_dotted)
    conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=rp_labels)
    conf_matrix_display.plot(xticks_rotation=45)

    plt.title(f"{model_name} RP Classification ({'With' if use_dot_obj_hint else 'Without'} Hints)")
    filename = f"{model_name}_{'with-hints' if use_dot_obj_hint else 'without-hints'}_rp_classification.png"
    plt.savefig(save_dir / filename, bbox_inches="tight")

    return SingleEvaluateRpOutput(classification_report=class_report, predictions=predictions)


def evaluate_rp(
    model_id: str,
    model_name: str,
    model: DottedWsd | DottedWsdHf | DataParallel[DottedWsd] | DataParallel[DottedWsdHf],
    save_dir: Path,
    test_set_path: Path,
    use_gpu: bool,
    save_results: bool = True,
    debug: bool = False,
) -> CompleteEvaluateRpOutput | None:
    logger.info("Evaluating RP")
    save_dir = save_dir / "rp"
    save_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(model, (DottedWsd, DottedWsdHf)):
        raise TypeError("Model must be an instance of DottedWsd or DottedWsdHf")
    tagger = DottedWsdTagger(model_id, model=model, use_gpu=use_gpu)

    rp_valid = pd.read_csv(test_set_path)
    rp_mask = rp_valid.apply(
        lambda r: r["RP Class"] in r["dot_obj"], axis=1
    )  # checks if the RP Class is in the dot_obj, e.g., location in location*organization
    rp_valid = rp_valid.loc[rp_mask]
    if debug:
        rp_valid = rp_valid.head(100)
    ref_labels = rp_valid["RP Class"]
    # rp_labels = sorted(ref_labels.unique().tolist())

    hinted_eval = _evaluate_rp(
        test_data=rp_valid,
        ref_labels=ref_labels,
        save_dir=save_dir,
        model=tagger,
        model_name=model_name,
        use_dot_obj_hint=True,
    )

    all_eval = _evaluate_rp(
        test_data=rp_valid,
        ref_labels=ref_labels,
        save_dir=save_dir,
        model=tagger,
        model_name=model_name,
        use_dot_obj_hint=False,
    )

    combined_df = pd.concat(
        [hinted_eval["classification_report"], all_eval["classification_report"]],
        axis=1,
        keys=["Hint", "NoHint"],
    )

    metadata = {
        "model_id": model_id,
        "test_set_path": test_set_path,
        "test_category": "rp",
    }
    payload = {
        "metadata": metadata,
        "hint": hinted_eval,
        "nohint": all_eval,
        "df": combined_df,
    }

    save_path = save_dir / f"{model_name}_rp_eval.pkl"
    if save_results:
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

    return CompleteEvaluateRpOutput(hint=hinted_eval, nohint=all_eval)


def get_device(use_gpu: bool = True, use_data_parallel: bool = False) -> tuple[str, bool]:
    """
    Determines the device to use for computation.

    Args:
      use_gpu: Whether to use a GPU if available.
      use_data_parallel: Whether to use data parallelism if multiple GPUs are available.

    Returns:
      A tuple containing the device string ("cuda" or "cpu") and a boolean
      indicating whether to use data parallelism.
    """
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1 and use_data_parallel:
            logger.info(f"Found {torch.cuda.device_count()} GPUs. Using data parallelism.")
            data_parallel = True
        else:
            data_parallel = False
    else:
        device = "cpu"
        data_parallel = False
        print("Warning: Using CPU for evaluation")
    return device, data_parallel


def main(
    model_id: Annotated[str, typer.Argument(help="Model ID from Hugging Face Model Hub")],
    eval_category: Annotated[EvalCategory, typer.Argument(help="Evaluation category.")],
    use_gpu: Annotated[bool, typer.Option(help="Use GPU or not")] = True,
    use_data_parallel: Annotated[
        bool,
        typer.Option(
            help="Run evaluation using data parallelism with multiple GPUs where each GPU has its own copy of the model."
        ),
    ] = True,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 16,
    preprocess_workers: Annotated[
        int,
        typer.Option(help="Number of workers for preprocessing."),
    ] = cpu_count() or 16,
    save_dir: Annotated[
        Path,
        typer.Option(
            help="Directory to save evaluation results",
        ),
    ] = DATA_DIR / "eval_results",
    wsd_by_example_test_set_path: Annotated[
        Path,
        typer.Option(help="Path to the WSD test set organized by examples", exists=True),
    ] = DATA_DIR / "wsd_examples.csv",
    load_in_4bit: Annotated[bool, typer.Option(help="Load model in 4-bit precision")] = False,
    wsd_by_instance_test_set_path: Annotated[
        Path,
        typer.Option(
            help="Path to the WSD test set with examples flattened into instances",
            exists=True,
        ),
    ] = DATA_DIR / "WSD_merge_test_v2.csv",
    remove_token_type_ids: Annotated[
        bool,
        typer.Option(help="Remove token_type_ids from input if your model does not use them"),
    ] = False,
    evaluate_wsd_by_example: Annotated[bool, typer.Option(help="Evaluate WSD by example")] = True,
    evaluate_wsd_by_instance: Annotated[bool, typer.Option(help="Evaluate WSD by instance")] = True,
    rp_test_set_path: Annotated[
        Path,
        typer.Option(help="Path to the RP test set", exists=True),
    ] = DATA_DIR / "RP_valid.csv",
    debug: Annotated[bool, typer.Option(help="Debug mode")] = False,
):
    device, use_data_parallel = get_device(use_gpu, use_data_parallel)

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype="auto")
    dotted_model = DottedWsdHf(model_or_model_id=model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert tokenizer.pad_token_id is not None
    if not model.config.pad_token_id:
        logger.info(f"Setting model's pad_token_id to tokenizer's: {tokenizer.pad_token_id}")
        model.config.pad_token_id = tokenizer.pad_token_id
    model_name = model_id.split("/")[-1]

    if use_data_parallel:
        logger.info("Wrapping model for DataParallel")
        model = DataParallelModelWrapper(model)
        model = DataParallel(model)
    model = model.to(device)

    save_dir = save_dir / model_name
    print(f"{YELLOW}****{RESET} {BOLD}{GREEN}{model_name}{BOLD}{RESET} {YELLOW}****{RESET}")

    if eval_category == EvalCategory.WSD:
        evaluate_wsd(
            dotted_model=dotted_model,
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            model_id=model_id,
            by_example_test_set_path=wsd_by_example_test_set_path,
            by_instance_test_set_path=wsd_by_instance_test_set_path,
            preprocess_workers=preprocess_workers,
            evaluate_by_example=evaluate_wsd_by_example,
            evaluate_by_instance=evaluate_wsd_by_instance,
            remove_token_type_ids=remove_token_type_ids,
            batch_size=batch_size,
            save_dir=save_dir,
            save_results=True,
            use_gpu=use_gpu,
            debug=debug,
        )
    elif eval_category == EvalCategory.RP:
        evaluate_rp(
            model_id=model_id,
            model_name=model_name,
            model=dotted_model,
            save_dir=save_dir,
            test_set_path=rp_test_set_path,
            use_gpu=use_gpu,
            debug=debug,
        )
    elif eval_category == EvalCategory.ALL:
        evaluate_wsd(
            dotted_model=dotted_model,
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            preprocess_workers=preprocess_workers,
            by_example_test_set_path=wsd_by_example_test_set_path,
            by_instance_test_set_path=wsd_by_instance_test_set_path,
            remove_token_type_ids=remove_token_type_ids,
            evaluate_by_example=evaluate_wsd_by_example,
            evaluate_by_instance=evaluate_wsd_by_instance,
            batch_size=batch_size,
            model_id=model_id,
            save_dir=save_dir,
            save_results=True,
            use_gpu=use_gpu,
            debug=debug,
        )
        evaluate_rp(
            model_id=model_id,
            model_name=model_name,
            model=dotted_model,
            save_dir=save_dir,
            test_set_path=rp_test_set_path,
            use_gpu=use_gpu,
            debug=debug,
        )
        # print(wsd_res, rp_res)


if __name__ == "__main__":
    typer.run(main)
