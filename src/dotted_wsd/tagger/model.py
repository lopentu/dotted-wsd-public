import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from CwnGraph import CwnImage
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from dotted_wsd.dwsd_datasets import DataCollatorForDottedWsd, DottedWsdDataset
from dotted_wsd.dwsd_types import DottedWsdModelOutput
from dotted_wsd.tagger.prediction import ExamplePrediction, InstancePredictions

from .preprocessing import (
    Token,
    find_candidate_senses,
    get_target_word,
    make_input_text,
)
from .tagger_types import (
    ExampleId,
    RpInsanceForInference,
    TokenId,
    WsdInstanceForInference,
)


class DataParallelModelWrapper(nn.Module):
    def __init__(self, model, remove_token_type_ids: bool = False):
        super().__init__()
        self.model = model
        self.remove_token_type_ids = remove_token_type_ids

    def forward(self, **kwargs):
        self.model.eval()
        if self.remove_token_type_ids:
            kwargs.pop("token_type_ids", None)
        with torch.no_grad():
            out = self.model(**kwargs)
        # eval_loss = out.loss.detach()
        eval_loss = out.loss
        logits = out.logits
        if logits.shape[1] == 2:  # binary classification
            logits = logits[:, 1]  # get the positive class; can't do tolist() here

        return {
            "eval_loss": eval_loss,
            "logits": logits,
        }


class DottedWsd(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, torch_dtype="auto")
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()  # Use cross-entropy for classification

    def forward(
        self,
        input_ids,
        example_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).squeeze()

        loss = self.loss_func(logits, labels) if labels is not None else None

        return DottedWsdModelOutput(
            loss=loss, logits=logits, example_ids=example_ids, labels=labels
        )


class DottedWsdHf(nn.Module):
    def __init__(
        self,
        model_or_model_id: str | PreTrainedModel,
        remove_token_type_ids: bool = False,
    ):
        super().__init__()
        if isinstance(model_or_model_id, str):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_or_model_id, torch_dtype="auto"
            )
        else:
            self.model = model_or_model_id
        self.remove_token_type_ids = remove_token_type_ids

    def forward(self, example_ids, **kwargs):
        self.model.eval()
        if self.remove_token_type_ids:
            kwargs.pop("token_type_ids", None)

        outputs = self.model(return_dict=True, **kwargs)
        logits = outputs.logits
        if logits.shape[1] == 2:
            logits = logits[:, 1]  # get the positive class
        loss = outputs.loss
        labels = kwargs.get("labels")

        return DottedWsdModelOutput(
            loss=loss, logits=logits, example_ids=example_ids, labels=labels
        )


class DottedWsdTagger:
    def __init__(
        self,
        model_id: str,
        model: DottedWsd | DottedWsdHf | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        pretrained_model_path: str | None = None,
        data_collator: DataCollatorForDottedWsd | None = None,
        remove_token_type_ids: bool = False,
        use_gpu: bool = True,
        cwn_image_ver: str = "v.2022.08.01",
    ):
        self.device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        self.tokenizer = tokenizer
        self.cwn = CwnImage.load(cwn_image_ver)
        self.gloss_dict = self.load_gloss_dict()
        self.tokenizer = self.load_tokenizer(model_id) if tokenizer is None else tokenizer
        self.model = model if model else self.load_model(model_id, pretrained_model_path)
        self.data_collator = (
            data_collator
            if data_collator
            else DataCollatorForDottedWsd(
                self.tokenizer, remove_token_type_ids=remove_token_type_ids
            )
        )

    def is_transformers_classifier(self, config: PretrainedConfig) -> bool:
        if any(a.endswith("ForSequenceClassification") for a in config.architectures):
            logger.info("Detected AutoModelForSequenceClassification model")
            return True
        return False

    def load_gloss_dict(self) -> dict:
        with open(Path(__file__).parent / "glossdict.json", encoding="UTF-8") as fin:
            return json.load(fin)

    def load_tokenizer(self, model_id: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not tokenizer.pad_token:
            logger.warning(
                f"pad_token needs to be set for batched inference. Setting tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})"
            )
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_model(
        self,
        model_id: str,
        pretrained_model_path: str | None = None,
    ) -> DottedWsd | DottedWsdHf:
        config = AutoConfig.from_pretrained(model_id)
        if self.is_transformers_classifier(config):
            model = DottedWsdHf(model_id)
        else:
            assert pretrained_model_path is not None, (
                "pretrained_model_path is required for non Hugging Face AutoModelForSequenceClassification models"
            )
            model = DottedWsd(model_id)
            model.load_state_dict(torch.load(pretrained_model_path, weights_only=True))
        model.to(self.device)
        return model

    def sense_tag(self, sentences: list[list[Token]], show_progress=False):
        if isinstance(sentences[0][0], str):
            raise TypeError("Expecting a list of sentences as input")

        tagged_outputs = []
        for sent_x in sentences:
            out = self.sense_tag_per_sentence(sent_x, show_progress)
            tagged_outputs.append(out)
        return tagged_outputs

    def sense_tag_per_sentence(self, sentence: list[Token], show_progress=False):
        exid_maps: dict[TokenId, ExampleId] = {}

        # collect instances
        all_instances = []
        pred_maps: dict[TokenId, ExamplePrediction] = {}

        for tok_i, (_word, pos) in enumerate(sentence):
            input_text = make_input_text(tok_i, sentence)
            example_id = len(exid_maps)
            exid_maps[TokenId(tok_i)] = ExampleId(example_id)
            instances = self.generate_wsd_instances(input_text, example_id, target_pos=pos)
            if len(instances) == 0 and pos in ("Nb", "Nc"):
                rp_instances = self.generate_rp_instances(input_text, example_id)
                instances.extend(rp_instances)

            if len(instances) == 1:
                pred_maps[tok_i] = ExamplePrediction(1.0, instances[0])
            else:
                all_instances.extend(instances)

        # model inference
        logits = self.predict(all_instances, show_progress=show_progress)
        predictions, _ = self.decode_examples(all_instances, logits)
        pred_maps.update(predictions)

        # prepare output
        out = []
        for tok_i, tok in enumerate(sentence):
            prediction = pred_maps[tok_i].prediction() if tok_i in pred_maps else ""
            out.append((*tok, prediction))
        return out

    def wsd_tag(
        self, input_text, hint: str | None = None
    ) -> tuple[ExamplePrediction, InstancePredictions]:
        instances = self.generate_wsd_instances(input_text, ex_id=1, target_pos=hint)
        logits = self.predict(instances)
        predictions, by_example = self.decode_examples(instances, logits)

        return (
            predictions[1],
            by_example[1],
        )  # this is a dictionary, not a list, so we select by ex_id, which is the integer 1

    def rp_tag(
        self, input_text, hint: str | None = None
    ) -> tuple[ExamplePrediction, InstancePredictions]:
        instances = self.generate_rp_instances(input_text, ex_id=1, rp_type=hint)
        logits = self.predict(instances)
        predictions, by_example = self.decode_examples(instances, logits)

        return (
            predictions[1],
            by_example[1],
        )  # this is a mapping, not a list, so we select by ex_id

    def dotted_tag(
        self, input_text: str, hint: str | None = None
    ) -> tuple[ExamplePrediction, InstancePredictions]:
        if hint is None:
            instances = self.generate_wsd_instances(input_text, 1, hint)
            if not instances:
                instances = self.generate_rp_instances(input_text, 1, hint)
        elif "*" in hint:  # dot object
            instances = self.generate_rp_instances(input_text, 1, hint)
        else:
            instances = self.generate_wsd_instances(input_text, 1, hint)

        logits = self.predict(instances)
        predictions, by_example = self.decode_examples(instances, logits)

        return (
            predictions[1],
            by_example[1],
        )  # this is a mapping, not a list, so we select by ex_id

    def generate_wsd_instances(
        self, input_text: str, ex_id: int, target_pos: str | None = None
    ) -> list[WsdInstanceForInference]:
        # check whether target_word is in input_sentence
        target_word = get_target_word(input_text)

        # find candidate senses given the word and pos
        candid_senses = find_candidate_senses(self.cwn, target_word, target_pos)

        # generate WSD instances
        instances: list[WsdInstanceForInference] = []
        for sense_x in candid_senses:
            avail_examples = [x for x in sense_x.all_examples() if x.strip()]
            instance = WsdInstanceForInference(
                example_id=ex_id,
                example_type="wsd",
                target_word=target_word,
                probe=input_text,
                sense_id=sense_x.id,
                target_pos=target_pos,
                cwn_pos=sense_x.pos,
                simplified_pos=target_pos,
                sense_def=sense_x.definition,
                sense_refex=avail_examples[
                    0
                ],  # reference example, selects the first example sentence for the sense
            )
            instances.append(instance)

        return instances

    def generate_rp_instances(
        self, input_text: str, ex_id: int, rp_type: str | None = None
    ) -> list[RpInsanceForInference]:
        # check whether target_word is in input_sentence
        target_word = get_target_word(input_text)

        # get candidate dotted-types
        if rp_type:
            candid_types = [
                (x, self.gloss_dict[x])
                for x in rp_type.split(
                    "*"
                )  # limits the candidates to a type class, e.g., if RP class is location and its dot_obj is location*organization, then the candidate types are location and organization
                if x in self.gloss_dict
            ]
        else:
            candid_types = list(
                self.gloss_dict.items()
            )  # if no RP class is provided, then the model must pick from all RP classes

        instances: list[RpInsanceForInference] = []
        for type_en, gloss in candid_types:
            instance = RpInsanceForInference(
                example_id=ex_id,
                example_type="rp",
                target_word=target_word,
                probe=input_text,
                typeclass_en=type_en,
                typeclass_zh=gloss["zh_trans"],
                typeclass_gloss_zh=gloss["zh_gloss"],
            )
            instances.append(instance)

        return instances

    def predict(
        self,
        instances: list[WsdInstanceForInference] | list[RpInsanceForInference],
        batch_size: int = 16,
        show_progress=False,
    ) -> npt.NDArray[np.float64]:
        """
        Predicts the logits for a list of WSD or RP instances.

        Args:
            instances (list[WsdInstanceForInference] | list[RpInsanceForInference]):
                A list of instances for which predictions are to be made.
            batch_size (int, optional):
                The number of instances to process in each batch. Defaults to 16.
            show_progress (bool, optional):
                Whether to display a progress bar during prediction. Defaults to False.

        Returns:
            np.ndarray:
                An array of logits corresponding to the predictions of input instances.
        """
        model = self.model
        dataset = DottedWsdDataset(instances)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=self.data_collator
        )

        if show_progress:
            loader = tqdm(iter(loader))

        all_logits = []
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                out = model(**batch)
                logits = out.logits.detach().tolist()
                all_logits.extend(logits)
        assert len(all_logits) == len(instances)
        return np.array(all_logits)

    def decode_examples(
        self,
        instances: list[WsdInstanceForInference] | list[RpInsanceForInference],
        logits: npt.NDArray[np.float64],
    ) -> tuple[dict[ExampleId, ExamplePrediction], dict[ExampleId, InstancePredictions]]:
        """
        Decodes the given instances and their corresponding logits into example-level and instance-level predictions.

        Args:
            instances (list[WsdInstanceForInference] | list[RpInsanceForInference]):
                A list of instances for inference, either WSD (Word Sense Disambiguation) or RP (Role Prediction).
            logits (npt.NDArray[np.float64]):
                A numpy array of logits where each logit corresponds to a prediction for a single class.

        Returns:
            tuple[dict[ExampleId, ExamplePrediction], dict[ExampleId, InstancePredictions]]:
                - A dictionary mapping example IDs to their corresponding example-level predictions.
                - A dictionary mapping example IDs to their corresponding instance-level predictions.
        """
        # each logit corresponds to a prediction for a single class, e.g., if there are 3 possible classes for RP (organization, location, human), then there are 3 logits
        # ex_item["logits"] contains the logits for each class for the example
        # ex_item["instances"] contains the instances for the example, e.g.,
        # [CLS]他最近為了〈哈佛〉學費...[SEP]哈佛:機構,泛指機關團體...。[SEP] (True)
        # [CLS]他最近為了〈哈佛〉學費...[SEP]哈佛:地點,所在的地方。[SEP] (False)
        # [CLS]他最近為了〈哈佛〉學費...[SEP]哈佛:人類,人的總稱。[SEP] (False)

        # groupby example_ids
        by_examples = {}
        for inst_x, logit_x in zip(instances, logits, strict=False):
            ex_id = inst_x["example_id"]
            ex_item = by_examples.setdefault(ex_id, {})
            ex_item.setdefault("logits", []).append(logit_x)
            ex_item.setdefault("instances", []).append(inst_x)

        ##  compute by-example metric
        example_pred_map: dict[ExampleId, ExamplePrediction] = {}
        inst_preds_map: dict[ExampleId, InstancePredictions] = {}
        for ex_id, ex_item in by_examples.items():
            ex_logits = ex_item["logits"]
            ex_insts = ex_item["instances"]
            ex_probs = np.exp(ex_logits) / np.exp(ex_logits).sum()  # softmax
            inst_preds = InstancePredictions(ex_probs, ex_insts)

            inst_preds_map[ex_id] = inst_preds
            example_pred_map[ex_id] = (
                inst_preds.top()
            )  # get the top prediction from the possible classes

        return example_pred_map, inst_preds_map
