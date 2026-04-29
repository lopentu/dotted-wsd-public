from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .tagger_types import RpInsanceForInference, WsdInstanceForInference


@dataclass
class ExamplePrediction:
    prob: float
    instance: WsdInstanceForInference | RpInsanceForInference

    def __repr__(self):
        return f"<ExamplePrediction: {self.prediction()}>"

    @property
    def pred_class(self):
        inst = self.instance
        if inst.get("example_type") == "rp":
            return inst.get("typeclass_en", "--")
        else:
            return inst.get("sense_id", "----")

    def prediction(self):
        inst = self.instance
        if inst.get("example_type") == "rp":
            return "[RP:{}] {} ({:.4f})".format(
                inst.get("typeclass_en", "--"),
                inst.get("typeclass_gloss_zh", "--"),
                self.prob,
            )
        else:
            return "[{}] {} ({:.4f})".format(
                inst.get("sense_id", "----"), inst.get("sense_def", "----"), self.prob
            )


class InstancePredictions:
    """
    InstancePredictions class holds the predictions for each instance in a given example.
    For a Regular Polysemy example with 3 possible classes (organization, location, human), the InstancePredictions object would contain 3 predictions, one for each class.
    Attributes:
        probs (npt.NDArray[np.float64]): An array of probabilities for each class.
        ex_preds (list[ExamplePrediction]): A list of ExamplePrediction objects, each containing a probability and its corresponding instance.
    Methods:
        __init__(probs: npt.NDArray[np.float64], instances: list[WsdInstanceForInference] | list[RpInsanceForInference]):
            Initializes the InstancePredictions with probabilities and instances.
        __repr__() -> str:
            Returns a string representation of the InstancePredictions object.
        __len__() -> int:
            Returns the number of example predictions.
        predictions() -> list:
            Returns a list of predictions for each instance.
        top() -> ExamplePrediction:
            Returns the top prediction based on the highest probability.
        top_k(k=1) -> list[ExamplePrediction]:
            Returns the top k predictions based on the highest probabilities.
    """

    """Contains all instance predictions for one example. For an example with Regular Polysemy with 3 possible classes (organization, location, human), the instance predictions would contain 3 predictions, one for each class."""

    def __init__(
        self,
        probs: npt.NDArray[np.float64],
        instances: list[WsdInstanceForInference] | list[RpInsanceForInference],
    ):
        self.probs = probs
        self.ex_preds = []
        for prob_x, inst_x in zip(probs, instances, strict=False):
            self.ex_preds.append(
                ExamplePrediction(prob_x, inst_x)  # joins a probability for a class with that class
            )

    def __repr__(self):
        return f"<InstancePredictions: {len(self.probs)} class(es)>"

    def __len__(self):
        return len(self.ex_preds)

    def predictions(self):
        return [pred_x.prediction for pred_x in self.ex_preds]

    def top(self) -> ExamplePrediction:
        return self.top_k()[0]

    def top_k(self, k=1) -> list[ExamplePrediction]:
        sorted_idxs = np.argsort(-self.probs)
        return [self.ex_preds[i] for i in sorted_idxs[:k]]
