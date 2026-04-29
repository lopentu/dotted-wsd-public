import re
from collections.abc import Generator
from pathlib import Path

from CwnGraph import CwnImage
from pydantic import BaseModel
from typing_extensions import TypedDict

from dotted_wsd.dwsd_types import WsdInstanceFromCsv
from dotted_wsd.tagger.preprocessing import find_candidate_senses, get_target_word

idx_regex = re.compile(r"asbc_dotted_tagged_(?P<idx>\d\d\d)-of-140")


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
    label: int | None


class AsbcTaggedToken(BaseModel):
    """
    A line from `dt-asbc` is composed of `AsbcTaggedToken`s.
    """

    text: str
    tag: str


class WsdInstancesGroup(BaseModel):
    """
    A group of WSD instances that share the same probe. Generated from a line of text from `dt-asbc`.
    Each group should only have one probe and one instance marked as ground truth (label = 1).
    """

    ex_id: str
    target_word: str
    instances: list[WsdInstanceFromCsv]


class WeaklySupervisedLine(BaseModel):
    text: str
    tag: str


class AsbcTaggedLine(BaseModel):
    line: list[AsbcTaggedToken]
    line_idx: int

    def __str__(self):
        return " ".join(inst.text for inst in self.line)

    def create_weakly_supervised_line(self) -> list[WeaklySupervisedLine]:
        """
        Creates a list of weakly supervised instances by iterating through each
        `TaggedInstance` in the `line`. For each instance, it constructs a string
        where the current instance's text is enclosed in angle brackets `< >`, and
        is surrounded by the concatenated texts of the previous and following
        instances.

        Returns:
            list[str]: A list of strings, each representing a weakly supervised
            instance with the target word highlighted.
        """
        out = []
        for i in range(len(self.line)):
            center = self.line[i]
            left = self.line[:i]
            right = self.line[i + 1 :]
            center_text = f"<{center.text}>"
            left_text = "".join(inst.text for inst in left)
            right_text = "".join(inst.text for inst in right)
            joined = f"{left_text}{center_text}{right_text}"
            out.append(WeaklySupervisedLine(text=joined, tag=center.tag))
        return out

    def generate_wsd_instances(
        self,
        cwn: CwnImage,
        filter_by_length: bool = False,
        min_token_length: int = 23,
        ex_id_prefix: str = "",
    ) -> list[WsdInstancesGroup]:
        """
        Given a list of `TaggedLine`s, generates a list of `WsdInstancesGroup`s
        by first creating weakly supervised lines (adding angle brackets to the target word) and then generating a group of WSD
        instances for each weakly supervised line.

        Args:
            cwn: The CwnImage object containing the information of the Chinese
                WordNet.
            offset_ex_id: The offset for the example id of the generated WSD
                instances.
            minimum_token_length: The minimum length of the input text for
                generating WSD instances. Defaults to 23. This was derived from the mean length of each WSD training example.

        Returns:
            tuple[list[WsdInstancesGroup], int]: A tuple containing a list of
                `WsdInstancesGroup`s and the final example id.
        """
        line = "".join(inst.text for inst in self.line)
        if filter_by_length and (len(line) < min_token_length):
            return []

        weakly_supervised_line = self.create_weakly_supervised_line()
        # TODO: check for instances WSD instances first, then add ex_id
        wsd_instances_groups = []  # each group should have one target word

        for i, line in enumerate(weakly_supervised_line):
            ex_id = f"{ex_id_prefix},line={self.line_idx},token={i}" if ex_id_prefix else str(i)
            instances, target_word = self._generate_wsd_instances(
                cwn=cwn,
                input_text=line.text,
                ex_id=ex_id,
                ground_truth_tag=line.tag,
            )
            if not instances:
                # logger.warning(f"{target_word} only has one sense or no sense")
                continue
            group = WsdInstancesGroup(ex_id=ex_id, instances=instances, target_word=target_word)
            wsd_instances_groups.append(group)
        return wsd_instances_groups

    def _generate_wsd_instances(
        self,
        cwn: CwnImage,
        input_text: str,
        ground_truth_tag: str,
        ex_id: str,
        filter_no_sense_candidate_matches_ground_truth: bool = False,
        filter_multiple_candidate_matches_ground_truth: bool = False,
    ) -> tuple[list[WsdInstanceFromCsv] | None, str]:
        """
        Generate a list of WsdInstanceForInference given a tagged line, a CwnImage and an example id.
        May be empty if there are no candidate senses, e.g., COMMACATEGORY.

        Args:
            cwn (CwnImage): An instance of CwnImage to search for senses.
            input_text (str): The input text string with the target word marked.
            ex_id (int): The example id.

        Returns:
            list[WsdInstanceForInference]: A list of WsdInstanceForInference.
        """
        target_word = get_target_word(input_text)
        candidate_senses = find_candidate_senses(cwn=cwn, target_word=target_word, target_pos=None)
        if len(candidate_senses) < 2:
            # logger.error(f"{target_word} only has one sense or no sense")
            return None, target_word

        no_of_candidate_senses_match_ground_truth = sum(
            [int(x.id == ground_truth_tag) for x in candidate_senses]
        )
        if (
            filter_no_sense_candidate_matches_ground_truth
            and no_of_candidate_senses_match_ground_truth == 0
        ):
            # logger.error(f"{target_word} has no correct candidate sense")
            return None, target_word

        if (
            filter_multiple_candidate_matches_ground_truth
            and no_of_candidate_senses_match_ground_truth > 1
        ):
            # logger.error(f"{target_word} has multiple correct candidate senses")
            return None, target_word

        instances = []
        test_cwn = cwn.from_sense_id(ground_truth_tag)  # get the pos of the correct sense
        test_pos = test_cwn.pos
        test_definition = test_cwn.definition

        for sense_x in candidate_senses:
            avail_examples = [x for x in sense_x.all_examples() if x.strip()]
            instance = WsdInstanceFromCsv(
                example_id=ex_id,
                test_word=target_word,
                test_pos=test_pos,
                test_sense_id=ground_truth_tag,
                test_definition=test_definition,
                test_sentence=input_text,
                cwn_sense_id=sense_x.id,
                cwn_definition=sense_x.definition,
                cwn_sentence=avail_examples[0],
                label=int(ground_truth_tag == sense_x.id),
                source="dt-asbc",
            )
            # instance = WsdInstanceForInference(
            #     example_id=ex_id,
            #     example_type="wsd",
            #     target_word=target_word,
            #     probe=input_text,
            #     sense_id=sense_x.id,
            #     target_pos=None,
            #     cwn_pos=sense_x.pos,
            #     simplified_pos=None,
            #     sense_def=sense_x.definition,
            #     label=int(ground_truth_tag == sense_x.id),
            #     sense_refex=avail_examples[
            #         0
            #     ],  # reference example, selects the first example sentence for the sense
            # )
            instances.append(instance)

        return instances, target_word


class AsbcTaggedFile(BaseModel):
    filename: str
    file_idx: str
    lines: list[AsbcTaggedLine]

    @classmethod
    def from_file(cls, path: Path) -> "AsbcTaggedFile":
        with path.open() as f:
            lines = f.read().split("\n")

        file_idx = idx_regex.search(path.name)
        if not file_idx:
            raise ValueError("Invalid file name")
        file_idx = file_idx.group("idx")

        out = []
        for idx, line in enumerate(lines):
            instances = line.split()
            tagged_instances = []
            for instance in instances:
                text, tag = instance.split("-")
                tagged_instances.append(AsbcTaggedToken(text=text, tag=tag))
            out.append(AsbcTaggedLine(line_idx=idx, line=tagged_instances))

        return cls(filename=path.name, file_idx=f"file={file_idx}", lines=out)

    def __repr__(self) -> str:
        return f"Filename: {self.filename}\nFile Index: {self.file_idx}\nNo. of Lines: {len(self.lines)}"

    def __getitem__(self, index):
        return self.lines[index]

    def __len__(self):
        """
        Returns the number of lines in the file.
        """
        return len(self.lines)

    def gather_instances(self, cwn: CwnImage) -> Generator[WsdInstanceFromCsv, None, None]:
        for line in self.lines:
            for group in line.generate_wsd_instances(cwn=cwn, ex_id_prefix=self.file_idx):
                yield from group.instances
