import re
from functools import lru_cache

from CwnGraph import CwnImage

from .cwn_pos_map import cwn_pos_map

Word = str
Pos = str
Token = tuple[Word, Pos]


def is_pos_matched(in_pos, ref_pos):
    if in_pos in ref_pos:  # e.g., "V" in "VA"
        return True
    else:
        return cwn_pos_map.get(in_pos, "OTHER") == cwn_pos_map.get(
            ref_pos, "OTHER"
        )  # checks if in_pos and ref_pos are mapped to the same value in cwn_pos_map


def simplify_pos(pos):
    try:
        return cwn_pos_map[pos]
    except KeyError:
        poses = [x.strip(" ") for x in pos.split(",") if x.lower() != "nom"]
        if poses:
            return cwn_pos_map.get(poses[0], "OTHER")
        return "OTHER"


def make_input_text(tok_idx, sentence: list[Token]):
    words = [x[0] for x in sentence]
    words[tok_idx] = f"<{words[tok_idx]}>"
    return "".join(words)


def get_target_word(input_text: str) -> str:
    target_word = re.findall(r"<(.+?)>", input_text)
    if len(target_word) < 1:
        raise ValueError("There is no marked target in input_sentence")
    return target_word[0]


@lru_cache(maxsize=10000)
def find_candidate_senses(cwn: CwnImage, target_word: str, target_pos: str):
    """
    Find candidate senses for a given target word and part of speech (POS) from a CwnImage.

    Args:
        cwn (CwnImage): An instance of CwnImage to search for senses.
        target_word (str): The word for which to find candidate senses.
        target_pos (str): The part of speech (POS) of the target word. If empty, all senses with examples are considered.

    Returns:
        list: A list of candidate senses that match the target word and POS, and have example sentences.
    """
    senses = cwn.find_all_senses(target_word)
    if target_pos:
        candid_senses = []
        for sense_x in senses:
            avail_sentences = [x for x in sense_x.all_examples() if x.strip()]
            if (
                is_pos_matched(target_pos, sense_x.pos) and len(avail_sentences) > 0
            ):  # collect senses that have examples and have the same POS
                candid_senses.append(sense_x)
    else:
        candid_senses = [
            x for x in senses if len([ex for ex in x.all_examples() if ex.strip()]) > 0
        ]  # collect senses that have examples
    return candid_senses
