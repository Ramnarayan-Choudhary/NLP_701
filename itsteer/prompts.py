# itsteer/prompts.py
from dataclasses import dataclass
from typing import List

CLASS_LABELS = ["VALID", "INVALID"]

TEMPLATE = (
    "You are given a syllogism. Decide whether the conclusion follows from the premises by formal logic ALONE.\n"
    "Ignore plausibility and world knowledge. Answer with exactly one of: VALID or INVALID.\n\n"
    "Syllogism:\n{syllogism}\n\nAnswer: "
)

@dataclass
class Prompt:
    text: str
    labels: List[str]

def build_prompt(syllogism: str) -> Prompt:
    return Prompt(text=TEMPLATE.format(syllogism=syllogism), labels=CLASS_LABELS)