import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass
class Example:
    id: str
    syllogism: str
    validity: bool | None
    plausibility: bool | None

class SemevalDataset:
    def __init__(self, path: str):
        self.path = path
        # Expect either a JSONL file or a directory with train.jsonl (and dev/test later)
        if os.path.isdir(path):
            cand = [os.path.join(path, "train.jsonl"), os.path.join(path, "train.json")]
            for c in cand:
                if os.path.exists(c):
                    path = c
                    break
        self.examples = self._read_any(path)
    
    def _read_any(self, path: str) -> List[Example]:
        exs = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    exs.append(Example(
                        id=str(j.get("id")),
                        syllogism=j["syllogism"],
                        validity=j.get("validity", None),
                        plausibility=j.get("plausibility", None),
                    ))
        elif path.endswith(".json"):
            data = json.load(open(path, "r", encoding="utf-8"))
            for j in data:
                exs.append(Example(
                    id=str(j.get("id")),
                    syllogism=j["syllogism"],
                    validity=j.get("validity", None),
                    plausibility=j.get("plausibility", None),
                ))
        else:
            raise ValueError(f"Unsupported data file: {path}")
        return exs

    def train_valid_split(self, seed: int = 42, valid_frac: float = 0.2) -> Tuple[list[Example], list[Example]]:
        """Stratified split on (validity, plausibility) so runs stay comparable across users."""
        if not 0 < valid_frac < 1:
            raise ValueError(f"valid_frac must be between 0 and 1, got {valid_frac}")

        train_ratio = 1.0 - valid_frac
        rng = random.Random(seed)
        buckets: dict[tuple[bool, bool], list[Example]] = defaultdict(list)
        for ex in self.examples:
            key = (bool(ex.validity), bool(ex.plausibility))
            buckets[key].append(ex)

        train, valid = [], []
        for key in sorted(buckets.keys()):
            group = buckets[key]
            if not group:
                continue
            rng.shuffle(group)
            n_train = max(1, int(len(group) * train_ratio))
            train.extend(group[:n_train])
            valid.extend(group[n_train:])

        rng.shuffle(train)
        rng.shuffle(valid)
        return train, valid


def stratified_split_examples(examples: List[Example], train_ratio: float, seed: int = 42) -> Tuple[list[Example], list[Example]]:
    """Standalone stratified split helper mirroring split_dataset.py logic."""
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    rng = random.Random(seed)
    buckets: dict[tuple[bool, bool], list[Example]] = defaultdict(list)
    for ex in examples:
        buckets[(bool(ex.validity), bool(ex.plausibility))].append(ex)

    train, test = [], []
    for key in sorted(buckets.keys()):
        group = buckets[key]
        if not group:
            continue
        rng.shuffle(group)
        n_train = max(1, int(len(group) * train_ratio))
        train.extend(group[:n_train])
        test.extend(group[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
