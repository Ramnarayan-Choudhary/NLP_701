import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import random

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
        rng = random.Random(seed)
        exs = self.examples[:]
        rng.shuffle(exs)
        n_valid = int(len(exs) * valid_frac)
        return exs[n_valid:], exs[:n_valid]
