from __future__ import annotations
from typing import List, Dict, Tuple

def accuracy(y_true: list[bool], y_pred: list[bool]) -> float:
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / max(1, len(y_true))

def content_effect_metrics(y_true: list[bool], y_pred: list[bool], plaus: list[bool]) -> dict:
    # Intra-Plausibility: diff in accuracy between valid and invalid given a plausibility value.
    # Cross-Plausibility: diff in accuracy between plausible and implausible given formal validity.
    import numpy as np
    import itertools as it

    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    plaus = np.array(plaus, dtype=bool)

    # helper
    def acc(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0.0
        return (y_true[idx] == y_pred[idx]).mean()

    intra_vals = []
    for p in [False, True]:
        m_valid = (plaus == p) & (y_true == True)
        m_invalid = (plaus == p) & (y_true == False)
        intra_vals.append(abs(acc(m_valid) - acc(m_invalid)))
    intra = float(sum(intra_vals) / len(intra_vals))

    cross_vals = []
    for v in [False, True]:
        m_plaus = (y_true == v) & (plaus == True)
        m_impl = (y_true == v) & (plaus == False)
        cross_vals.append(abs(acc(m_plaus) - acc(m_impl)))
    cross = float(sum(cross_vals) / len(cross_vals))

    total = 0.5 * (intra + cross)
    return {
        "intra_plausibility": intra,
        "cross_plausibility": cross,
        "total_content_effect": total,
    }
