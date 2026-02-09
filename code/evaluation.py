#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation helpers for AI Clinician core pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ai_utils import soften_policy
from offpolicy import offpolicy_multiple_eval_010518


def build_qldata3(
    blocs: np.ndarray,
    idx: np.ndarray,
    actionbloc: np.ndarray,
    y90: np.ndarray,
    r2: np.ndarray,
    ptid: np.ndarray,
    abss: np.ndarray,
) -> np.ndarray:
    """
    Build qldata3 matrix for off-policy evaluation.
    Columns: bloc, state, action, reward, pi(s,a), b(s,a), optimal_action, ptid
    """
    qldata = np.column_stack([blocs, idx, actionbloc, y90, np.zeros(idx.shape[0]), r2, ptid])
    qldata3 = np.zeros((int(np.floor(qldata.shape[0] * 1.2)), 8))
    c = 0
    for i in range(qldata.shape[0] - 1):
        qldata3[c, :] = qldata[i, [0, 1, 2, 4, 6, 6, 6, 6]]
        c += 1
        if qldata[i + 1, 0] == 1:
            qldata3[c, :] = np.array([qldata[i, 0] + 1, abss[int(qldata[i, 3])], -1, qldata[i, 5], 0, 0, 0, qldata[i, 6]])
            c += 1
    return qldata3[:c, :]


def apply_policy_probabilities(
    qldata3: np.ndarray,
    physpol: np.ndarray,
    optimal_action: np.ndarray,
    nact: int,
    p: float,
    ncl: int,
) -> np.ndarray:
    """
    Add behavior and target policy probabilities to qldata3.
    """
    softpi = soften_policy(physpol, p)
    softb = np.full((ncl + 2, nact), p / 24.0)
    for s in range(0, ncl):
        softb[s, optimal_action[s]] = 1 - p

    for i in range(qldata3.shape[0]):
        if qldata3[i, 1] < ncl:
            s = int(qldata3[i, 1])
            a = int(qldata3[i, 2])
            if a >= 0:
                qldata3[i, 4] = softpi[s, a]
                qldata3[i, 5] = softb[s, a]
                qldata3[i, 6] = optimal_action[s]
    return qldata3


def evaluate_policy(
    qldata3: np.ndarray,
    physpol: np.ndarray,
    gamma: float,
    iter_ql: int,
    iter_wis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run off-policy evaluation (TD-learning + WIS).
    """
    return offpolicy_multiple_eval_010518(qldata3, physpol, gamma, 1, iter_ql, iter_wis)
