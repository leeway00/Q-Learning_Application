#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core helpers for AI Clinician pipeline (non-MDP, non-evaluation).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import rankdata

import mdptoolbox.mdp as mdp

def build_actions(
    reformat5: np.ndarray,
    iol: int,
    vcl: int,
    train_mask: np.ndarray,
    include_eicu: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Build discrete actions (5x5 bins) for fluids and vasopressors.

    Returns:
    - actionbloc: action id per row (0-based)
    - actionbloctrain: action id per training row (0-based)
    If include_eicu=True, also returns:
    - unique_values_dose: median dose per action (vaso, fluid)
    - io: fluid bin per row (1..5)
    - vc: vaso bin per row (1..5)
    - ma1: median fluid dose per fluid bin
    - ma2: median vaso dose per vaso bin
    """
    # IV fluids (input_4hourly)
    a = reformat5[:, iol]
    a = rankdata(a[a > 0]) / float(len(a[a > 0]))
    iof = np.floor((a + 0.2499999999) * 4)
    a = reformat5[:, iol]
    a = np.where(a > 0)[0]
    io = np.ones((reformat5.shape[0],), dtype=np.int64)
    io[a] = iof.astype(np.int64) + 1

    # Vasopressors (max_dose_vaso)
    vc = reformat5[:, vcl].copy()
    vcr = rankdata(vc[vc != 0]) / float(np.count_nonzero(vc != 0))
    vcr = np.floor((vcr + 0.249999999999) * 4).astype(np.int64)
    vcr[vcr == 0] = 1
    vc[vc != 0] = vcr + 1
    vc[vc == 0] = 1

    ma1 = np.array([np.median(reformat5[io == k, iol]) for k in range(1, 6)])
    ma2 = np.array([np.median(reformat5[vc == k, vcl]) for k in range(1, 6)])

    med = np.column_stack([io, vc])
    unique_values, actionbloc = np.unique(med, axis=0, return_inverse=True)
    actionbloctrain = actionbloc[train_mask]

    unique_values_dose = np.column_stack(
        [
            ma2[unique_values[:, 1].astype(int) - 1],
            ma1[unique_values[:, 0].astype(int) - 1],
        ]
    )

    if include_eicu:
        return actionbloc, actionbloctrain, unique_values_dose, io, vc, ma1, ma2
    return actionbloc, actionbloctrain


def build_qldata3_transition(
    blocs: np.ndarray,
    idx: np.ndarray,
    actionbloctrain: np.ndarray,
    y90: np.ndarray,
    r2: np.ndarray,
    ncl: int,
) -> np.ndarray:
    """
    Build 4-column qldata3 for transition probability estimation.
    Columns: bloc, state, action, reward.
    """
    qldata = np.column_stack([blocs, idx, actionbloctrain, y90, r2])
    qldata3 = np.zeros((int(np.floor(qldata.shape[0] * 1.2)), 4))
    c = 0
    abss = np.array([ncl + 1, ncl])

    for i in range(qldata.shape[0] - 1):
        qldata3[c, :] = qldata[i, 0:4]
        c += 1
        if qldata[i + 1, 0] == 1:
            qldata3[c, :] = np.array([qldata[i, 0] + 1, abss[int(qldata[i, 3])], -1, qldata[i, 4]])
            c += 1

    return qldata3[:c, :]


def build_transition_matrices(
    qldata3: np.ndarray,
    n_states: int,
    n_actions: int,
    transthres: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build transition matrices and behavior policy from qldata3.

    qldata3 columns (0-based):
    - [0] bloc
    - [1] state
    - [2] action
    - [3] outcome_or_reward (unused here)

    Returns:
    - transition_sprime_s_a: T[s', s, a] (S, S, A)
    - transition_s_sprime_a: T[s, s', a] (S, S, A)
    - physpol: behavior policy P(a|s) (S, A)
    """
    transition_sprime_s_a = np.zeros((n_states, n_states, n_actions), dtype=float)
    transition_s_sprime_a = np.zeros((n_states, n_states, n_actions), dtype=float)
    sums0a0 = np.zeros((n_states, n_actions), dtype=float)

    for i in range(qldata3.shape[0] - 1):
        if qldata3[i + 1, 0] != 1:
            s0 = int(qldata3[i, 1])
            s1 = int(qldata3[i + 1, 1])
            a = int(qldata3[i, 2])
            transition_sprime_s_a[s1, s0, a] += 1
            transition_s_sprime_a[s0, s1, a] += 1
            sums0a0[s0, a] += 1

    sums0a0[sums0a0 <= transthres] = 0

    # Normalize transitions by (s,a) counts
    for s in range(n_states):
        for a in range(n_actions):
            denom = sums0a0[s, a]
            if denom == 0:
                transition_sprime_s_a[:, s, a] = 0
                transition_s_sprime_a[s, :, a] = 0
            else:
                transition_sprime_s_a[:, s, a] /= denom
                transition_s_sprime_a[s, :, a] /= denom

    transition_sprime_s_a[np.isnan(transition_sprime_s_a)] = 0
    transition_sprime_s_a[np.isinf(transition_sprime_s_a)] = 0
    transition_s_sprime_a[np.isnan(transition_s_sprime_a)] = 0
    transition_s_sprime_a[np.isinf(transition_s_sprime_a)] = 0

    # Ensure stochasticity: if a (s,a) row was pruned to all zeros, make it self-loop.
    row_sums = np.sum(transition_s_sprime_a, axis=1, keepdims=True)
    zero_rows = row_sums == 0
    if np.any(zero_rows):
        # zero_rows shape: (S, 1, A) after keepdims, broadcast to (S, S, A)
        s_indices = np.where(zero_rows[:, 0, :])
        transition_s_sprime_a[s_indices[0], :, s_indices[1]] = 0
        transition_s_sprime_a[s_indices[0], s_indices[0], s_indices[1]] = 1.0
        # Keep the transpose version consistent.
        transition_sprime_s_a[:, s_indices[0], s_indices[1]] = 0
        transition_sprime_s_a[s_indices[0], s_indices[0], s_indices[1]] = 1.0

    # Behavior policy from observed (s,a) counts
    row_sums = np.sum(sums0a0, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        physpol = sums0a0 / row_sums
    physpol[np.isnan(physpol)] = 0
    physpol[np.isinf(physpol)] = 0

    return transition_sprime_s_a, transition_s_sprime_a, physpol


def reward_from_transition(
    transition_sprime_s_a: np.ndarray,
    death_state: int,
    survive_state: int,
) -> np.ndarray:
    """
    Build reward matrix R(s,a) from transition tensor T[s', s, a].
    Rewards are -100 for death_state, +100 for survive_state, 0 otherwise.
    """
    n_states, _, n_actions = transition_sprime_s_a.shape
    r3 = np.zeros((n_states, n_states, n_actions), dtype=float)
    r3[death_state, :, :] = -100
    r3[survive_state, :, :] = 100
    r = np.sum(transition_sprime_s_a * r3, axis=0)
    return r


def policy_iteration_with_q(
    transition_s_sprime_a: np.ndarray,
    reward_s_a: np.ndarray,
    discount: float,
    max_iter: int = 2000,
    eval_type: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run policy iteration using pymdptoolbox and compute Q-values.

    Inputs:
    - transition_s_sprime_a: T[s, s', a] (S, S, A)
    - reward_s_a: R[s, a] (S, A)

    Returns:
    - Q: Q-values (S, A)
    - policy: optimal action per state (S,) 0-based
    """
    p_a_s_sprime = np.transpose(transition_s_sprime_a, (2, 0, 1))
    pi = mdp.PolicyIteration(p_a_s_sprime, reward_s_a, discount, max_iter=max_iter, eval_type=eval_type)
    pi.run()
    v = np.asarray(pi.V)

    q = np.zeros((reward_s_a.shape[0], reward_s_a.shape[1]), dtype=float)
    for a in range(p_a_s_sprime.shape[0]):
        q[:, a] = reward_s_a[:, a] + discount * p_a_s_sprime[a].dot(v)

    policy = np.argmax(q, axis=1)
    return q, policy
