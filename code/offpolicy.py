#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-policy evaluation helpers translated from MATLAB:
refrepo/AI_Clinician/offpolicy_eval_tdlearning.m
refrepo/AI_Clinician/offpolicy_eval_wis.m
refrepo/AI_Clinician/offpolicy_multiple_eval_010518.m
refrepo/AI_Clinician/OffpolicyQlearning150816.m
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def offpolicy_qlearning_150816(
    qldata3: np.ndarray, gamma: float, alpha: float, numtraces: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Off-policy Q-learning (Monte Carlo style) over sampled trajectories.
    Returns:
    - Q: state-action values
    - sumQ: running sum of Q entries (used to detect convergence)
    """
    sumq = np.zeros(numtraces)
    actions = np.unique(qldata3[:, 2])
    actions = actions[actions >= 0]
    nact = int(actions.max()) + 1
    ncl = int(np.unique(qldata3[:, 1]).max()) + 1
    q = np.zeros((ncl, nact))
    maxavgq = 1.0
    modu = 100

    listi = np.where(qldata3[:, 0] == 1)[0]
    nrepi = listi.size
    jj = 0

    for j in range(numtraces):
        i = listi[np.random.randint(0, nrepi - 2)]
        trace = []
        while qldata3[i + 1, 0] != 1:
            s1 = int(qldata3[i + 1, 1])
            a1 = int(qldata3[i + 1, 2])
            r1 = float(qldata3[i + 1, 3])
            trace.append((r1, s1, a1))
            i += 1

        if not trace:
            continue

        return_t = trace[-1][0]
        for t in range(len(trace) - 2, -1, -1):
            s = int(trace[t][1])
            a = int(trace[t][2])
            if a >= 0:
                q[s, a] = (1 - alpha) * q[s, a] + alpha * return_t
            return_t = return_t * gamma + trace[t][0]

        sumq[jj] = np.sum(q)
        jj += 1

        if (j + 1) % (500 * modu) == 0:
            s = float(np.mean(sumq[j - 49999 : j]))
            d = (s - maxavgq) / maxavgq
            if abs(d) < 0.001:
                break
            maxavgq = s

    sumq = sumq[:jj]
    return q, sumq


def offpolicy_eval_tdlearning(
    qldata3: np.ndarray, physpol: np.ndarray, gamma: float, num_iter: int
) -> np.ndarray:
    """
    TD-learning estimate of clinician policy value.
    Bootstraps trajectories and evaluates mean value over the state distribution.
    """
    ncl = physpol.shape[0] - 2
    bootql = []
    p = np.unique(qldata3[:, 7])
    prop = min(5000 / p.size, 0.75)

    ii = qldata3[:, 0] == 1
    a = qldata3[ii, 1]
    d = np.zeros(ncl)
    for i in range(ncl):
        d[i] = np.sum(a == i)

    for _ in range(num_iter):
        ii = np.random.binomial(n=1, p=prop, size=p.shape[0])
        j = np.isin(qldata3[:, 7], p[ii == 1])
        q = qldata3[j, 0:4]

        qoff, _ = offpolicy_qlearning_150816(q, gamma, 0.1, 300000)

        v = physpol[:ncl, :] * qoff[:ncl, :]
        vs = np.nansum(v, axis=1)
        bootql.append(np.nansum(vs * d) / np.sum(d))

    return np.array(bootql)


def offpolicy_eval_wis(
    qldata3: np.ndarray, gamma: float, num_iter: int
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Weighted Importance Sampling (WIS) estimate for the target policy value.
    Returns:
    - bootwis: bootstrap distribution of WIS estimates
    - c: count of non-zero importance weights in the last iteration
    - individual_trial_estimators: per-trajectory WIS estimators for last iteration
    """
    bootwis = []
    p = np.unique(qldata3[:, 7])
    prop = min(25000 / p.size, 0.75)
    last_individual = np.array([])
    last_rho = np.array([])
    last_c = 0

    for _ in range(num_iter):
        ii = np.random.binomial(n=1, p=prop, size=p.shape[0])
        j = np.isin(qldata3[:, 7], p[ii == 1])
        q = qldata3[j, :]
        fence_posts = np.where(q[:, 0] == 1)[0]
        num_trials = fence_posts.size
        individual = np.full(num_trials, np.nan)
        rho_array = np.full(num_trials, np.nan)
        c = 0

        for i in range(num_trials - 1):
            rho = 1.0
            for t in range(fence_posts[i], fence_posts[i + 1] - 1):
                rho *= q[t, 5] / q[t, 4]
            if rho > 0:
                c += 1
            rho_array[i] = rho

        ii_bad = np.isinf(rho_array) | np.isnan(rho_array)
        normalization = np.nansum(rho_array[~ii_bad])

        for i in range(num_trials - 1):
            current = 0.0
            rho = 1.0
            discount = 1.0 / gamma
            for t in range(fence_posts[i], fence_posts[i + 1] - 1):
                rho *= q[t, 5] / q[t, 4]
                discount *= gamma
                current += discount * q[t + 1, 3]
            individual[i] = current * rho

        bootwis.append(np.nansum(individual[~ii_bad]) / normalization)
        last_individual = individual
        last_rho = rho_array
        last_c = c

    individual_trial = last_individual[~(np.isinf(last_rho) | np.isnan(last_rho))] / last_rho[
        ~(np.isinf(last_rho) | np.isnan(last_rho))
    ]
    return np.array(bootwis), last_c, individual_trial


def offpolicy_multiple_eval_010518(
    qldata3: np.ndarray, physpol: np.ndarray, gamma: float, do_ql: int, iter_ql: int, iter_wis: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper that runs TD-learning (clinician) and WIS (AI policy) in one call.
    Mirrors offpolicy_multiple_eval_010518.m.
    """
    if do_ql == 1:
        bootql = offpolicy_eval_tdlearning(qldata3, physpol, gamma, iter_ql)
    else:
        bootql = np.array([55.0])
    bootwis, _, _ = offpolicy_eval_wis(qldata3, gamma, iter_wis)
    return bootql, bootwis
