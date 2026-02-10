#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba-accelerated off-policy evaluation helpers.
Keeps code/offpolicy.py unchanged and provides JIT alternatives.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


if njit:

    @njit
    def _offpolicy_qlearning_150816_jit(
        qldata3: np.ndarray, gamma: float, alpha: float, numtraces: int
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            # MATLAB-equivalent sampling: 1..(nrepi-2) -> Python 0..(nrepi-3)
            i = listi[np.random.randint(0, nrepi - 2)]

            trace_len = 0
            # First pass: count length
            k = i
            while qldata3[k + 1, 0] != 1:
                trace_len += 1
                k += 1

            if trace_len == 0:
                continue

            # Second pass: gather trace
            trace_r = np.empty(trace_len)
            trace_s = np.empty(trace_len, dtype=np.int64)
            trace_a = np.empty(trace_len, dtype=np.int64)
            k = i
            t = 0
            while qldata3[k + 1, 0] != 1:
                trace_r[t] = qldata3[k + 1, 3]
                trace_s[t] = int(qldata3[k + 1, 1])
                trace_a[t] = int(qldata3[k + 1, 2])
                t += 1
                k += 1

            return_t = trace_r[trace_len - 1]
            for t in range(trace_len - 2, -1, -1):
                s = trace_s[t]
                a = trace_a[t]
                if a >= 0:
                    q[s, a] = (1 - alpha) * q[s, a] + alpha * return_t
                return_t = return_t * gamma + trace_r[t]

            sumq[jj] = np.sum(q)
            jj += 1

            if (j + 1) % (500 * modu) == 0:
                s = float(np.mean(sumq[j - 49999 : j]))
                d = (s - maxavgq) / maxavgq
                if abs(d) < 0.001:
                    break
                maxavgq = s

        return q, sumq[:jj]


def offpolicy_qlearning_150816_jit(
    qldata3: np.ndarray, gamma: float, alpha: float, numtraces: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-accelerated version of offpolicy_qlearning_150816.
    Falls back to the pure-NumPy implementation if numba is unavailable.
    """
    if njit is None:
        from offpolicy import offpolicy_qlearning_150816

        return offpolicy_qlearning_150816(qldata3, gamma, alpha, numtraces)
    return _offpolicy_qlearning_150816_jit(qldata3, gamma, alpha, numtraces)


def offpolicy_eval_tdlearning_jit(
    qldata3: np.ndarray, physpol: np.ndarray, gamma: float, num_iter: int
) -> np.ndarray:
    """
    TD-learning evaluation using JIT Q-learning kernel.
    Mirrors offpolicy_eval_tdlearning but swaps in the JIT core.
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

        qoff, _ = offpolicy_qlearning_150816_jit(q, gamma, 0.1, 300000)

        v = physpol[:ncl, :] * qoff[:ncl, :]
        vs = np.nansum(v, axis=1)
        bootql.append(np.nansum(vs * d) / np.sum(d))

    return np.array(bootql)


def offpolicy_eval_wis_vectorized(
    qldata3: np.ndarray, gamma: float, num_iter: int
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Vectorized WIS evaluation. Keeps the same logic as offpolicy_eval_wis
    but replaces inner loops with bincount-based aggregation.
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
        if num_trials < 2:
            bootwis.append(np.nan)
            last_individual = np.array([])
            last_rho = np.array([])
            last_c = 0
            continue

        # Match MATLAB loop: use trajectories 0..(num_trials-2)
        start0 = fence_posts[0]
        end = fence_posts[-1]
        q_seg = q[start0:end, :]
        fence = fence_posts - start0

        lengths = np.diff(fence)
        seg_id = np.repeat(np.arange(num_trials - 1), lengths)

        valid = np.ones(q_seg.shape[0], dtype=bool)
        valid[fence[1:] - 1] = False  # exclude last row of each trajectory
        valid_idx = np.nonzero(valid)[0]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = q_seg[:, 5] / q_seg[:, 4]
            log_ratio = np.log(ratio[valid_idx])

        sum_logs = np.bincount(seg_id[valid_idx], weights=log_ratio, minlength=num_trials - 1)
        rho_array = np.exp(sum_logs)
        c = int(np.sum(rho_array > 0))

        pos = valid_idx - fence[seg_id[valid_idx]]
        weights = gamma ** pos
        rewards = q_seg[valid_idx + 1, 3]
        current = np.bincount(
            seg_id[valid_idx],
            weights=weights * rewards,
            minlength=num_trials - 1,
        )
        individual = current * rho_array

        ii_bad = np.isinf(rho_array) | np.isnan(rho_array)
        normalization = np.nansum(rho_array[~ii_bad])
        bootwis.append(np.nansum(individual[~ii_bad]) / normalization if normalization > 0 else np.nan)

        last_individual = individual
        last_rho = rho_array
        last_c = c

    individual_trial = last_individual[~(np.isinf(last_rho) | np.isnan(last_rho))] / last_rho[
        ~(np.isinf(last_rho) | np.isnan(last_rho))
    ]
    return np.array(bootwis), last_c, individual_trial


def offpolicy_multiple_eval_010518_jit(
    qldata3: np.ndarray, physpol: np.ndarray, gamma: float, do_ql: int, iter_ql: int, iter_wis: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT/vectorized equivalent of offpolicy_multiple_eval_010518.
    Preserves the same signature and do_ql behavior.
    """
    if do_ql == 1:
        bootql = offpolicy_eval_tdlearning_jit(qldata3, physpol, gamma, iter_ql)
    else:
        bootql = np.array([55.0])
    bootwis, _, _ = offpolicy_eval_wis_vectorized(qldata3, gamma, iter_wis)
    return bootql, bootwis
