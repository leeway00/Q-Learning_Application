#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plain script variant of AIClinician core for debugging/index checks.
- No logging
- No save helpers
- No run_core/main wrapper
"""
#%%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ai_utils import (
    cluster_states,
    drop_index_like_column,
    knn_search,
    require_columns,
    split_train_test,
    zscore_matlab,
)
from config import COLBIN, COLLOG, COLNORM, data_dir
from core_utils import (
    build_actions,
    build_qldata3_transition,
    build_transition_matrices,
    policy_iteration_with_q,
    reward_from_transition,
)
from evaluation import apply_policy_probabilities, build_qldata3, evaluate_policy

# MDP / model configuration
NCL = 750
NRA = 5
NACT = NRA * NRA
TRANSTHRES = 5
NCLUSTERING = 32
PROP = 0.25
GAMMA = 0.99
NCV = 5
N_STATES = NCL + 2
DEATH_STATE = NCL
SURVIVE_STATE = NCL + 1

# Hardcoded params for kernel/debug runs
MIMIC_CSV = data_dir / "mimictable.csv"
NR_REPS = 500
SEED = None

rng = np.random.default_rng(SEED)

mimic_df = drop_index_like_column(pd.read_csv(MIMIC_CSV))
require_columns(mimic_df, COLBIN + COLNORM + COLLOG + ["bloc", "icustayid", "mortality_90d"])

reformat5 = mimic_df.values.copy()
icustayidlist = mimic_df["icustayid"].values
icuuniqueids = np.unique(icustayidlist)

OA = np.full((N_STATES, NR_REPS), np.nan)
recqvi = np.full((NR_REPS * 2, 30), np.nan)
idxs = np.full((icustayidlist.shape[0], NR_REPS), np.nan)

colbin_idx = [mimic_df.columns.get_loc(c) for c in COLBIN]
colnorm_idx = [mimic_df.columns.get_loc(c) for c in COLNORM]
collog_idx = [mimic_df.columns.get_loc(c) for c in COLLOG]
#%%
mimicraw = mimic_df.iloc[:, colbin_idx + colnorm_idx + collog_idx]
mimiczs = np.concatenate(
    [
        reformat5[:, colbin_idx] - 0.5,
        zscore_matlab(reformat5[:, colnorm_idx])[0],
        zscore_matlab(np.log(0.1 + reformat5[:, collog_idx]))[0],
    ],
    axis=1,
)
mimiczs[:, 2] = np.log(mimiczs[:, 2] + 0.6) # second column is MAX DOSES
mimiczs[:, 44] = 2 * mimiczs[:, 44]
#%%
for modl in tqdm(range(NR_REPS), desc="Models", unit="model"):
    train, test = split_train_test(icustayidlist, icuuniqueids, NCV, rng)

    x = mimiczs[train, :]
    xtestmimic = mimiczs[~train, :]
    blocs = reformat5[train, 0]
    bloctestmimic = reformat5[~train, 0]
    ptid = reformat5[train, 1]
    ptidtestmimic = reformat5[~train, 1]
    outcome = 9
    y90 = reformat5[train, outcome]

    centroids, idx = cluster_states(x, PROP, NCL, NCLUSTERING, 10000, rng)

    iol = mimic_df.columns.get_loc("input_4hourly")
    vcl = mimic_df.columns.get_loc("max_dose_vaso")
    actionbloc, actionbloctrain = build_actions(reformat5, iol, vcl, train, include_eicu=False)

    r = 100.0
    r2 = r * (2 * (1 - y90) - 1)
    qldata3 = build_qldata3_transition(blocs, idx, actionbloctrain, y90, r2, NCL)

    transitionr, transitionr2, physpol = build_transition_matrices(
        qldata3,
        N_STATES,
        NACT,
        TRANSTHRES,
    )

    R = reward_from_transition(transitionr, DEATH_STATE, SURVIVE_STATE)
    Q, optimal_action = policy_iteration_with_q(transitionr2, R, GAMMA)
    OA[:, modl] = optimal_action

    r2 = r * (2 * (1 - y90) - 1)
    abss = np.array([SURVIVE_STATE, DEATH_STATE])
    qldata3 = build_qldata3(blocs, idx, actionbloctrain, y90, r2, ptid, abss)
    qldata3 = apply_policy_probabilities(qldata3, physpol, optimal_action, NACT, 0.01, NCL)
    qldata3train = qldata3.copy()
    bootql, bootwis = evaluate_policy(qldata3, physpol, GAMMA, 6, 750)

    recqvi[modl, 0] = modl + 1
    recqvi[modl, 3] = np.nanmean(bootql)
    recqvi[modl, 4] = np.quantile(bootql, 0.99)
    recqvi[modl, 5] = np.nanmean(bootwis)
    recqvi[modl, 6] = np.quantile(bootwis, 0.05)

    idxtest = knn_search(centroids, xtestmimic)
    idxs[test, modl] = idxtest
    actionbloctest = actionbloc[~train]
    y90test = reformat5[~train, outcome]
    r2 = r * (2 * (1 - y90test) - 1)

    qldata3 = build_qldata3(
        bloctestmimic,
        idxtest,
        actionbloctest,
        y90test,
        r2,
        ptidtestmimic,
        abss,
    )
    qldata3 = apply_policy_probabilities(qldata3, physpol, optimal_action, NACT, 0.01, NCL)
    qldata3test = qldata3.copy()
    bootmimictestql, bootmimictestwis = evaluate_policy(qldata3, physpol, GAMMA, 6, 2000)

    recqvi[modl, 18] = np.quantile(bootmimictestql, 0.95)
    recqvi[modl, 19] = np.nanmean(bootmimictestql)
    recqvi[modl, 20] = np.quantile(bootmimictestql, 0.99)
    recqvi[modl, 21] = np.nanmean(bootmimictestwis)
    recqvi[modl, 22] = np.quantile(bootmimictestwis, 0.01)
    recqvi[modl, 23] = np.quantile(bootmimictestwis, 0.05)

recqvi = recqvi[:NR_REPS, :]

print("Done")
print("recqvi:", recqvi.shape)
print("OA:", OA.shape)
print("idxs:", idxs.shape)
