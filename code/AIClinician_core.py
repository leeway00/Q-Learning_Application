#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python replication of refrepo/AI_Clinician/AIClinician_core_160219.m
Reference: MATLAB files under refrepo/AI_Clinician (author: Matthieu Komorowski).

Notes:
- This is a direct translation of the MATLAB core logic. It favors fidelity over speed.
- Indices are 0-based for states/actions.
- eICU evaluation is intentionally removed (private dataset).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
    _CUM_L_AVAILABLE,
)
from config import COLBIN, COLLOG, COLNORM, data_dir, mdp_log_dir, mdp_output_dir
from core_utils import (
    build_actions,
    build_qldata3_transition,
    build_transition_matrices,
    policy_iteration_with_q,
    reward_from_transition,
)
from evaluation import apply_policy_probabilities, build_qldata3, evaluate_policy


# ----------------------------- Core logic -----------------------------

# Logging
LOGGER = logging.getLogger("ai_clinician_core")


class _NoLoopLogsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "loop_log", False)

# MDP / model configuration (kept module-level for easy inspection)
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


def run_core(
    mimic_csv: Path,
    nr_reps: int = 500,
    seed: Optional[int] = None,
    do_plots: bool = False,
) -> Dict[str, object]:
    """Main entry to replicate AIClinician_core_160219.m"""
    rng = np.random.default_rng(seed)
    LOGGER.info("Starting run_core: nr_reps=%d seed=%s", nr_reps, str(seed))
    LOGGER.info("cuML GPU available: %s", _CUM_L_AVAILABLE)

    # Load data
    mimic_df = drop_index_like_column(pd.read_csv(mimic_csv))
    LOGGER.info("Loaded MIMIC CSV: rows=%d cols=%d", mimic_df.shape[0], mimic_df.shape[1])

    require_columns(mimic_df, COLBIN + COLNORM + COLLOG + ["bloc", "icustayid", "mortality_90d"])
    LOGGER.info("Verified required columns")

    reformat5 = mimic_df.values.copy()
    icustayidlist = mimic_df["icustayid"].values
    icuuniqueids = np.unique(icustayidlist)

    OA = np.full((N_STATES, nr_reps), np.nan)
    recqvi = np.full((nr_reps * 2, 30), np.nan)
    allpols: Dict[int, Dict[str, object]] = {}
    polkeep = 1

    # Build MIMICraw and MIMICzs
    colbin_idx = [mimic_df.columns.get_loc(c) for c in COLBIN]
    colnorm_idx = [mimic_df.columns.get_loc(c) for c in COLNORM]
    collog_idx = [mimic_df.columns.get_loc(c) for c in COLLOG]

    # mimicraw is kept for MATLAB parity (used for eICU conversion factors in the original code).
    # mimicraw = mimic_df.iloc[:, colbin_idx + colnorm_idx + collog_idx].values.copy()
    mimiczs = np.concatenate(
        [
            reformat5[:, colbin_idx] - 0.5,
            zscore_matlab(reformat5[:, colnorm_idx])[0],
            zscore_matlab(np.log(0.1 + reformat5[:, collog_idx]))[0],
        ],
        axis=1,
    )
    mimiczs[:, 3] = np.log(mimiczs[:, 3] + 0.6)
    mimiczs[:, 44] = 2 * mimiczs[:, 44]
    LOGGER.info("Built MIMICraw/MIMICzs")

    # Conversion factors (kept for MATLAB parity; used for eICU z-scoring in original code)
    # _, cmu, csigma = zscore_matlab(mimicraw[:, 4:36])
    # _, dmu, dsigma = zscore_matlab(np.log(0.1 + mimicraw[:, 36:47]))

    idxs = np.full((icustayidlist.shape[0], nr_reps), np.nan)

    for modl in tqdm(range(nr_reps), desc="Models", unit="model"):
        # LOGGER.info("Model %d/%d", modl + 1, nr_reps)
        if (modl + 1) % 10 == 0 or modl == 0:
            LOGGER.info("Progress: model %d/%d", modl + 1, nr_reps, extra={"loop_log": True})
        train, test = split_train_test(icustayidlist, icuuniqueids, NCV, rng)
        # LOGGER.info("Split train/test: train_rows=%d test_rows=%d", np.sum(train), np.sum(test))

        x = mimiczs[train, :]
        xtestmimic = mimiczs[~train, :]
        blocs = reformat5[train, 0]
        bloctestmimic = reformat5[~train, 0]
        ptid = reformat5[train, 1]
        ptidtestmimic = reformat5[~train, 1]
        outcome = 9  # 90d mortality (column index in MATLAB is 10 -> here 0-based)
        y90 = reformat5[train, outcome]

        # K-means on sampled rows
        centroids, idx = cluster_states(x, PROP, NCL, NCLUSTERING, 10000, rng)
        # LOGGER.info("Clustering done: centroids=%s", centroids.shape)

        # Create actions
        iol = mimic_df.columns.get_loc("input_4hourly")
        vcl = mimic_df.columns.get_loc("max_dose_vaso")
        actionbloc, actionbloctrain = build_actions(reformat5, iol, vcl, train, include_eicu=False)
        # LOGGER.info("Actions built: unique_actions=%d", np.unique(actionbloc).size)

        # Create QLDATA3 for transition estimation
        r = 100.0
        r2 = r * (2 * (1 - y90) - 1)
        qldata3 = build_qldata3_transition(blocs, idx, actionbloctrain, y90, r2, NCL)
        # LOGGER.info("Transition qldata3 built: rows=%d", qldata3.shape[0])

        # Transition matrices and behavior policy
        transitionr, transitionr2, physpol = build_transition_matrices(
            qldata3,
            N_STATES,
            NACT,
            TRANSTHRES,
        )
        # LOGGER.info("Transition matrices built")

        # Reward matrix
        R = reward_from_transition(transitionr, DEATH_STATE, SURVIVE_STATE)

        # Policy iteration with Q (via pymdptoolbox policy iteration + Q reconstruction).
        Q, optimal_action = policy_iteration_with_q(transitionr2, R, GAMMA)
        OA[:, modl] = optimal_action
        # LOGGER.info("Policy iteration complete")

        # Off-policy evaluation: MIMIC train
        r = 100.0
        r2 = r * (2 * (1 - y90) - 1)
        abss = np.array([SURVIVE_STATE, DEATH_STATE])
        qldata3 = build_qldata3(blocs, idx, actionbloctrain, y90, r2, ptid, abss)
        qldata3 = apply_policy_probabilities(qldata3, physpol, optimal_action, NACT, 0.01, NCL)
        qldata3train = qldata3.copy()
        # LOGGER.info("Start evaluating policy on train data")
        bootql, bootwis = evaluate_policy(qldata3, physpol, GAMMA, 6, 750)
        # LOGGER.info("Off-policy eval (train) complete")

        recqvi[modl, 0] = modl + 1
        recqvi[modl, 3] = np.nanmean(bootql)
        recqvi[modl, 4] = np.quantile(bootql, 0.99)
        recqvi[modl, 5] = np.nanmean(bootwis)
        recqvi[modl, 6] = np.quantile(bootwis, 0.05)

        # Off-policy evaluation: MIMIC test
        idxtest = knn_search(centroids, xtestmimic)
        idxs[test, modl] = idxtest
        actionbloctest = actionbloc[~train]
        y90test = reformat5[~train, outcome]
        r2 = r * (2 * (1 - y90test) - 1)
        qldata3 = build_qldata3(bloctestmimic, idxtest, actionbloctest, y90test, r2, ptidtestmimic, abss)
        qldata3 = apply_policy_probabilities(qldata3, physpol, optimal_action, NACT, 0.01, NCL)
        qldata3test = qldata3.copy()
        # LOGGER.info("Start evaluating policy on test data")
        bootmimictestql, bootmimictestwis = evaluate_policy(qldata3, physpol, GAMMA, 6, 2000)
        # LOGGER.info("Off-policy eval (test) complete")
        recqvi[modl, 18] = np.quantile(bootmimictestql, 0.95)
        recqvi[modl, 19] = np.nanmean(bootmimictestql)
        recqvi[modl, 20] = np.quantile(bootmimictestql, 0.99)
        recqvi[modl, 21] = np.nanmean(bootmimictestwis)
        recqvi[modl, 22] = np.quantile(bootmimictestwis, 0.01)
        recqvi[modl, 23] = np.quantile(bootmimictestwis, 0.05)

        # Store good models
        if recqvi[modl, 23] > 0:
            allpols[polkeep] = {
                "modl": modl + 1,
                "Qon": Q,
                "physpol": physpol,
                "transitionr": transitionr,
                "transitionr2": transitionr2,
                "R": R,
                "C": centroids,
                "train": train,
                "qldata3train": qldata3train,
                "qldata3test": qldata3test,
            }
            polkeep += 1

    recqvi = recqvi[:nr_reps, :]
    LOGGER.info("run_core complete")

    return {
        "recqvi": recqvi,
        "OA": OA,
        "idxs": idxs,
        "allpols": allpols,
    }


def save_outputs(
    outputs: Dict[str, object],
    out_dir: Path,
    run_tag: str,
    meta: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"mdp_core_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        run_dir / "results.npz",
        recqvi=outputs["recqvi"],
        OA=outputs["OA"],
        idxs=outputs["idxs"],
    )

    allpols = outputs.get("allpols", {})
    if allpols:
        pol_dir = run_dir / "allpols"
        pol_dir.mkdir(parents=True, exist_ok=True)
        for key, payload in allpols.items():
            np.savez_compressed(
                pol_dir / f"model_{key:04d}.npz",
                modl=payload["modl"],
                Qon=payload["Qon"],
                physpol=payload["physpol"],
                transitionr=payload["transitionr"],
                transitionr2=payload["transitionr2"],
                R=payload["R"],
                C=payload["C"],
                train=payload["train"],
                qldata3train=payload["qldata3train"],
                qldata3test=payload["qldata3test"],
            )

    meta_path = run_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python replication of AIClinician_core_160219.m")
    parser.add_argument("--mimic-csv", type=Path, default=data_dir / "mimictable.csv")
    parser.add_argument("--out-dir", type=Path, default=mdp_output_dir)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--nr-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = mdp_log_dir / f"mdp_core_{run_tag}.log"
    mdp_log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.addFilter(_NoLoopLogsFilter())

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[file_handler, stream_handler],
    )

    outputs = run_core(
        mimic_csv=args.mimic_csv,
        nr_reps=args.nr_reps,
        seed=args.seed,
    )

    meta = {
        "run_tag": run_tag,
        "mimic_csv": str(args.mimic_csv),
        "nr_reps": args.nr_reps,
        "seed": args.seed,
        "ncl": NCL,
        "nra": NRA,
        "nact": NACT,
        "gamma": GAMMA,
        "transthres": TRANSTHRES,
        "nclustering": NCLUSTERING,
        "prop": PROP,
        "ncv": NCV,
        "death_state": DEATH_STATE,
        "survive_state": SURVIVE_STATE,
        "feature_groups": {"colbin": COLBIN, "colnorm": COLNORM, "collog": COLLOG},
        "notes": "eICU-related code omitted; outputs are MIMIC-only.",
    }
    save_outputs(outputs, args.out_dir, run_tag, meta)


if __name__ == "__main__":
    main()
