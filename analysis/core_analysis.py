#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-run analysis helpers for AIClinician_core outputs.

Implements:
1) Best model selection from recqvi.
2) Recovery/loading of the selected model payload from allpols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import MDP_OUTPUT_DIR


def _run_dir_from_tag(run_tag: str) -> Path:
    return MDP_OUTPUT_DIR / f"mdp_core_{run_tag}"


def _results_path(run_tag: str) -> Path:
    run_dir = _run_dir_from_tag(run_tag)
    new_path = run_dir / "results.npz"
    old_path = MDP_OUTPUT_DIR / f"mdp_core_{run_tag}.npz"
    if new_path.exists():
        return new_path
    return old_path


def select_best_model(run_tag: str) -> Dict[str, float]:
    """
    Select best model id from recqvi.

    MATLAB reference:
    - best model maximizes recqvi(:,24) (95% LB of AI value on MIMIC test).
    - model id is in recqvi(:,1).

    Returns:
    - bestpol: selected model id (1-based, same as MATLAB model numbering)
    - best_value: max value of recqvi(:,24) equivalent
    - n_candidates: number of candidate rows considered
    """
    results_path = _results_path(run_tag)
    if not results_path.exists():
        raise FileNotFoundError(f"results file not found for run_tag={run_tag}: {results_path}")

    recqvi = np.load(results_path, allow_pickle=True)["recqvi"]
    if recqvi.ndim != 2 or recqvi.shape[1] < 24:
        raise ValueError(f"Unexpected recqvi shape: {recqvi.shape}")

    model_id = recqvi[:, 0]
    mimic_test_lb = recqvi[:, 23]  # MATLAB col 24 (1-based) -> Python 23 (0-based)

    valid = np.isfinite(model_id) & np.isfinite(mimic_test_lb)
    if not np.any(valid):
        raise ValueError("No valid rows found for best-model selection in recqvi.")

    candidate_ids = model_id[valid].astype(int)
    candidate_values = mimic_test_lb[valid]

    best_idx = int(np.nanargmax(candidate_values))
    bestpol = int(candidate_ids[best_idx])
    best_value = float(candidate_values[best_idx])

    return {
        "bestpol": bestpol,
        "best_value": best_value,
        "n_candidates": int(candidate_values.shape[0]),
    }


def recover_best_model(run_tag: str, bestpol: Optional[int] = None) -> Dict[str, object]:
    """
    Recover best model payload from allpols directory.

    If bestpol is None, it is selected using select_best_model(run_tag).
    """
    if bestpol is None:
        bestpol = int(select_best_model(run_tag)["bestpol"])

    allpols_dir = _run_dir_from_tag(run_tag) / "allpols"
    if not allpols_dir.exists():
        raise FileNotFoundError(f"allpols directory not found: {allpols_dir}")

    matched_path: Optional[Path] = None
    for model_path in sorted(allpols_dir.glob("model_*.npz")):
        with np.load(model_path, allow_pickle=True) as payload:
            modl = int(payload["modl"])
            if modl == bestpol:
                matched_path = model_path
                break

    if matched_path is None:
        raise FileNotFoundError(
            f"No allpols payload found for bestpol={bestpol} in {allpols_dir}"
        )

    with np.load(matched_path, allow_pickle=True) as payload:
        return {
            "path": str(matched_path),
            "modl": int(payload["modl"]),
            "Qon": payload["Qon"],
            "physpol": payload["physpol"],
            "transitionr": payload["transitionr"],
            "transitionr2": payload["transitionr2"],
            "R": payload["R"],
            "C": payload["C"],
            "train": payload["train"],
            "qldata3train": payload["qldata3train"],
            "qldata3test": payload["qldata3test"],
        }
