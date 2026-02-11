#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIMIC-only implementations of MATLAB core analysis items:
3) Safety curve (Fig 2A, without eICU curve)
4) Policy value boxplot (Fig 2B, available columns only)
5) Calibration curve (Fig 2C)
6) Per-patient average return histogram (Fig 2D)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
CODE_DIR = REPO_ROOT / "code"
for p in (REPO_ROOT, CODE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import MDP_OUTPUT_DIR
from core_analysis import recover_best_model
from offpolicy_jit import offpolicy_qlearning_150816_jit


def _run_dir_from_tag(run_tag: str) -> Path:
    return MDP_OUTPUT_DIR / f"mdp_core_{run_tag}"


def _load_recqvi(run_tag: str) -> np.ndarray:
    run_dir = _run_dir_from_tag(run_tag)
    new_path = run_dir / "results.npz"
    old_path = MDP_OUTPUT_DIR / f"mdp_core_{run_tag}.npz"
    path = new_path if new_path.exists() else old_path
    if not path.exists():
        raise FileNotFoundError(f"results file not found for run_tag={run_tag}: {path}")
    return np.load(path, allow_pickle=True)["recqvi"]


def plot_fig2a_safety_curve(run_tag: str, save_path: Path | None = None) -> plt.Figure:
    """
    MATLAB Fig 2A (MIMIC-only subset):
    - running max of physicians' 95% UB (recqvi col 19 in MATLAB)
    - running max of AI 95% LB in MIMIC test (recqvi col 24 in MATLAB)
    """
    recqvi = _load_recqvi(run_tag)
    phys_ub = recqvi[:, 18]  # MATLAB col 19
    ai_lb_mimic = recqvi[:, 23]  # MATLAB col 24

    def running_max(x: np.ndarray) -> np.ndarray:
        y = np.full_like(x, np.nan, dtype=float)
        current = -np.inf
        for i, v in enumerate(x):
            if np.isfinite(v) and v > current:
                current = v
            y[i] = current if current > -np.inf else np.nan
        return y

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogx(running_max(ai_lb_mimic), linewidth=2, label="AI 95% LB (MIMIC test)")
    ax.semilogx(running_max(phys_ub), linewidth=2, label="Clinician 95% UB (MIMIC test)")
    ax.set_xlabel("Number of models built")
    ax.set_ylabel("Estimated policy value")
    ax.set_title("Fig 2A (MIMIC-only)")
    ax.set_xlim(left=1)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    ax.set_box_aspect(1)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_fig2b_boxplot(run_tag: str, save_path: Path | None = None) -> plt.Figure:
    """
    MATLAB Fig 2B (policy value boxplot).
    Uses available Python columns:
    - Clinicians: recqvi col 20 in MATLAB -> recqvi[:,19]
    - AI: recqvi col 22 in MATLAB -> recqvi[:,21]
    Optional (if present): zero-drug/random from cols 25/26.
    """
    recqvi = _load_recqvi(run_tag)
    series: List[np.ndarray] = []
    labels: List[str] = []

    clinicians = recqvi[:, 19]
    ai = recqvi[:, 21]
    if np.isfinite(clinicians).any():
        series.append(clinicians[np.isfinite(clinicians)])
        labels.append("Clinicians")
    if np.isfinite(ai).any():
        series.append(ai[np.isfinite(ai)])
        labels.append("AI")

    # MATLAB also uses cols 25/26; include only if populated.
    for idx, name in [(24, "Zero drug"), (25, "Random")]:
        if recqvi.shape[1] > idx:
            col = recqvi[:, idx]
            if np.isfinite(col).any():
                series.append(col[np.isfinite(col)])
                labels.append(name)

    if not series:
        raise ValueError("No valid policy-value columns available for boxplot.")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.boxplot(series, labels=labels)
    if np.isfinite(ai).any():
        ymax = float(np.nanmax(ai))
        if len(labels) >= 2:
            ax.plot([1.5, 2.5], [ymax, ymax], linewidth=2, color="g", label="Chosen policy")
            ax.legend(loc="lower left")
    ax.set_ylabel("Estimated policy value")
    ax.set_title("Fig 2B (MIMIC-only)")
    ax.grid(alpha=0.2, axis="y")
    ax.set_box_aspect(1)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


@dataclass
class CalibrationData:
    bootql: np.ndarray
    prog: np.ndarray  # columns: Qoff, morta, id, rep


def build_calibration_data(
    run_tag: str,
    bestpol: int,
    gamma: float = 0.99,
    num_iter: int = 100,
) -> CalibrationData:
    """
    Approximation of MATLAB offpolicy_eval_tdlearning_with_morta using saved qldata3train.
    """
    payload = recover_best_model(run_tag, bestpol=bestpol)
    qldata3train = payload["qldata3train"]
    physpol = payload["physpol"]
    ncl = 750

    p = np.unique(qldata3train[:, 7])
    prop = min(5000 / p.size, 0.75)

    start_mask = qldata3train[:, 0] == 1
    start_states = qldata3train[start_mask, 1]
    d = np.zeros(ncl, dtype=float)
    for i in range(ncl):
        d[i] = np.sum(start_states == i)

    # mortality by patient id from terminal rows (action=-1, reward +/-100)
    terminal = qldata3train[:, 2] < 0
    terminal_rows = qldata3train[terminal]
    mortality_by_id: Dict[float, int] = {}
    for row in terminal_rows:
        pid = float(row[7])
        mortality_by_id[pid] = 1 if row[3] < 0 else 0

    bootql: List[float] = []
    prog_rows: List[Tuple[float, int, float, int]] = []

    for rep in range(1, num_iter + 1):
        ii = np.random.binomial(n=1, p=prop, size=p.shape[0])
        selected_ids = p[ii == 1]
        row_mask = np.isin(qldata3train[:, 7], selected_ids)
        q = qldata3train[row_mask, 0:4]
        qoff, _ = offpolicy_qlearning_150816_jit(q, gamma, 0.1, 300000)

        v = physpol[:ncl, :] * qoff[:ncl, :]
        vs = np.nansum(v, axis=1)
        bootql.append(float(np.nansum(vs * d) / np.sum(d)))

        # For each selected non-terminal row, record (Qoff(actual_action), mortality, ptid, rep)
        selected_rows = qldata3train[row_mask]
        non_terminal = selected_rows[:, 2] >= 0
        selected_rows = selected_rows[non_terminal]
        for row in selected_rows:
            s = int(row[1])
            a = int(row[2])
            pid = float(row[7])
            morta = mortality_by_id.get(pid)
            if morta is None:
                continue
            prog_rows.append((float(qoff[s, a]), int(morta), pid, rep))

    prog = np.array(prog_rows, dtype=float) if prog_rows else np.zeros((0, 4), dtype=float)
    return CalibrationData(bootql=np.array(bootql, dtype=float), prog=prog)


def plot_fig2c_calibration(
    calibration: CalibrationData,
    nbins: int = 100,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    MATLAB Fig 2C style mortality-vs-return calibration.
    """
    if calibration.prog.shape[0] == 0:
        raise ValueError("Calibration prog is empty.")

    a = calibration.prog[:, 0]
    m = calibration.prog[:, 1]
    qv = np.floor((a + 100.0) / (200.0 / nbins)).astype(int) + 1
    qv = np.clip(qv, 1, nbins)

    mean_mort = np.full(nbins, np.nan)
    sem_mort = np.full(nbins, np.nan)
    for i in range(1, nbins + 1):
        ii = qv == i
        if np.any(ii):
            vals = m[ii]
            mean_mort[i - 1] = np.nanmean(vals)
            sem_mort[i - 1] = np.nanstd(vals) / np.sqrt(np.sum(ii))

    # Light smoothing approximation (moving average) for readability.
    kernel = np.ones(7) / 7.0
    smooth = np.convolve(np.nan_to_num(mean_mort, nan=np.nanmean(mean_mort)), kernel, mode="same")

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(1, nbins + 1)
    ax.plot(x, mean_mort, color="b", linewidth=1)
    ax.plot(x, mean_mort + sem_mort, color="b", linewidth=0.5)
    ax.plot(x, mean_mort - sem_mort, color="b", linewidth=0.5)
    ax.plot(x, smooth, color="r", linewidth=1)
    ax.axhline(0.5, linestyle=":", color="k")
    ax.axvline(nbins / 2, linestyle=":", color="k")
    ax.set_xlim(1, nbins)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Return of actions")
    ax.set_ylabel("Mortality risk")
    ax.set_title("Fig 2C (MIMIC-only)")
    ax.grid(alpha=0.2)
    ax.set_box_aspect(1)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_fig2d_histogram(
    calibration: CalibrationData,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    MATLAB Fig 2D: per-patient average return histogram by mortality group.
    """
    if calibration.prog.shape[0] == 0:
        raise ValueError("Calibration prog is empty.")

    df = pd.DataFrame(calibration.prog, columns=["Qoff", "morta", "id", "rep"])
    grouped = df.groupby(["rep", "id"], as_index=False).agg(
        mean_Qoff=("Qoff", "mean"),
        mean_morta=("morta", "mean"),
    )

    surv = grouped[grouped["mean_morta"] < 0.5]["mean_Qoff"].values
    nonsurv = grouped[grouped["mean_morta"] >= 0.5]["mean_Qoff"].values
    edges = np.arange(-100, 105, 5)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(surv, bins=edges, alpha=0.7, label="Survivors", density=True, color="b")
    ax.hist(nonsurv, bins=edges, alpha=0.7, label="Non-survivors", density=True, color="r")
    ax.set_xlabel("Average return per patient")
    ax.set_ylabel("Probability")
    ax.set_title("Fig 2D (MIMIC-only)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    ax.set_box_aspect(1)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
