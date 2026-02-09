#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small utilities used by AIClinician_core.py.
These mirror MATLAB helper behavior but use numpy/scipy.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree


def zscore_matlab(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB-like zscore with ddof=1. Returns (z, mean, std)."""
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0, ddof=1)
    z = (x - mu) / sigma
    return z, mu, sigma


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def drop_index_like_column(df: pd.DataFrame) -> pd.DataFrame:
    # If a CSV index was saved, it is often called "Unnamed: 0".
    if "Unnamed: 0" in df.columns:
        return df.drop(columns=["Unnamed: 0"])
    return df


def kmeans_best(
    x: np.ndarray,
    n_clusters: int,
    n_reps: int,
    max_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run kmeans multiple times and keep the best (lowest inertia)."""
    best_centroids = None
    best_inertia = math.inf
    for _ in range(n_reps):
        centroids, labels = kmeans2(
            x,
            n_clusters,
            iter=max_iter,
            minit="++",
            seed=rng.integers(1, 2**31 - 1),
        )
        diffs = x - centroids[labels]
        inertia = float(np.sum(diffs * diffs))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
    return best_centroids


def knn_search(centroids: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Nearest centroid index (0-based) using cKDTree (fastknnsearch analog)."""
    tree = cKDTree(centroids)
    _, idx = tree.query(x, k=1)
    return idx.astype(np.int64)


def split_train_test(
    icustayidlist: np.ndarray,
    icuuniqueids: np.ndarray,
    ncv: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split train/test by icustayid. Uses random group assignment per model.
    Returns boolean masks (train, test) aligned to icustayidlist.
    """
    n = icuuniqueids.size
    grp = np.floor(ncv * rng.random(n) + 1).astype(int)
    crossval = 1
    trainidx = icuuniqueids[grp != crossval]
    testidx = icuuniqueids[grp == crossval]
    train = np.isin(icustayidlist, trainidx)
    test = np.isin(icustayidlist, testidx)
    return train, test


def cluster_states(
    x_train: np.ndarray,
    prop: float,
    ncl: int,
    nclustering: int,
    max_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample training rows, run k-means, then assign all training rows to centroids.
    Returns centroids and training state indices (0-based).
    """
    sampl_mask = np.floor(rng.random(x_train.shape[0]) + prop).astype(bool)
    sampl = x_train[sampl_mask, :]
    centroids = kmeans_best(sampl, ncl, nclustering, max_iter, rng)
    idx = knn_search(centroids, x_train)
    return centroids, idx


def soften_policy(policy: np.ndarray, p: float) -> np.ndarray:
    """
    Soften policy probabilities:
    - Zero-prob actions get p mass uniformly.
    - Non-zero actions are reduced by p mass uniformly.
    Mirrors MATLAB logic in AIClinician_core_160219.m.
    """
    soft = policy.copy()
    n_states, _ = soft.shape
    for i in range(n_states):
        zeros = soft[i, :] == 0
        if np.any(zeros):
            z = p / np.sum(zeros)
            nz = p / np.sum(~zeros)
            soft[i, zeros] = z
            soft[i, ~zeros] = soft[i, ~zeros] - nz
    return soft
