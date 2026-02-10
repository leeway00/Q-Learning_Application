#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small utilities used by AIClinician_core.py.
These mirror MATLAB helper behavior but use numpy/scipy.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Tuple
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import pairwise_distances_argmin


def _init_cuml():
    import cupy as cp  # type: ignore
    from cuml.cluster import KMeans as cuKMeans  # type: ignore

    if cp.cuda.runtime.getDeviceCount() < 1:
        return False, None, None

    return True, cp, cuKMeans

_CUM_L_AVAILABLE, _CP, _CUKMEANS = _init_cuml()

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
    """Run k-means multiple times with k-means++ init and keep best inertia.

    Set AI_CLINICIAN_USE_CUML=1 to try cuML GPU KMeans if available.
    """
    best_centroids = None
    best_inertia = math.inf
    for _ in range(n_reps):
        if _CUM_L_AVAILABLE:
            seed = int(rng.integers(1, 2**31 - 1))
            x_gpu = _CP.asarray(x, dtype=_CP.float32)
            kmeans = _CUKMEANS(
                n_clusters=n_clusters,
                init="k-means++",
                max_iter=max_iter,
                n_init=1,
                random_state=seed,
            )
            kmeans.fit(x_gpu)
            inertia = float(kmeans.inertia_)
            centroids = _CP.asnumpy(kmeans.cluster_centers_)
        else:
            init_centers, _ = kmeans_plusplus(
                x,
                n_clusters=n_clusters,
                random_state=int(rng.integers(1, 2**31 - 1)),
            )
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init_centers,
                n_init=1,
                max_iter=max_iter,
                random_state=int(rng.integers(1, 2**31 - 1)),
                algorithm="lloyd",
            )
            kmeans.fit(x)
            inertia = float(kmeans.inertia_)
            centroids = kmeans.cluster_centers_
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
    return best_centroids


def knn_search(centroids: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Nearest centroid index (0-based) using exact pairwise distances."""
    idx = pairwise_distances_argmin(x, centroids, metric="euclidean")
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
        if np.all(zeros):
            # Unseen state: fall back to uniform distribution to avoid NaNs.
            soft[i, :] = 1.0 / soft.shape[1]
            continue
        if np.any(zeros):
            z = p / np.sum(zeros)
            nz = p / np.sum(~zeros)
            soft[i, zeros] = z
            soft[i, ~zeros] = soft[i, ~zeros] - nz
    return soft
