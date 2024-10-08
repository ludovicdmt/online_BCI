#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Filter Bank Common Spatial Pattern (FBCSP) for motor imagery classification.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import filtering
from .csp import CSP


def get_FBCSP_model(
    fs=500, n_components=2, reg="ledoit_wolf", f_order=2, f_type="butter", k=-1
):
    """Get a FBCSP model and its search space for hyperparameters optimization."""
    search_space = {
        "feat__regularize_cov": [False, True],
        "feat__f_order": (2, 5),
        "classifier__C": scipy.stats.loguniform.rvs(1e-3, 1e3, size=10),
    }
    model = Pipeline(
        [
            (
                "feat",
                FBCSP(
                    fs=fs,
                    f_type=f_type,
                    f_order=f_order,
                    n_components=n_components,
                    k=k,
                    reg=reg,
                ),
            ),
            ("classifier", SVC(kernel="rbf", gamma="scale", probability=True)),
        ]
    )
    return model, search_space


class FBCSP(BaseEstimator, TransformerMixin):
    """Filter Bank Common Spatial Pattern (FBCSP) for motor imagery classification."""

    def __init__(
        self,
        fs=500,
        n_components=2,
        reg="None",
        f_order=2,
        f_type="butter",
        k=-1,
    ):
        """Initialize the FBCSP model."""
        self.fs = fs
        self.n_components = n_components
        self.freq_bands = [
            [4, 8],
            [8, 12],
            [12, 16],
            [16, 20],
            [20, 24],
            [24, 28],
            [28, 32],
            [32, 36],
            [36, 40],
        ]
        self.f_order = f_order
        self.f_type = f_type
        self.k = k
        self.reg = reg

    def fit(self, X: np.array, y: np.array):
        """Apply filter bank to input EEG signal, fit each CSP block and the MIBIF feature selection.

        Args:
            X (numpy array): EEG data in numpy format (trials, channels, samples).
            y (numpy array): EEG labels numpy format (trial).
        """
        self.csp_blocks = []
        feats = []
        self.n_classes = len(set(y))
        for f_band in self.freq_bands:
            # Filter signal on given frequency band
            X_filt = filtering(
                X,
                fs=self.fs,
                f_order=self.f_order,
                f_low=f_band[0],
                f_high=f_band[1],
                f_type=self.f_type,
            )

            # Apply CSP and save block (with trained filters)
            csp = CSP(n_components=self.n_components, reg=self.reg)
            csp_feats = csp.fit_transform(X_filt, y)  # (n_trials, n_csp)

            feats = csp_feats if len(feats) == 0 else np.hstack([feats, csp_feats])
            self.csp_blocks.append(csp)

        self.n_csp = self.csp_blocks[-1].n_csp
        self.n_feats_tot = len(self.freq_bands) * self.n_csp
        assert feats.shape == (X.shape[0], self.n_feats_tot)

        # Feature selection based on MIBIF algorithm
        if self.k > 0:
            self.feature_selection = SelectKBest(
                mutual_info_classif,
                k=self.k,
            )
            self.feature_selection.fit(feats, y)

        return self

    def transform(self, X: np.array) -> np.array:
        """Apply filter bank and CSP transform to input EEG signal and selection features.

        Args:
            - X (numpy array): array of shape (n_trials, n_channels, n_samples).
        Output:
            - selected_feats: extracted features of shape (n_trials, 2*k).
        """
        feats = []
        n_components = self.n_components
        for band_idx, f_band in enumerate(self.freq_bands):
            # Filter input signal on given frequency band
            X_filt = filtering(
                X,
                fs=self.fs,
                f_order=self.f_order,
                f_low=f_band[0],
                f_high=f_band[1],
                f_type=self.f_type,
            )

            # Compute CSP features associated to current frequency band
            csp_feats = self.csp_blocks[band_idx].transform(X_filt)

            # Concatenate CSP features
            feats = csp_feats if len(feats) == 0 else np.hstack([feats, csp_feats])

        # Select k best features and their pairs
        selected_feats = []
        if self.k > 0:
            select_idxs = self.feature_selection.get_support(indices=True)
            for idx in select_idxs:
                # feature index inside 2*m block of features
                sub_idx = idx % (2 * n_components)
                # corresponding paired index
                comp_idx = idx + (
                    n_components if sub_idx < n_components else -n_components
                )

                pair_feats = np.hstack(
                    [feats[:, idx].reshape(-1, 1), feats[:, comp_idx].reshape(-1, 1)]
                )
                selected_feats = (
                    pair_feats
                    if len(selected_feats) == 0
                    else np.hstack([selected_feats, pair_feats])
                )
        else:
            selected_feats = feats
        return selected_feats
