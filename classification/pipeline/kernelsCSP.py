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
from mne.decoding import CSP as CSP_mne
from sklearn.preprocessing import RobustScaler

from .preprocessing import kernelConvo
from .csp import CSP


def get_kernelCSP_model(
    n_components=2,
    reg="ledoit_wolf",
    kernels=None,
    k=-1,
):
    """Get a kernelCSP model and its search space for hyperparameters optimization."""
    search_space = {
        "classifier__C": scipy.stats.loguniform.rvs(1e-3, 1e3, size=10),
        "feat__n_components": [2, 4],
    }

    if kernels is None:
        kernels = np.load(
            "C:\\Users\\ludov\\Documents\\repos\\MIonline\\classification\\pipeline\\zhou_2016_kernels.npy",
            allow_pickle=True,
        )
    model = Pipeline(
        [
            (
                "feat",
                kernelCSP(
                    kernels=kernels,
                    n_components=n_components,
                    k=k,
                    reg=reg,
                ),
            ),
            ("classifier", SVC(kernel="rbf", gamma="scale", probability=True)),
        ]
    )
    return model, search_space


class kernelCSP(BaseEstimator, TransformerMixin):
    """Filter Bank Common Spatial Pattern (FBCSP) for motor imagery classification."""

    def __init__(
        self, n_components=2, reg="None", kernels=None, k=-1, CSP_implem="numpy"
    ):
        """Initialize the kernelCSP model."""
        self.n_components = n_components
        self.kernels = kernels
        if self.kernels is None:
            raise ValueError("f_kernels must be a list of kernels")
        self.k = k
        self.reg = reg
        self.CSP_implem = CSP_implem
        self.ss = RobustScaler()

    def fit(self, X: np.array, y: np.array):
        """Apply a bank of convolution with given kernels on the EEG signal, fit each CSP block and the MIBIF feature selection.

        Args:
            X (numpy array): EEG data in numpy format (trials, channels, samples).
            y (numpy array): EEG labels numpy format (trial).
        """
        self.csp_blocks = []
        feats = []
        self.n_classes = len(set(y))
        for kernel in self.kernels:
            # Filter signal with a given kernel

            X_filt = kernelConvo(
                X,
                kernel=kernel,
            )

            # Apply CSP and save block (with trained filters)
            if self.CSP_implem == "mne":
                csp = CSP_mne(n_components=self.n_components, reg=self.reg)
            elif self.CSP_implem == "numpy":
                csp = CSP(n_components=self.n_components, reg=self.reg)
            csp_feats = csp.fit_transform(X_filt, y)  # (n_trials, n_csp)

            feats = csp_feats if len(feats) == 0 else np.hstack([feats, csp_feats])
            self.csp_blocks.append(csp)

        # self.n_csp = 2 * self.n_components * np.sum(range(self.n_classes))
        # self.n_feats_tot = len(self.kernels) * self.n_csp
        feats = self.ss.fit_transform(feats)
        assert feats.shape[0] == X.shape[0]

        # Feature selection based on MIBIF algorithm
        if self.k > 0:
            self.feature_selection = SelectKBest(
                mutual_info_classif,
                k=self.k,
            )
            self.feature_selection.fit(feats, y)

        return self

    def transform(self, X: np.array) -> np.array:
        """Apply bank of convolution and CSP transform to input EEG signal and selection features.

        Args:
            - X (numpy array): array of shape (n_trials, n_channels, n_samples).
        Output:
            - selected_feats: extracted features of shape (n_trials, 2*k).
        """
        feats = []
        n_commponents = self.n_components
        for kernel_idx, kernel in enumerate(self.kernels):
            # Filter signal with a given kernel
            X_filt = kernelConvo(
                X,
                kernel=kernel,
            )

            # Compute CSP features associated to the given kernel
            csp_feats = self.csp_blocks[kernel_idx].transform(X_filt)

            # Concatenate CSP features
            feats = csp_feats if len(feats) == 0 else np.hstack([feats, csp_feats])

        feats = self.ss.transform(feats)
        # Select k best features and their pairs
        selected_feats = []
        if self.k > 0:
            select_idxs = self.feature_selection.get_support(indices=True)
            for idx in select_idxs:
                # feature index inside 2*m block of features
                sub_idx = idx % (2 * n_commponents)
                # corresponding paired index
                comp_idx = idx + (
                    n_commponents if sub_idx < n_commponents else -n_commponents
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
