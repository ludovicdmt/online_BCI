#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Riemann geometry-based Motor Imagery classification using kernel filtering.
Implement of MDM classifier and Tangent Space + Logistic Regression.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import numpy as np

from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import BlockCovariances

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def get_kernelTS_model(
    reg="oas",
    metric="euclid",
    kernels=None,
    k=-1,
):
    """Get a kernel Tangent Space model and its search space for hyperparameters optimization."""
    search_space = {
        # "projection__reg": ["oas", "lwf", "scm"],
        # "projection__metric": ["riemann", "euclid", "logeuclid"],
        "classifier__C": [10, 1, 0.1, 0.01, 0.001]
    }

    if kernels is None:
        kernels = np.load(
            f"C:\\Users\\ludov\\Documents\\repos\\kernelsConstruction\\extra_kernels\\kernels_2014004.npy",
            allow_pickle=True,
        ).tolist()
        kernels = np.array(kernels)[[2, 3, 5, 6, 7]]

    model = Pipeline(
        [
            ("projection", kernelTS(kernels=kernels, reg=reg, metric=metric, k=k)),
            ("classifier", LogisticRegression()),
        ]
    )
    return model, search_space


class kernelTS(BaseEstimator, TransformerMixin):
    """Filter Bank using kernels estimated from beta nursts + Tangent Space for motor imagery classification."""

    def __init__(self, kernels=None, reg="oas", metric="euclid", k=-1):
        """Initialize the kernelTS model."""
        self.kernels = kernels
        self.reg = reg
        self.metric = metric
        self.k = k
        self.ss = RobustScaler()

    def fit(self, X: np.array, y: np.array):
        """Apply a bank of convolution with given kernels on the EEG signal, fit each CSP block and the MIBIF feature selection.

        Args:
            X (numpy array): EEG data in numpy format (trials, channels, samples).
            y (numpy array): EEG labels numpy format (trial).
        """
        if self.kernels is None:
            raise ValueError("Kernels has not been set.")
        self.block_size = X.shape[1]
        # Construct a new set of epochs with an augmented number of channels
        # that correspond to the filtered signal with each kernel
        data_filtered = np.empty((X.shape[0], 0, X.shape[2]))
        self.n_classes = len(set(y))
        for kernel in self.kernels:
            # Filter signal with a given kernel
            conv_kernel_data = np.copy(X)
            X_filt = np.apply_along_axis(
                np.convolve, -1, conv_kernel_data, kernel, mode="same"
            )
            # Stack filtered data as they are new EEG channels
            data_filtered = np.concatenate((data_filtered, X_filt), axis=1)

        # Extract covariance on blocks correponding to signal fitted with each kernel
        block_cov = BlockCovariances(
            estimator=self.reg, block_size=self.block_size
        ).transform(data_filtered)
        # Project this diagonal block covariance matrix in the tangent space
        self.ts = TangentSpace(metric=self.metric, tsupdate=False)
        feats = self.ts.fit_transform(block_cov)
        feats = self.ss.fit_transform(feats)
        assert feats.shape[0] == X.shape[0]
        self.n_components = feats.shape[1]

        # Feature selection based on MIBIF algorithm
        if self.k > 0:
            self.feature_selection = SelectKBest(
                mutual_info_classif,
                k=self.k,
            )
            self.feature_selection.fit(feats, y)

        return self

    def transform(self, X: np.array) -> np.array:
        """Apply a bank of kernels, then extract BlockCovariances, and finally apply Tangent Space.

        Args:
            - X (numpy array): array of shape (n_trials, n_channels, n_samples).
        Output:
            - selected_feats: extracted features of shape (n_trials, 2*k).
        """
        n_commponents = self.n_components
        data_filtered = np.empty((X.shape[0], 0, X.shape[2]))
        for kernel in self.kernels:
            # Filter signal with a given kernel
            conv_kernel_data = np.copy(X)
            X_filt = np.apply_along_axis(
                np.convolve, -1, conv_kernel_data, kernel, mode="same"
            )
            # Stack filtered data as they are new EEG channels
            data_filtered = np.concatenate((data_filtered, X_filt), axis=1)

        block_cov = BlockCovariances(
            estimator=self.reg, block_size=self.block_size
        ).transform(data_filtered)
        feats = self.ts.transform(block_cov)
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


if __name__ == "__main__":
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    X = np.random.rand(140, 9, 750)
    y = np.random.randint(2, size=140)
    kernels = np.random.rand(8, 64)

    # Trial single model
    ts = kernelTS(reg="oas", kernels=kernels, k=-1)
    ts.fit(X, y)
    ts.transform(X)

    # Trial in a pipeline
    model, search_space = get_kernelTS_model()
    grid = GridSearchCV(model, param_grid=search_space, cv=5, n_jobs=-1)
    grid.fit(X, y)
    grid.predict(X)
