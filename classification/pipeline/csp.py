#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common Spatial Pattern (CSP) for motor imagery classification.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import numpy as np
import scipy
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances


def get_CSP_model(n_components=2, reg="ledoit_wolf", lr=True):
    """Get a CSP model and its search space for hyperparameters optimization."""
    if lr:
        search_space = {
            "feat__n_components": [2, 4],
            "feat__reg": ["oas", "lwf"],
            "classifier__C": [10, 1, 0.1, 0.01, 0.001],
        }
        model = Pipeline(
            steps=[
                ("feat", CSP(n_components, reg)),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
    else:
        search_space = {
            "feat__n_components": [2, 4],
            "classifier__C": scipy.stats.loguniform.rvs(1e-3, 1e3, size=10),
        }
        model = Pipeline(
            steps=[
                ("feat", CSP(n_components, reg)),
                ("classifier", SVC(probability=True)),
            ]
        )
    return model, search_space


class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Pattern (CSP) for motor imagery classification."""

    def __init__(
        self, n_components: int, reg="ledoit_wolf", component_order="mutual_info"
    ):
        """Initialize the CSP model."""
        self.n_components = n_components
        self.reg = reg
        if component_order not in ["alternate", "mutual_info"]:
            raise ValueError("component_order must be 'alternate' or 'mutual_info'")
        self.component_order = component_order

    def fit(self, X, y=None):
        """Compute CSP filters that optimize the variance ratios between trials of different classes.

        Args:
            X (numpy array): Filtered EEG data with zero-mean component in numpy format (trials, channels, samples).
            y (numpy array): EEG labels numpy format (trial).
        """
        if self.reg == "None" or "empirical":
            self.reg_ = "scm"
        elif self.reg == "ledoit_wolf":
            self.reg_ = "lwf"
        else:
            self.reg_ = self.reg
        n_components = self.n_components
        n_trials, n_channels, n_samples = X.shape
        labels_list = np.unique(y)
        self.n_classes = len(set(y))

        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        if self.n_classes > 2 and self.component_order == "alternate":
            raise ValueError(
                "component_order='alternate' requires two "
                "classes, but data contains {} classes; use "
                "component_order='mutual_info' "
                "instead.".format(self.n_classes)
            )

        # Estimate covariance matrices
        cov = Covariances(estimator=self.reg_)
        cov_matrices = cov.fit_transform(X)

        # Averaging covariance matrices of the same class
        cov_avg = np.zeros((self.n_classes, n_channels, n_channels), dtype=float)
        for c in range(self.n_classes):
            idxs = np.where(y == labels_list[c])[0]
            cov_avg[c] = np.sum(cov_matrices[idxs], axis=0) / len(idxs)

        # Generalized Eigenvalue Decomposition (compare classes 2 by 2)
        self.filters_ = []

        for c1 in range(self.n_classes):
            for c2 in range(c1 + 1, self.n_classes):
                # Solve C_1*U = C_2*U*D where D = diag(w) and U = vr
                eig_vals, U = scipy.linalg.eig(cov_avg[c1], cov_avg[c2])
                assert np.allclose(
                    cov_avg[c1] @ U - cov_avg[c2] @ U @ np.diag(eig_vals),
                    np.zeros((n_channels, n_channels)),
                    atol=1e-5,
                ), "CSP failed to find eigenvalues and eigenvectors that satisfy the equation."

                # Sort eigenvalues and pair i-th biggest with i-th smallest
                if self.component_order == "alternate" and self.n_classes == 2:
                    eig_vals_abs = np.abs(eig_vals)
                    sorted_idxs = np.argsort(eig_vals_abs)  # (increasing order)
                elif self.component_order == "mutual_info" and self.n_classes > 2:
                    mutual_info = self._compute_mutual_info(eig_vals, U)
                    sorted_idxs = np.argsort(mutual_info)[::-1]
                    # eig_vecs = U[:, sorted_idxs]
                elif self.component_order == "mutual_info" and self.n_classes == 2:
                    sorted_idxs = np.argsort(np.abs(eig_vals - 0.5))[::-1]
                    # eig_vecs = U[:, sorted_idxs]

                # Extract corresponding eigenvectors (spatial filters)
                chosen_idxs = np.zeros(2 * n_components, dtype=np.int16)
                chosen_idxs[:n_components] = sorted_idxs[:n_components]  # m smallest
                chosen_idxs[n_components : 2 * n_components] = sorted_idxs[
                    -n_components:
                ]  # m biggest
                eig_vecs = U[:, chosen_idxs]
                # Stack these 2*m spatial filters horizontally with the previous ones
                self.filters_ = (
                    eig_vecs
                    if len(self.filters_) == 0
                    else np.hstack([self.filters_, eig_vecs])
                )

        self.n_csp = 2 * n_components * np.sum(range(self.n_classes))
        assert self.filters_.shape == (
            n_channels,
            self.n_csp,
        ), "Got w of shape {} instead of {}.".format(
            self.filters_.shape, [n_channels, self.n_csp]
        )
        self.filters_ = self.filters_.astype(float)

        return self

    def transform(self, X: np.array) -> np.array:
        """Apply CSP transform on the input EEG data and compute the log-variance features.

        Args:
            - X (numpy array): EEG array of shape (n_trials, n_channels, n_samples).
        Output:
            - feats (numpy array): extracted features of shape (n_trials, n_csp).
        """
        n_trials, _, n_samples = X.shape
        n_csp = self.n_csp

        # Apply spatial transformation to input signal using the previously computed CSP filters
        X_transformed = np.array(
            [self.filters_.T @ X[trial_idx] for trial_idx in range(X.shape[0])]
        )
        assert X_transformed.shape == (n_trials, n_csp, n_samples)

        # Compute variance of each row of the CSP-transformed signal (apply on temporal axis)
        variances = np.array(
            [
                X_transformed[trial_idx] @ X_transformed[trial_idx].T
                for trial_idx in range(n_trials)
            ]
        )
        assert variances.shape == (n_trials, n_csp, n_csp)

        # Compute normalized log-variance features
        feats = np.array(
            [
                np.log10(np.diag(variances[trial_idx]) / np.trace(variances[trial_idx]))
                for trial_idx in range(n_trials)
            ]
        )
        assert feats.shape == (n_trials, n_csp)

        return feats

    def _compute_mutual_info(self, covs, eigen_vectors):
        """Compute the mutual information between the eigenvalues of the covariance matrices."""
        sample_weights = np.ones(len(covs))
        class_probas = sample_weights / sample_weights.sum()

        mutual_info = []
        for jj in range(eigen_vectors.shape[1]):
            aa, bb = 0, 0
            for cov, prob in zip(covs, class_probas):
                tmp = np.dot(np.dot(eigen_vectors[:, jj].T, cov), eigen_vectors[:, jj])
                aa += prob * np.log(np.sqrt(tmp))
                bb += prob * (tmp**2 - 1)
            mi = -(aa + (3.0 / 16) * (bb**2))
            mutual_info.append(mi)

        return mutual_info
