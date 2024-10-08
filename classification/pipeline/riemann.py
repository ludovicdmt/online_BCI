#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Riemann geometry-based Motor Imagery classification.
Implement of MDM classifier and Tangent Space + Logistic Regression.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

from pyriemann.classification import MDM, FgMDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import numpy as np
from sklearn.preprocessing import RobustScaler


def get_MDM_model(reg="oas"):
    """Get a MDM model."""

    search_space = {
        "feat__estimator": ["oas", "lwf", "scm"],
        "classifier__metric": [
            dict(mean="euclid", distance="euclid", map="euclid"),
            dict(mean="riemann", distance="riemann", map="riemann"),
            dict(mean="logeuclid", distance="logeuclid", map="logeuclid"),
        ],
    }
    model = Pipeline(
        steps=[
            ("feat", Covariances(reg)),
            (
                "classifier",
                FgMDM(metric=dict(mean="euclid", distance="euclid", map="euclid")),
            ),
        ]
    )
    return model, search_space


def get_TS_model(reg="ledoit_wolf"):
    """Get Tangent Space + Linear Regression model."""
    if reg == "None" or "empirical":
        reg = "scm"
    elif reg == "ledoit_wolf":
        reg = "lwf"

    search_space = {
        "feat__estimator": ["oas", "lwf", "scm"],
        "projection__metric": ["riemann", "euclid", "logeuclid"],
        "classifier__C": [10, 1, 0.1, 0.01, 0.001],
    }
    model = Pipeline(
        steps=[
            ("feat", Covariances(reg)),
            ("projection", TangentSpace(metric="euclid", tsupdate=False)),
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return model, search_space
