#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utils to reject artifacted windows.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import numpy as np


def reject_artifacts(
    new_chunk: np.array,
    window_size=0.5,
    overlap=0.25,
    sfreq=250,
    amplitude_threshold=5e-5,
    zscore_threshold=4.0,
) -> bool:
    """Using amplitude thresholding and z-score determine if a window is artifacted.

    Args:
        new_chunk (np.array): chunk of eeg data (n_channels, n_samples).
        window_size (float): size (in s) of the sub-window to search for artifact.
        overlap (float): overlap (in s) between sub windows.
        sfreq (float): sampling frequency in Hz
        amplitude_threshold (float): amplitude threshold in micro-volts for artifact rejection
        zscore_threshold (float): z-score threshold for artifact rejection

    Outputs:
        bool: True if the window contains artifacts, False otherwise
    """
    assert (
        window_size > overlap
    ), "Window size (in s) should be larger than overlap (in s)."
    window_samples = int(window_size * sfreq)
    overlap_samples = int(overlap * sfreq)
    step_size = window_samples - overlap_samples

    for i in range(0, new_chunk.shape[1], step_size):
        window_data = new_chunk[:, i : i + step_size]
        # Amplitude thresholding
        if np.abs(window_data).reshape(-1).max() > amplitude_threshold:
            print(
                f"Max: {np.abs(window_data).reshape(-1).max()}, treshold: {amplitude_threshold}"
            )
            print("Artifact detected based on amplitude threshold")
            return True

        # Z-score detection
        mean = np.mean(window_data, axis=1, keepdims=True)
        std = np.std(window_data, axis=1, keepdims=True)
        z_scores = (window_data - mean) / std
        if np.any(np.abs(z_scores) > zscore_threshold):
            print("Artifact detected based on z-score threshold")
            return True

    return False
