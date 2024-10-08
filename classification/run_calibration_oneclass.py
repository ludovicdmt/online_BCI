#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect data for a calibration session with a single class.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import numpy as np
import scipy.signal
import pickle
import json
import os
import random
import joblib
import argparse
from scipy.linalg import sqrtm, inv


from PyQt5 import QtWidgets

from pyacq_ext.sosfilter import SosFilterTimestamps
from pyacq_ext.epochermultilabel import EpocherMultiLabel
from pyacq_ext.markers_lsl import LslMarkers
from pyacq_ext.eeg_lsl import LslEEG
from pyacq_ext.noisegenerator import NoiseGenerator

from pipeline.classifier_calib import save_model, train_model
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance


def calibration(
    nb_epochs: int,
    tmin: float,
    tmax: float,
    iti: float,
    epoch_duration: str,
    model_name: str,
    filtering: list,
    idx_electrodes: list,
) -> None:
    """Collect data for a calibration session.

    Args:
        nb_epochs (int): Number of epochs MI to collect for calibration. Some ITI epochs will be added.
        tmin (float): Start time of the epoch.
        tmax (float): End time of the epoch.
        iti (float): Inter-trial interval.
        epoch_duration (str): Duration of the MI epoch.
        model_name (str): Name for the first stage of classification (rebound).
        filtering (list): List of filtering parameters.
        idx_electrodes (list): List of indices of the electrodes to keep.
    """
    # in main App
    app = QtWidgets.QApplication([])

    # Simulate EEG Node
    # ng = NoiseGenerator()
    # ng.configure(sample_rate=250)
    # ng.output.configure(protocol="tcp", transfermode="plaindata")
    # ng.initialize()
    # ng.start()

    # EEG data node
    ba = LslEEG()
    ba.configure(idx_electrodes=idx_electrodes)
    ba.outputs["signals"].configure(
        protocol="tcp",
        interface="127.0.0.1",
        transfermode="plaindata",
    )
    ba.initialize()

    # Triggers node
    trig = LslMarkers(marker_name="MotorImageryMarkers")  # MotorImageryMarkers
    trig.configure()
    trig.outputs["triggers"].configure(
        protocol="tcp",
        transfermode="plaindata",
    )
    trig.initialize()

    # Filtering node
    sample_rate = ba.outputs["signals"].spec["sample_rate"]
    # Compute coefficients for a cascaded 2nd-order sections (sos) bandpass Butterworth filter
    coefficients = scipy.signal.iirfilter(
        30,
        [filtering[0] / sample_rate * 2, filtering[1] / sample_rate * 2],
        btype="bandpass",
        ftype="butter",
        output="sos",
    )
    filt = SosFilterTimestamps()
    filt.configure(coefficients=coefficients)
    # This node is connected to the output of the device node
    filt.input.connect(ba.outputs["signals"])
    filt.output.configure(
        protocol="tcp",
        interface="127.0.0.1",
        transfermode="plaindata",
    )
    filt.initialize()

    # Epocher Node
    ## Parameters to slice epochs
    trial_length = tmax - tmin
    if epoch_duration + iti - 0.5 >= 2 * epoch_duration:
        tend = epoch_duration + iti - 0.5
    elif epoch_duration + iti >= 2 * epoch_duration:
        tend = epoch_duration + iti
    else:
        raise ValueError("ITI is not long enough to match epoch duration")
    params = {
        "tmin": tmin,
        "tmax": tend,
    }  # Correspond to the full MI epoch + duration of the ITI
    print(f"Epocher params: {params}")

    epocher = EpocherMultiLabel(
        params, max_xsize=tend + 1
    )  # Add 1 seconds to buffersize as a security margin
    epocher.configure()
    epocher.inputs["signals"].connect(filt.output)
    epocher.inputs["triggers"].connect(trig.outputs["triggers"])
    epocher.initialize()

    # A list to store the data
    calib_data = []
    labels = []
    data_iti = []

    cmd = model_name.split("_")[-1]

    # kernels = np.load(
    #     "C:\\Users\\ludov\\Documents\\repos\\MIonline\\classification\\pipeline\\zhou_2016_kernels.npy",
    #     allow_pickle=True,
    # )

    # file_asr = "saved_models\\asr.pickle"
    # # Load asr
    # with open(file_asr, "rb") as f:
    #     asr = pickle.load(f)

    def on_new_chunk(label, new_chunk) -> None:
        """Trigger when a new chunk from the epocher is available.

        Args:
            label (str): Label code.
            new_chunk (array): New chunk of data.
        """
        nonlocal trial_length, data_iti, calib_data, labels, cmd  # , asr
        # From uV to V
        # electrode selection
        new_chunk = new_chunk.T * 1e-6  # n_channels, n_samples
        # Electrode selection
        # new_chunk = new_chunk[idx_electrodes, :]

        # Common average re-referencing (CAR)
        ref = np.mean(new_chunk, axis=0, keepdims=True)
        new_chunk = new_chunk - ref

        # asr preprocess
        # new_chunk = asr.transform(new_chunk)
        new_chunk = new_chunk[np.newaxis, :, :]

        chunk_MI = new_chunk[
            :, :, : int(trial_length * sample_rate)
        ]  # Retrieved epochs starts at tmin after the trigger

        chunk_MI -= np.mean(chunk_MI, axis=-1, keepdims=True)  # Baseline removal
        calib_data.append(chunk_MI)  # (n_epoch, n_channel, n_trials)
        labels.append(label)

        # ITI data for rebound and rest starts 0.5s before epoch_duration length and last trial_length
        chunk_iti = new_chunk[
            :,
            :,
            int((epoch_duration - tmin - 0.5) * sample_rate) : int(
                (epoch_duration - tmin - 0.5 + trial_length) * sample_rate
            ),
        ]
        chunk_iti -= np.mean(chunk_iti, axis=-1, keepdims=True)  # Baseline removal
        data_iti.append(chunk_iti)
        # Save the calib data
        data_dict = {"calib_data": np.squeeze(np.array(calib_data)), "labels": labels}
        file_calib = f"saved_models\\calib_data_{cmd}.pickle"
        with open(file_calib, "wb") as f:
            pickle.dump(data_dict, f)
        # Also save the ITI data to be sure that we can train model at the end, even if it crashes before
        file_iti = f"saved_models\\iti_data_{cmd}.pickle"
        with open(file_iti, "wb") as f:
            pickle.dump(np.squeeze(data_iti), f)
        print(
            "Calibration data saved. Number of calibration data: {}".format(
                len(calib_data)
            )
        )
        if len(calib_data) == nb_epochs:
            print("Calibration finished.")
            print(f"Class balance : {np.mean(labels)}")
            # Remove epochs every 5 events because they include keyboard press
            # indices_to_keep = np.delete(np.arange(len(data_iti)), np.arange(4, len(data_iti), 5))
            # # Use the indices to index into the events array
            # data_iti = np.array(data_iti)[indices_to_keep]
            # Downsample the MI data to have the same amount of epochs as ITI epochs
            labels_rebound = np.zeros(len(data_iti))  # Rebound label is 0

            # Create dataset to distinguish between MI and rebound
            # Downsample MI epochs to the same number of rebound epochs
            # idx = np.arange(0, len(calib_data), 1)
            # idx = np.random.choice(idx, len(data_iti), replace=False)
            epochs_MI_stage1 = np.copy(calib_data)  # [idx]
            labels_MI_stage1 = np.zeros(len(epochs_MI_stage1)) + 1  # MI label is 1

            # Combine MI and rebound epochs
            combined_epochs = np.squeeze(
                np.concatenate((epochs_MI_stage1, data_iti), axis=0)
            )
            combined_labels = np.concatenate((labels_MI_stage1, labels_rebound), axis=0)

            # Euclidean aligment
            # A whitening that will re-center data and allow transfer learning
            cov_matrices = Covariances(estimator="oas").fit_transform(combined_epochs)
            refEA = mean_covariance(cov_matrices, metric="euclid")
            R_inv = sqrtm(inv(refEA))
            combined_epochs = np.einsum("ij,tjk->tik", R_inv, combined_epochs)
            print(f"Calib data shape: {np.array(calib_data).shape}")
            calib_data = np.einsum(
                "ij,tjk->tik", R_inv, np.squeeze(np.array(calib_data))
            )

            # Store everything
            file = f"saved_models\\refEA_{cmd}.pickle"
            with open(file, "wb") as f:
                pickle.dump(R_inv, f)
            data_dict = {
                "calib_data": np.array(combined_epochs),
                "labels": combined_labels,
            }
            file = f"saved_models\\rebound_data_{cmd}.pickle"
            with open(file, "wb") as f:
                pickle.dump(data_dict, f)
            data_dict = {
                "calib_data": np.squeeze(np.array(calib_data)),
                "labels": labels,
            }
            with open(file_calib, "wb") as f:
                pickle.dump(data_dict, f)

            # Train a model to detect rebound
            model_name_M1 = model_name
            grid = train_model(
                model_name=model_name_M1,
                X=combined_epochs,
                y=combined_labels,
                sample_rate=sample_rate,
                lr=False,
            )
            clf = grid.best_estimator_
            if model_name == "MDM":
                cov = clf["feat"].fit_transform(combined_epochs)
                dist = clf["classifier"].transform(cov)
                tresh = np.percentile(dist[combined_labels == 1, 1], 95)
                file = f"saved_models\\tresh_M1_{cmd}.pickle"
                with open(file, "wb") as f:
                    pickle.dump(tresh, f)
            file = f"saved_models\\{model_name_M1}.pickle"
            joblib.dump(grid, file)

            # Write to a file to signal the other script
            with open("saved_models\calibration_done.txt", "w") as f:
                f.write("Calibration done")
            ba.stop()
            trig.stop()
            filt.stop()
            epocher.stop()
            app.quit()

    epocher.new_chunk.connect(on_new_chunk)

    ba.start()
    filt.start()
    trig.start()
    epocher.start()
    app.exec_()


if __name__ == "__main__":
    path = os.getcwd()

    parser = argparse.ArgumentParser(description="Config file name")
    parser.add_argument(
        "-f",
        "--file",
        metavar="ConfigFile",
        type=str,
        default="config.json",
        help="Name of the config file. Default: %(default)s.",
    )

    parser.add_argument(
        "-c",
        "--command",
        metavar="Command",
        type=str,
        default="forward",
        help="Train for the backward or forward direction. Default: %(default)s.",
    )

    args = parser.parse_args()
    config_path = os.path.join(path, args.file)

    with open(config_path, "r") as config_file:
        params = json.load(config_file)

    # Experimental params
    trial_n = params["trial_number"]
    block_number = params["block_number"]
    tmin = params["tmin"]
    tmax = params["tmax"]
    iti = params["iti_duration"]
    assert tmin < tmax, "tmin should be less than tmax"
    assert iti >= (tmax - tmin), "iti should be greater than MI epoch duration"
    epoch_duration = params["epoch_duration"]
    assert epoch_duration >= (
        tmax - tmin
    ), "epoch_duration should be greater than MI epoch duration"
    nb_epochs = block_number * trial_n  # MI only trials
    model_name = params["model_name_oneclass"] + "_" + str(args.command).strip()
    filtering = params["filtering"]
    idx_electrodes = params["electrodes"]
    if idx_electrodes == "all":
        idx_electrodes = [i for i in range(32)]
    calibration(
        nb_epochs,
        tmin,
        tmax,
        iti,
        epoch_duration,
        model_name,
        filtering,
        idx_electrodes,
    )
