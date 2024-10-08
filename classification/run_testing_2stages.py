#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run online classification on sliding windows with overlap.
The classificaiton is performed in two stages: first detection of MI vs rest+rebound,
then detection of left vs right MI.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

from pyexpat import model
import numpy as np
import scipy.signal
import json
import os
import argparse
import pickle
import time
import pylsl as lsl

from pynput.keyboard import Key, Controller
from udp_cyb import UDPClient

from PyQt5 import QtWidgets

from pyacq_ext.sosfilter import SosFilterTimestamps
from pyacq_ext.epochermultilabel import EpocherMultiLabel
from pyacq_ext.markers_lsl import LslMarkers
from pyacq_ext.eeg_lsl import LslEEG
from pyacq_ext.noisegenerator import NoiseGenerator
from pylsl import local_clock

from pylsl import StreamInfo, StreamOutlet

from pipeline.classifier_calib import load_model

from _utils_online import reject_artifacts


def ema(prices: list, period: int) -> float:
    """Calculate exponential moving average.

    Args:
        prices (list): List of prices.
        period (int): Number of periods.
    Returns:
        float: Exponential moving average.
    """
    if len(prices) < period:  # Can't calculate EMA at the start
        return np.mean(prices)
    else:
        return ema_helper(prices, period, (2 / (period + 1)), len(prices))


def ema_helper(prices: list, N: int, k: float, length: int) -> float:
    """Recursive function to calculate EMA.

    Args:
        prices (list): List of prices.
        N (int): Number of periods.
        k (float): Smoothing factor.
        length (int): Length of the list of prices.
    Returns:
        float: Exponential moving average.
    """
    if len(prices) == length - N:
        return prices[0]
    res_ema = prices[N - 1]
    for t in range(N, length):
        res_ema = prices[t] * k + res_ema * (1 - k)
    return res_ema


def testing(
    w_length: float,
    filtering: list,
    idx_electrodes: list,
    model_name: str,
    model_name2: str,
) -> None:
    """Collect data for a calibration session.

    Args:
        w_length (float): Length of the sliding window.
        filtering (list): List of filtering parameters.
        idx_electrodes (list): List of indices of the electrodes to keep.
        model_name (str): Name of the model to use for stage 1.
        model_name2 (str): Name of the model to use for stage 2.
    """
    # Stream predictions to the GUI
    info = StreamInfo(
        name="PredictionMarkers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="presentationPC",
    )

    # in main App
    app = QtWidgets.QApplication([])

    # Noise generator node
    # Replay of EEG data collected during calibration
    # ng = NoiseGenerator()
    # ng.configure(chunksize = 30, sample_rate=250, data_path = 'C:\\Users\\ludov\\Documents\\pre_tests\\S2\\0724\\calib.xdf')
    # ng.output.configure(protocol="tcp", transfermode="plaindata")
    # ng.initialize()
    # ng.start()

    # keyboard = Controller()
    t = time.perf_counter()
    # EEG data node
    ba = LslEEG()
    ba.configure(idx_electrodes=idx_electrodes)
    ba.outputs["signals"].configure(
        protocol="tcp",
        interface="127.0.0.1",
        transfermode="plaindata",
    )
    ba.initialize()
    sample_rate = ba.outputs["signals"].spec["sample_rate"]

    info.desc().append_child_value("sample_rate", f"{sample_rate}")
    outlet = StreamOutlet(info)

    # Triggers node
    trig = LslMarkers(marker_name="MarkersEmulator")
    trig.configure()
    trig.outputs["triggers"].configure(
        protocol="tcp",
        transfermode="plaindata",
    )
    trig.initialize()

    # Filtering node
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
    params = {"tmin": 0, "tmax": w_length}

    epocher = EpocherMultiLabel(params)
    epocher.configure()
    # epocher.inputs['signals'].connect(ba.output)
    epocher.inputs["signals"].connect(filt.output)
    epocher.inputs["triggers"].connect(trig.outputs["triggers"])
    epocher.initialize()

    # Load UPDClient
    client = UDPClient(ip="localhost", port=59075)
    client.start()

    # Load model
    path_to_model = f"saved_models/{model_name}_stage1.pickle"
    clf_M1 = load_model(path_to_model)
    path_to_model = f"saved_models/{model_name2}_stage2.pickle"
    clf_M2 = load_model(path_to_model)
    path_to_EA = "saved_models\\refEA.pickle"
    with open(path_to_EA, "rb") as f:
        R_inv = pickle.load(f)
    if model_name == "MDM":
        file = f"saved_models\\tresh_M1.pickle"
        with open(file, "rb") as f:
            tresh = pickle.load(f)

    # Load data to compute amplitude treshold
    file = "saved_models\\rebound_data.pickle"
    with open(file, "rb") as f:
        data_dict = pickle.load(f)
    tresh_amp = np.percentile(data_dict["calib_data"], q=95)

    print("Model loaded")

    tresh_M1 = 0.55
    tresh_M2 = 0.55
    forward = 0.5

    pred_history = [0.5]  # Initialize with -1 to avoid triggering at the start

    # file_asr = "saved_models\\asr.pickle"
    # # Load asr
    # with open(file_asr, "rb") as f:
    #     asr = pickle.load(f)

    def on_new_chunk(label, new_chunk, eeg_timestamp) -> None:
        """Trigger when a new chunk from the epocher is available.

        Args:
            label (str): Label code. Unused here on testing, only -1.
            new_chunk (array): New chunk of data.
        """
        nonlocal pred_history, tresh_M1, client  # wait,#, client #, asr#, client

        if client.control == False:  # Don't predict if the game is not running
            return

        # Pre-processing
        new_chunk = new_chunk.T * 1e-6  # Convert from uV to V
        # new_chunk = new_chunk[idx_electrodes, :] # Electrode selection
        ref = np.mean(
            new_chunk, axis=0, keepdims=True
        )  # Common average re-referencing (CAR)
        new_chunk = new_chunk - ref
        # new_chunk = asr.transform(new_chunk) # asr preprocess
        new_chunk = new_chunk[np.newaxis, :, :]
        # Baseline removal
        # new_chunk = np.mean(new_chunk, axis=-1, keepdims=True)
        new_chunk = np.einsum("ij,tjk->tik", R_inv, new_chunk)  # Euclidean Alignment

        # Check for NaN values
        if np.isnan(new_chunk).any():
            return

        # Artifact rejection
        if reject_artifacts(
            new_chunk, amplitude_threshold=tresh_amp, zscore_threshold=4.0
        ):
            print("Artifact detected")
            return

        # Classify the data
        pred_M1 = clf_M1.predict_proba(new_chunk)[0][1]  # First stage (MI vs rebound)
        pred = clf_M2.predict_proba(new_chunk)[0]  # Second stage (left vs right)
        print(f"Proba M1: {pred_M1}, Proba M2 {pred}")

        # Resting class filtering for left/right MI detection
        # if (model_name == "MDM") and (pred_M1) > 0.5:
        # cov = clf_M1["feat"].fit_transform(new_chunk)
        # dist = clf_M1["classifier"].transform(cov)
        # If too far from the center of class for MI
        # It is an outlier that is missclassified
        # if dist[0, 1] > tresh_MDM:
        #     pred_M1 = 0

        # Handle left vs right MI
        if pred_M1 > tresh_M1:
            pred_class = np.argmax(pred)
            pred_proba = np.max(pred)
            pred_history.append(pred_proba if pred_class == 1 else 1 - pred_proba)

        # No clear MI detected
        else:
            outlet.push_sample([f"{pred_M1},-1"])
            # pred_window.append(np.squeeze(new_chunk))
            # pred_label.append(-1)
            return

        # Update prediction history and calculate EMA
        pred_history.pop(0)
        ema_pred = ema(pred_history, 1)

        # Make a prediction based on the EMA
        print(f"Ema pred: {ema_pred}, proba: {pred_proba}")
        if ema_pred > tresh_M2 + 0.1:
            print(f"Predict 1")  # at {t}")
            client.send_command(0, 0, +0.2, 0)  # Rotate right
            outlet.push_sample([f"{pred_M1},{ema_pred}"])
        elif ema_pred < tresh_M2:
            print(f"Predict 0 ")  # at {t}")
            client.send_command(0, 0, -0.2, 0)  # Rotate left
            outlet.push_sample([f"{pred_M1},{ema_pred}"])
        else:
            outlet.push_sample([f"{pred_M1},-1"])

    # On new chunk received
    epocher.new_chunk.connect(on_new_chunk)

    trig.start()
    ba.start()
    filt.start()
    epocher.start()

    offset = time.perf_counter() - t
    print(f"Offset: {offset}")
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

    args = parser.parse_args()
    config_path = os.path.join(path, args.file)

    with open(config_path, "r") as config_file:
        params = json.load(config_file)

    # Experimental params
    trial_n = params["trial_number"]
    block_number = params["block_number"]
    tmin = params["tmin"]
    tmax = params["tmax"]
    w_length = tmax - tmin
    nb_epochs = block_number * trial_n
    model_name = params["model_name1"]
    model_name2 = params["model_name2"]
    filtering = params["filtering"]
    idx_electrodes = params["electrodes"]
    if idx_electrodes == "all":
        idx_electrodes = [i for i in range(32)]
    testing(w_length, filtering, idx_electrodes, model_name, model_name2)
