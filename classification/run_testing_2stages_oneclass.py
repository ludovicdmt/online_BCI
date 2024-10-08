#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run online classification on sliding windows with overlap for a single class BCI (rest vs MI).

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
from pipeline import csp


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
    w_length: float, filtering: list, idx_electrodes: list, model_name: str
) -> None:
    """Collect data for a calibration session.

    Args:
        w_length (float): Length of the sliding window.
        filtering (list): List of filtering parameters.
        idx_electrodes (list): List of indices of the electrodes to keep.
        model_name (str): Name of the model to use.
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
    # ng.configure(chunksize = 50, sample_rate=250, data_path = 'C:\\Users\\ludov\\Documents\\pre_tests\\S0\\4sMI_4-5sITI_70trials_0403.xdf')
    # ng.output.configure(protocol="tcp", transfermode="plaindata")
    # ng.initialize()
    # ng.start()

    keyboard = Controller()
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
    time.sleep(2)
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
    # client = UDPClient(ip="localhost", port=59075)
    # client.start()

    # Load model
    path_to_model = f"saved_models/{model_name}_stage1_tongue.pickle"
    clf_M1 = load_model(path_to_model)
    path_to_EA = "saved_models\\refEA_tongue.pickle"
    with open(path_to_EA, "rb") as f:
        R_inv = pickle.load(f)
    if model_name == "MDM":
        file = f"saved_models\\tresh_M1_tongue.pickle"
        with open(file, "rb") as f:
            tresh = pickle.load(f)
    print("Model loaded")
    pred_history = 50 * [0.0]  # Initialize with 0 to avoid triggering at the start

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
        nonlocal tresh  # , client#, asr
        # Predict
        # From uV to V
        new_chunk = new_chunk.T * 1e-6  # n_channels, n_samples
        # Electrode selection
        # new_chunk = new_chunk[idx_electrodes, :]

        # Common average re-referencing (CAR)
        ref = np.mean(new_chunk, axis=0, keepdims=True)
        new_chunk = new_chunk - ref

        # asr preprocess
        # new_chunk = asr.transform(new_chunk)
        new_chunk = new_chunk[np.newaxis, :, :]
        new_chunk -= np.mean(new_chunk, axis=-1, keepdims=True)  # Baseline removal
        # Euclidean Alignment
        new_chunk = np.einsum("ij,tjk->tik", R_inv, new_chunk)

        # If there is NAN stop here
        if np.isnan(new_chunk).any():
            return

        # Classification of tongue vs rebound + rest
        tresh = 0.505
        pred_M1 = clf_M1.predict_proba(new_chunk)[0][1]
        if (pred_M1) > tresh:
            cov = clf_M1["feat"].fit_transform(new_chunk)
            dist = clf_M1["classifier"].transform(cov)
            # If too far from the center of class for MI
            # It is an outlier that is missclassified
            # if dist[0, 1] > tresh:
            #     pred_M1 = 0
        pred_history.append(pred_M1)
        pred_history.pop(0)

        # Smoothing of predictions
        # Calculate EMA on the last 3 predictions (3 windows represent a slide over the last 0.6s)
        ema_pred = ema(pred_history, 1)
        print("Raw tongue proba:", pred_M1, "EMA filtered proba:", ema_pred)

        # Send proba to update interface
        outlet.push_sample([f"{pred_M1}"])

        # Thresholding for an hard decision
        # Just for printing
        if pred_M1 > tresh:
            print(f"Predict Tongue")
            # Forward increment
            # client.send_command(0, 0, 0, 0.5)
            keyboard.press(Key.up)
            time.sleep(0.1)
            keyboard.release(Key.up)
        else:
            print(f"No predictions triggered, {ema_pred}")  #: Predict nothing at {t}")

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
    model_name = params["model_name_oneclass"]
    filtering = params["filtering"]
    idx_electrodes = params["electrodes"]
    if idx_electrodes == "all":
        idx_electrodes = [i for i in range(32)]
    testing(w_length, filtering, idx_electrodes, model_name)
