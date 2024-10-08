#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run online classification on sliding windows with overlap.
The classificaiton is performed in three stages: first detection of left and right MI vs rest+rebound,
then consequently either detection of left vs right or otherwise detection of tongue/jaw MI vs rest+rebound.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

from matplotlib.pyplot import hist
import numpy as np
import scipy.signal
import json
import os
import argparse
import pickle
import time

# from pynput.keyboard import Key, Controller
from udp_cyb import UDPClient

from PyQt5 import QtWidgets

from pyacq_ext.sosfilter import SosFilterTimestamps
from pyacq_ext.epochermultilabel import EpocherMultiLabel
from pyacq_ext.markers_lsl import LslMarkers
from pyacq_ext.eeg_lsl import LslEEG
from pyacq_ext.noisegenerator import NoiseGenerator

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
    model_name_oneclass: str,
) -> None:
    """Run online classification on sliding windows with overlap.

    Args:
        w_length (float): Length of the sliding window.
        filtering (list): List of filtering parameters.
        idx_electrodes (list): List of indices of the electrodes to keep.
        model_name (str): Name of the model to use for left/right motor intention detection.
        model_name2 (str): Name of the model to use for left/right MI detection.
        model_name_oneclass (str): Name of the model to use for forward/backward detection.
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
    # ng.configure(chunksize = 30, sample_rate=250, data_path = 'C:\\Users\\ludov\\Documents\\pre_tests\\S1\\0916\\calib_arms.xdf')
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
    coefficients = scipy.signal.iirfilter(
        30,
        [filtering[0] / sample_rate * 2, filtering[1] / sample_rate * 2],
        btype="bandpass",
        ftype="butter",
        output="sos",
    )  # Compute coefficients for a cascaded 2nd-order sections (sos) bandpass Butterworth filter
    filt = SosFilterTimestamps()
    filt.configure(coefficients=coefficients)
    filt.input.connect(
        ba.outputs["signals"]
    )  # This node is connected to the output of the device node
    filt.output.configure(
        protocol="tcp",
        interface="127.0.0.1",
        transfermode="plaindata",
    )
    filt.initialize()
    time.sleep(2)

    # Epocher Node
    params = {"tmin": 0, "tmax": w_length}  # Parameters to slice epochs

    epocher = EpocherMultiLabel(params)
    epocher.configure()
    epocher.inputs["signals"].connect(filt.output)
    epocher.inputs["triggers"].connect(trig.outputs["triggers"])
    epocher.initialize()

    # Start UPDClient
    client = UDPClient(ip="localhost", port=59075)
    client.start()

    # Load models for left vs right
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
            tresh_MDM = pickle.load(f)

    # Load data to compute amplitude treshold for
    file = "saved_models\\rebound_ft_data.pickle"
    with open(file, "rb") as f:
        data_dict = pickle.load(f)
    flattened_electrodes = np.abs(data_dict["calib_data"]).reshape(-1)
    tresh_amp = np.percentile(flattened_electrodes, 98)
    print(f" Tresh amp: {tresh_amp}")

    # Load model for tongue
    path_to_model = f"saved_models/{model_name_oneclass}_forward.pickle"
    clf_tongue = load_model(path_to_model)
    # file = f"saved_models\\tresh_M1_tongue.pickle"
    # with open(file, "rb") as f:
    #     tresh_tongue_MDM = pickle.load(f)
    print("Models loaded")

    # tresh_tongue = 0.512
    # tresh_M1 = 0.505
    # tresh_M2 = 0.52

    pred_history = [0.5]  # Initialize with -1 to avoid triggering at the start

    # Load ASR model to pre-process filtered and epoched data
    # file_asr = "saved_models\\asr.pickle"
    # with open(file_asr, "rb") as f:
    #     asr = pickle.load(f)
    # pred_window = []
    # pred_label = []

    hist_M1 = []
    hist_tongue = []
    hist_M2 = []

    def on_new_chunk(label, new_chunk, eeg_timestamp) -> None:
        """Trigger when a new chunk from the epocher is available.

        Args:
            label (str): Label code. Unused during testing, only -1.
            new_chunk (array): New chunk of data.
            eeg_timestamp (float): EEG timestamp.
        """
        nonlocal tresh_MDM, client  # ,  tresh_tongue_MDM, pred_window, pred_label#, asr #tresh_M1, tresh_M2, tresh_tongue,

        # Don't predict if the game is not running
        if client.control == False:
            return
        if client.device == "arm":
            forward = 0.6
            tresh_tongue = 0.515
            tresh_M1 = 0.506
            tresh_M2 = 0.3
        else:
            forward = 0.5
            tresh_tongue = 0.4842
            tresh_M1 = 0.502
            tresh_M2 = 0.42
        # if len(pred_window) % 10 == 0:
        #     file = "pred_window.npy"
        #     with open(file, "wb") as f:
        #         pickle.dump({"pred_window":pred_window, "pred_label":pred_label}, f)

        # Pre-processing
        new_chunk = new_chunk.T * 1e-6  # Convert from uV to V
        # new_chunk = new_chunk[idx_electrodes, :] # Electrode selection
        ref = np.mean(
            new_chunk, axis=0, keepdims=True
        )  # Common average re-referencing (CAR)
        new_chunk = new_chunk - ref
        # new_chunk = asr.transform(new_chunk) # asr preprocess
        new_chunk = new_chunk[np.newaxis, :, :]
        new_chunk -= np.mean(new_chunk, axis=-1, keepdims=True)  # Baseline removal
        new_chunk = np.einsum("ij,tjk->tik", R_inv, new_chunk)  # Euclidean Alignment

        # Artifact rejection
        if reject_artifacts(
            new_chunk / 2, amplitude_threshold=tresh_amp, zscore_threshold=4.0
        ):
            return

        # Check for NaN values
        if np.isnan(new_chunk).any():
            return

        # Classify the data
        pred_M1 = clf_M1.predict_proba(new_chunk)[0][1]  # First stage (MI vs rebound)
        pred = clf_M2.predict_proba(new_chunk)[0]  # Second stage (left vs right)
        pred_tongue = clf_tongue.predict_proba(new_chunk)[0][1]  # Tongue vs rest
        hist_M1.append(pred_M1)
        hist_tongue.append(pred_tongue)
        hist_M2.append(pred[1])
        if len(hist_M1) % 100 == 0:
            np.save("hist_all.npy", (hist_M1, hist_tongue, hist_M2), allow_pickle=True)
        print(f"Proba tongue: {pred_tongue}, Proba M1: {pred_M1}, Proba M2 {pred}")

        # Resting class filtering for left/right MI detection
        # if (model_name == "MDM") and (pred_M1) > 0.5:
        # cov = clf_M1["feat"].fit_transform(new_chunk)
        # dist = clf_M1["classifier"].transform(cov)
        # If too far from the center of class for MI
        # It is an outlier that is missclassified
        # if dist[0, 1] > tresh_MDM:
        #     pred_M1 = 0

        # Handle tongue MI
        if pred_tongue > tresh_tongue:
            if pred_tongue > 0.9:  # Force forward if certain enough
                outlet.push_sample([f"{pred_M1},-1, {pred_tongue}"])
                # Forward increment
                client.send_command(0, 0, 0, forward)
                print(f"Predict forward")
                return
            elif (pred_M1 > tresh_M1 + 0.02) and (pred_tongue < tresh_tongue + 0.05):  #
                pred_class = np.argmax(pred)
                pred_proba = np.max(pred)
                pred_history.append(pred_proba if pred_class == 1 else 1 - pred_proba)
            else:  # Forward if Motor Intention for left/right is not high enough
                outlet.push_sample([f"{pred_M1},-1, {pred_tongue}"])
                # Forward increment
                client.send_command(0, 0, 0, forward)
                # keyboard.press(Key.up)
                # time.sleep(0.1)
                # keyboard.release(Key.up)
                print(f"Predict forward")
                # pred_window.append(np.squeeze(new_chunk))
                # pred_label.append(2)
                return

        # Handle left vs right MI
        elif pred_M1 > tresh_M1:
            pred_class = np.argmax(pred)
            pred_proba = np.max(pred)
            pred_history.append(pred_proba if pred_class == 1 else 1 - pred_proba)

        # No clear MI detected
        else:
            outlet.push_sample([f"{pred_M1},-1,0"])
            # pred_window.append(np.squeeze(new_chunk))
            # pred_label.append(-1)
            return

        # Update prediction history and calculate EMA
        pred_history.pop(0)
        ema_pred = ema(pred_history, 1)

        # Make a prediction based on the EMA
        print(f"Ema pred: {ema_pred}, proba: {pred_proba}")
        if ema_pred > tresh_M2:
            print(f"Predict 1")  # at {t}")
            # pred_window.append(np.squeeze(new_chunk))
            # pred_label.append(1)
            # Rotate from an increment
            client.send_command(0, 0, +0.2, 0)  # Rotate right
            # keyboard.press(Key.right)
            # time.sleep(0.1)
            # keyboard.release(Key.right)
            outlet.push_sample([f"{pred_M1},{ema_pred},0"])
        elif ema_pred < tresh_M2:
            print(f"Predict 0 ")  # at {t}")
            # pred_window.append(np.squeeze(new_chunk))
            # pred_label.append(0)
            # Rotate from an increment
            client.send_command(0, 0, -0.2, 0)  # Rotate left
            # keyboard.press(Key.left)
            # time.sleep(0.1)
            # keyboard.release(Key.left)
            outlet.push_sample([f"{pred_M1},{ema_pred},0"])
        else:
            outlet.push_sample([f"{pred_M1},-1,0"])

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
    model_name_oneclass = params["model_name_oneclass"]
    model_name = params["model_name1"]
    model_name2 = params["model_name2"]
    filtering = params["filtering"]
    idx_electrodes = params["electrodes"]
    if idx_electrodes == "all":
        idx_electrodes = [i for i in range(32)]
    testing(
        w_length,
        filtering,
        idx_electrodes,
        model_name,
        model_name2,
        model_name_oneclass,
    )
