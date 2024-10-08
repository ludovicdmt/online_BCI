#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Send trigger on a regular pace to trigger the slicing of an epoch to be classified.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

from pylsl import StreamInfo, StreamOutlet
import time
import numpy as np

info = StreamInfo(
    name="MarkersEmulator",
    type="Markers",
    channel_count=1,
    nominal_srate=0,
    channel_format="string",
    source_id="presentationPC",
)
outlet = StreamOutlet(info)

labels = ["0"] * 10 + ["1"] * 10
idx = 0

while True:
    # keys are separated by commas, and key-value pairs are separated by equal signs
    lab = labels[idx]
    idx += 1
    if idx == len(labels):
        idx = 0
    outlet.push_sample([lab])
    # Send a trigger every 0.2s to trigger the slicing of an epoch to be classified
    tic = time.perf_counter()
    while time.perf_counter() - tic < 0.22:
        continue
