#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate EEG data with random signal.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

from PyQt5 import QtWidgets
from pyacq_ext.noisegenerator import NoiseGenerator


def main():
    """Initialize a QApplication, that generate noise similar to EEG signal."""
    app = QtWidgets.QApplication([])

    ng = NoiseGenerator()
    ng.configure(sample_rate=250)
    ng.output.configure(protocol="tcp", transfermode="plaindata")
    ng.initialize()
    ng.start()

    app.exec_()


if __name__ == "__main__":
    main()
