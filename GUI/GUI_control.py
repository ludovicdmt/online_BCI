#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launcher from the different phases and script using a GUI.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import os
import os.path as op

# Import `QApplication` and all the required widgets
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QPushButton,
    QRadioButton,
    QGridLayout,
    QVBoxLayout,
)
from PyQt5.QtCore import Qt
import signal
import sys
import subprocess
import platform
import threading

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from presentation.utils_presentation import get_screen_settings


class MI_gui(QWidget):
    """Launch the different phases and script using a GUI."""

    def __init__(self) -> None:
        """Launcher from the different phases and script using a GUI."""
        super().__init__()
        self.processes = []
        system = platform.system()
        width, height = get_screen_settings(system)

        self.setWindowTitle("Burst Motor Imagery")
        self.setGeometry(100, 100, 600, 500)
        self.move(width // 2 - 300, height // 2 - 250)  # center the window
        helloMsg = QLabel("<h1>Welcome to this Motor Imagery BCI!</h1>", parent=self)
        helloMsg.setAlignment(Qt.AlignCenter)

        # Create buttons
        self.button_calibration_two_classes = QPushButton(
            "Calibration two classes", self
        )
        self.button_calibration_two_classes.setMinimumHeight(50)
        self.button_calibration_two_classes.clicked.connect(
            self.button_calibration_clicked
        )

        self.button_calibration_forward = QPushButton("Calibration forward", self)
        self.button_calibration_forward.setMinimumHeight(50)
        self.button_calibration_forward.clicked.connect(
            self.button_calibration_forward_clicked
        )
        self.button_calibration_backward = QPushButton("Calibration backward", self)
        self.button_calibration_backward.setMinimumHeight(50)
        self.button_calibration_backward.clicked.connect(
            self.button_calibration_backward_clicked
        )

        self.button_ft_two_classes = QPushButton("Fine Tuning two classes", self)
        self.button_ft_two_classes.setMinimumHeight(50)
        self.button_ft_two_classes.clicked.connect(self.button_ft_clicked)

        self.button_ft_forward = QPushButton("Fine Tuning forward", self)
        self.button_ft_forward.setMinimumHeight(50)
        self.button_ft_forward.clicked.connect(self.button_ft_clicked_forward)

        self.button_ft_backward = QPushButton("Fine Tuning backward", self)
        self.button_ft_backward.setMinimumHeight(50)
        self.button_ft_backward.clicked.connect(self.button_ft_clicked_backward)

        self.button_testing_two_classes = QPushButton(
            "Online Testing two classes", self
        )
        self.button_testing_two_classes.setMinimumHeight(50)
        self.button_testing_two_classes.clicked.connect(self.button_testing_clicked)

        self.button_testing_cyb = QPushButton("Online Testing Cybathlon Game", self)
        self.button_testing_cyb.setMinimumHeight(50)
        self.button_testing_cyb.clicked.connect(self.button_testing_clicked_cyb)

        # Create a grid layout and add the buttons to it
        grid = QGridLayout()
        grid.addWidget(self.button_calibration_two_classes, 1, 0)
        grid.addWidget(self.button_calibration_forward, 1, 1)
        grid.addWidget(self.button_calibration_backward, 1, 2)
        grid.addWidget(self.button_ft_two_classes, 2, 0)
        grid.addWidget(self.button_ft_forward, 2, 1)
        grid.addWidget(self.button_ft_backward, 2, 2)
        grid.addWidget(self.button_testing_two_classes, 3, 0)
        grid.addWidget(self.button_testing_cyb, 3, 1)
        grid.setVerticalSpacing(20)  # Increase vertical spacing between buttons

        # Create a main vertical layout and add the QLabel and grid layout to it
        main_layout = QVBoxLayout()
        main_layout.addWidget(helloMsg)
        main_layout.addLayout(grid)

        # Set the layout for the main window
        self.setLayout(main_layout)

    def run_process(self, processes, backward=False):
        """Run the process in a new terminal.

        Args:
          processes (list): List of the process to run.
          backward (boolean): For the one class command.
        """
        for p in processes:
            if platform.system() == "Linux":
                try:
                    self.processes.append(
                        subprocess.Popen(
                            ["gnome-terminal -- python " + f"{p}"], shell=True
                        )
                    )
                except Exception as e:
                    self.processes.append(
                        subprocess.Popen(["gnome-terminal -- python3 " + p], shell=True)
                    )

            else:  # windows
                if backward:
                    cmd = [sys.executable, p, "-c backward"]
                else:
                    cmd = [sys.executable, p]
                print("Command sent to subprocess: ", cmd[1:])
                try:
                    self.processes.append(subprocess.Popen(cmd))
                except Exception as e:
                    print(e)
        # Start a new thread to monitor the process
        monitor_thread = threading.Thread(
            target=self.monitor_process, args=(self.processes,)
        )
        monitor_thread.start()

    def monitor_process(self, process):
        """Monitor a process and print a message if it stops.

        Args:
          process (subprocess.Popen): The process to monitor.
        """
        while True:
            for p in self.processes:
                if p.poll() is not None:
                    print(f"Process {p.args[1]} has stopped.")
                    self.processes.remove(p)
                    self.__del__()

    def button_calibration_clicked(self):
        """Run the calibration scripts."""
        if platform.system() == "Linux":
            processes = [
                op.join("./presentation", "offline_biceps.py"),
                op.join("./classification", "run_calibration.py"),
            ]
        else:  # windows
            processes = [
                op.join("presentation", "offline_biceps.py"),
                op.join("classification", "run_calibration.py"),
            ]

        print("Calibration Button clicked")
        self.run_process(processes)

    def button_calibration_forward_clicked(self):
        """Run the calibration scripts for one class."""
        if platform.system() == "Linux":
            processes = [
                op.join("./presentation", "offline_oneclass.py"),
                op.join("./classification", "run_calibration_oneclass.py"),
            ]
        else:  # windows
            processes = [
                op.join("presentation", "offline_oneclass.py"),
                op.join("classification", "run_calibration_oneclass.py"),
            ]

        print("Calibration Button clicked")
        self.run_process(processes)

    def button_calibration_backward_clicked(self):
        """Run the calibration scripts for one class."""
        if platform.system() == "Linux":
            processes = [
                op.join("./presentation", "offline_oneclass.py"),
                op.join("./classification", "run_calibration_oneclass.py"),
            ]
        else:  # windows
            processes = [
                op.join("presentation", "offline_oneclass.py"),
                op.join("classification", "run_calibration_oneclass.py"),
            ]

        print("Calibration Button clicked")
        self.run_process(processes, backward=True)

    def button_ft_clicked(self):
        """Run the calibration scripts."""
        if platform.system() == "Linux":
            processes = [
                op.join("./presentation", "finetune_biceps.py"),
                op.join("./classification", "run_finetuning.py"),
            ]
        else:  # windows
            processes = [
                op.join("presentation", "finetune_biceps.py"),
                op.join("classification", "run_finetuning.py"),
            ]

        print("Calibration Button clicked")
        self.run_process(processes)

    def button_ft_clicked_forward(self):
        """Run the finetuning scripts."""
        if platform.system() == "Linux":
            processes = [
                op.join("./presentation", "finetune_oneclass.py"),
                op.join("./classification", "run_finetuning_oneclass.py"),
            ]
        else:  # windows
            processes = [
                op.join("presentation", "finetune_oneclass.py"),
                op.join("classification", "run_finetuning_oneclass.py"),
            ]

        print("Calibration Button clicked")
        self.run_process(processes)

    def button_ft_clicked_backward(self):
        """Run the finetuning scripts."""
        if platform.system() == "Linux":
            processes = [
                op.join("./presentation", "finetune_oneclass.py"),
                op.join("./classification", "run_finetuning_oneclass.py"),
            ]
        else:  # windows
            processes = [
                op.join("presentation", "finetune_oneclass.py"),
                op.join("classification", "run_finetuning_oneclass.py"),
            ]

        print("Calibration Button clicked")
        self.run_process(processes, backward=True)

    def button_testing_clicked(self):
        """Run the online testing scripts."""
        if platform.system() == "Linux":
            processes = [
                op.join("./classification", "triggerEmulator.py"),
                # op.join("./classification", "run_testing_2stages_threeclasses.py"),
                op.join("./classification", "run_testing_2stages.py"),
                op.join("./presentation", "online_biceps.py"),
            ]
        else:  # windows
            processes = [
                op.join("presentation", "triggerEmulator.py"),
                # op.join("classification", "run_testing_2stages_threeclasses.py"),
                op.join("./classification", "run_testing_2stages.py"),
                op.join("presentation", "online_biceps.py"),
            ]

        print("Testing Button clicked")
        self.run_process(processes)

    def button_testing_clicked_cyb(self):
        """Run the online testing scripts."""
        if platform.system() == "Linux":
            processes = [
                op.join("presentation", "triggerEmulator.py"),
                op.join("classification", "run_testing_2stages_threeclasses.py"),
            ]
        else:  # windows
            processes = [
                op.join("presentation", "triggerEmulator.py"),
                op.join("classification", "run_testing_2stages_threeclasses.py"),
            ]

        print("Testing Button clicked")
        self.run_process(processes)

    def __del__(self):
        """Kill all the subprocess created."""
        for p in self.processes:
            os.kill(p.pid, signal.SIGTERM)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MI_gui()
    ex.show()
    sys.exit(app.exec_())
