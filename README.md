# Motor Imagery Asynchronous and Online BCI using Beta Burst Features 🧠💻🎮

## Overview 
🪧
✨ Python scripts for a Brain-Computer Interface (BCI) leveraging motor imagery to control a 2D arrow. The beta burst features methodology have been described by [Papdopoulos](https://iopscience-iop-org.insb.bib.cnrs.fr/article/10.1088/1741-2552/ad19ea).

The GUI is powered by Pygame. Data stream collection relies on [PyACQ](https://github.com/pyacq/pyacq/tree/master) and [LSL](https://github.com/sccn/labstreaminglayer).

Developed by [Ludovic DARMET](http://www.isc.cnrs.fr/index.rvt?language=en&member=ludovic%5Fdarmet) from the [DANC lab](https://www.danclab.com/) and [COPHY](https://www.crnl.fr/en/equipe/cophy), under the supervision of [Jimmy Bonaiuto](http://www.isc.cnrs.fr/index.rvt?member=james%5Fbonaiuto) and [Jérémie Mattout](https://www.crnl.fr/en/user/236).

## Contents
📁
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Example Usage](#example-usage)
- [Help](#help)

## Dependencies

- [Pygame](https://www.pygame.org/news)
- [PyACQ](https://github.com/pyacq/pyacq/tree/master)
- [pylsl](https://github.com/chkothe/pylsl)
- [Sklearn](https://scikit-learn.org/stable/install.html)

## Installation
👩‍💻
Clone the repo:

```bash
git clone https://github.com/ludovicdmt/online_BCI.git
```
Then using `Ananconda Prompt` (terminal from Anaconda):
```bash
cd ${INSTALL_PATH}
conda env create -f MIBCI.yml
conda activate MIBCI
pip install -e .
```
This will install the module in editable mode. That means that any changes you do to the code will be updated so you don't have to re-install every time.

## Example Usage
🗜️
To run the BCI, simply launch the GUI:

```bash
cd ${INSTALL_PATH}/GUI
python GUI_control.py
```

>  During calibration, a PyLSL stream for the markers is created so it can be synchronized with the EEG stream for recording.

To adjust trial time, inter-trial, and other experimental parameters, please refer to the config file.

After data collection, click on Online classification for an asynchronous decoding of the motor imagery.

## Help
🆘
If you encounter any issues while using this code, feel free to post a new issue on the [issues webpage](https://github.com/ludovicdmt/online_BCI/issues). I'll get back to you promptly, as I'm keen on continuously improving it. 🚀

