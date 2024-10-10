# BCI Setup Instructions

## Softwares:

- **actiCAP Control**: For visualizing impedances, close the software after impedances check.
- **BrainVisionRecorder**: Use it **only** to visualize data quality during capping (incompatible with LSL; must be closed afterwards).
- **BrainAmpSeries**: LSL connector. 
- **LabRecorder** (https://github.com/labstreaminglayer/App-LabRecorder): For recording LSL streams together
- **Cybathlon 2024 BCI game**

## Python Setup:

- Install the main repository using Conda and following instructions. 
- **Launch the script in the GUI folder**: Allows launching different parts. Configure the number of blocks and other settings in the **config.json** file at the root.
- Install the UDP communication module for the game: https://github.com/ludovicdmt/udp_cyb

## Data:

All session data is stored and can be used to pre-train models differently.

## Usage:

- Start with a pre-trained model and perform fine-tuning for the specific day. **Fine-tuning is mandatory** to calculate a matrix that adjusts to the session.
- In the *classification* folder, there are two notebooks: *train_modelfor_multiclasses.ipynb* and *train_modelfor_oneclass.ipynb* for pre-training models.
- It's often preferable to start by fine-tuning the forward command, then left/right. 

## Threshold Adjustments:

Refer to: https://github.com/ludovicdmt/MIonline/blob/main/classification/run_testing_2stages_threeclasses.py#L220