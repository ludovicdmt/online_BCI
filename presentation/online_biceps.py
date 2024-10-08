#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feedback of the classification (left and right MI) during online testing.
Feeback is a biceps growing with motor intention decoding and then flexing with MI decoding.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import time

import pygame
import json
import os
import sys
import platform
from utils_presentation import pause, get_screen_settings
from utils_pygame import str_to_surface, multilines_surface, draw_cross
import argparse

from pylsl import StreamInlet, StreamInfo, StreamOutlet, resolve_byprop

# LSL stream for the Triggerbox
info = StreamInfo(
    "TriggerBox", "Markers", 1, 0, channel_format="string", source_id="presentationPC"
)
outlet = StreamOutlet(info)

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Load config file
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
size = params["arrow_size"]
trial_n = params["trial_number"]
block_number = params["block_number"]
epoch_duration = params["epoch_duration"]
iti_duration = params["iti_duration"]

# Create a window
# Window parameters
system = platform.system()
width, height = get_screen_settings(system)

# Compute framerate
clock = pygame.time.Clock()
fps = 60
for i in range(20):
    print("{}: tick={}, fps={}".format(i + 1, clock.tick(fps), clock.get_fps()))
refresh_rate = clock.get_fps()

# Time conversion to frames
epoch_frames = int(epoch_duration * refresh_rate)
iti_frames = int(iti_duration * refresh_rate)

# Marker stream
streams = resolve_byprop("name", "PredictionMarkers", timeout=15)
if len(streams) == 0:
    raise (RuntimeError("Can't find marker stream..."))
clf_inlet = StreamInlet(streams[0], max_chunklen=1, processing_flags=1)

# Initialize Pygame
pygame.init()

# Cue
# Set the font and text content
font = pygame.font.SysFont(None, 48)  # None means default font, 36 is the font size

calib_text_start = "L'expérience va commencer. \n Eviter les mouvements parasites et de la machoire pendant les blocs. \n \
Imagination libre d'un mouvement de la main gauche ou droite. \n \n \
Un retour visuel va apparaitre sous la forme d'une flexion du bras droit ou gauche. \n \n \
"

calib_text_surfaces, start_y_calib, line_spacing = str_to_surface(
    calib_text_start, font, height
)

# Set up display
screen = pygame.display.set_mode((width, height), display=1)
screen.fill(BLACK)
pygame.display.set_caption("Cybathlon Motor Imagery online")


# Load images (Ensure you have these images prepared and available in the correct path)
left_arm_image = pygame.image.load(
    "C:\\Users\\ludov\\Documents\\repos\\MIonline\\presentation\\img\\biceps_left\\frame_1.png"
)
right_arm_image = pygame.image.load(
    "C:\\Users\\ludov\\Documents\\repos\\MIonline\\presentation\\img\\biceps_right\\frame_1.png"
)
left_bicep_images = [
    pygame.image.load(
        f"C:\\Users\\ludov\\Documents\\repos\\MIonline\\presentation\\img\\biceps_left\\frame_{i}.png"
    )
    for i in range(1, 39)
]
right_bicep_images = [
    pygame.image.load(
        f"C:\\Users\\ludov\\Documents\\repos\\MIonline\\presentation\\img\\biceps_right\\frame_{i}.png"
    )
    for i in range(1, 39)
]
relax = pygame.image.load(
    "C:\\Users\\ludov\\Documents\\repos\\MIonline\\presentation\\img\\relax.png"
)


# Function to determine the image index based on probability
def get_image_index(prob):
    return min(int(prob * len(left_bicep_images)), 38)


# Function to scale an image based on probability
def scale_image(image, prob):
    scaled_im = image.copy()
    scale_factor = prob + 0.5  # between 0.5 and 1
    width, height = scaled_im.get_size()
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return pygame.transform.scale(scaled_im, new_size), scale_factor


# ====================================================================================================

# Experiment structure
## Calibration
time.sleep(5)
# Start presentation
multilines_surface(calib_text_surfaces, start_y_calib, line_spacing, width, screen)
pygame.display.flip()
pause()
screen.fill(BLACK)

# Fixation cross thickness
thickness = 20

prev_pred = -1
patience = 0
same_pred = 0

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    # Check for keypresses and send triggers
    if keys[pygame.K_LEFT]:
        outlet.push_sample([str(0)])
        print("Trigger left send")
    if keys[pygame.K_RIGHT]:
        outlet.push_sample([str(1)])
        print("Trigger right send")
    if keys[pygame.K_DOWN]:
        outlet.push_sample([str(-1)])
        print("Trigger end send")

    # Fixation cross
    draw_cross(screen, WHITE, width // 2, height // 2)

    # Retrieve the prediction and update display accordingly
    y_pred, _ = clf_inlet.pull_sample(timeout=0.0)
    while y_pred == None:
        y_pred, _ = clf_inlet.pull_sample(timeout=0.0)
    y_pred_M1 = float(y_pred[0].split(",")[0])
    y_pred_M2 = float(y_pred[0].split(",")[1])

    print(
        f"Prediction M1: {y_pred_M1}, Prediction M2: {y_pred_M2}, Previous prediction: {prev_pred}, Patience: {patience}, Patience: {same_pred}."
    )

    # If only motor intention
    if (y_pred_M2 == -1) and (y_pred_M1 > 0.47):
        # Proba of Lucas are moving between 0.47 and 0.51 for MI detection
        scaled_proba = (y_pred_M1 - 0.47) / (0.51 - 0.47)
        # scaled_proba = y_pred_M1
        if scaled_proba > 1:
            scaled_proba = 0.7
        elif scaled_proba < 0:
            scaled_proba = 0.3
        im_left_scaled, scale_factor = scale_image(left_arm_image, scaled_proba)
        im_right_scaled, _ = scale_image(right_arm_image, scaled_proba)
        screen.blit(
            im_left_scaled,
            (
                (width - size * scale_factor) // 2 - 3.5 * size * scale_factor,
                100 + (1 - scale_factor) * size,
            ),
        )
        screen.blit(
            im_right_scaled,
            (
                (width - size * scale_factor) // 2 + 1.5 * size * scale_factor,
                100 + (1 - scale_factor) * size,
            ),
        )

    elif y_pred_M2 >= 0:
        if prev_pred < 0:
            prev_pred = round(y_pred_M2)
        if round(y_pred_M2) == prev_pred:
            same_pred += 1
            # if we have too much of the same pred ask Lucas to relax for 3s
            # if same_pred > 9:
            #     screen.blit(relax, (width // 2, height // 2))
            #     tac = time.perf_counter()
            #     if tac - tic > 3:
            #         same_pred = 0
            #         del tac, tic
            # else:
            # if same_pred == 9:
            #     tic = time.perf_counter()
            # Right prediction
            if y_pred_M2 > 0.50:
                x_pos = (width - size) // 2 + 1.5 * size
                # y_pred_M2 is between 0.5 and 1.0 and we want to have it between 0 and 1
                idx_im = min(int((y_pred_M2 * 2 - 1) * len(right_bicep_images)), 37)
                pics = right_bicep_images
            # Left prediction
            else:
                x_pos = (width - size) // 2 - 3.5 * size
                # y_pred_M2 is between 0 and 0.5 and we want to have it between 0 and 1
                idx_im = min(int((y_pred_M2 * 2) * len(left_bicep_images)), 37)
                pics = left_bicep_images
            screen.blit(pics[idx_im], (x_pos, 100))

        else:
            same_pred = 0
            # Don't change output
            patience += 1
            screen.blit(pics[idx_im], (x_pos, 100))

            # if we had 4 different preds in a row
            # Clear outputs and remove the image
            if patience == 4:
                patience = 0
                prev_pred = round(y_pred_M2)

    # update display
    pygame.display.flip()
    screen.fill(BLACK)
    clock.tick(60)

sys.exit()
