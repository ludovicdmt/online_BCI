#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Biceps presentation (left and right) to collect calibration data for a Motor Imagery based BCI.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import os
import numpy as np
import json
import platform
import argparse
import pygame
from pylsl import StreamInfo, StreamOutlet
from utils_presentation import pause, get_screen_settings, checkCalibration
from utils_pygame import str_to_surface, multilines_surface, draw_cross
import random
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

path = os.getcwd()  # Load config file

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

system = platform.system()
width, height = get_screen_settings(system)  # Create a window


clock = pygame.time.Clock()
fps = 60
for i in range(20):
    print("{}: tick={}, fps={}".format(i + 1, clock.tick(fps), clock.get_fps()))
refresh_rate = clock.get_fps()  # Compute framerate

epoch_frames = int(epoch_duration * refresh_rate)  # Time conversion to frames
iti_frames = int(iti_duration * refresh_rate)

info = StreamInfo(
    name="MotorImageryMarkers",
    type="Markers",
    channel_count=1,
    nominal_srate=0,
    channel_format="string",
    source_id="presentationPC",
)
info.desc().append_child_value("n_train", f"{trial_n}")
info.desc().append_child_value("epoch_duration", f"{epoch_duration}")
info.desc().append_child_value("n_block", f"{block_number}")
outlet = StreamOutlet(info)

pygame.init()

font = pygame.font.SysFont(None, 48)  # None means default font, 36 is the font size

calib_text_start = "L'expérience va commencer. \n Eviter les mouvements parasites et de la machoire pendant les blocs. \n \n  \
Suivre la flexion du bras droit ou gauche \n pour imaginer un mouvement de la main gauche ou droite. \n \n\
Merci de maintenir votre imagination jusqu'à la fin du bloc.  \n \
"
calib_text_surfaces, start_y_calib, line_spacing = str_to_surface(
    calib_text_start, font, height
)

try:  # if multiple screens
    screen = pygame.display.set_mode((width, height), display=1)
except:  # if single screen
    screen = pygame.display.set_mode((width, height), display=0)
black = (0, 0, 0)  # Set the background color
screen.fill(black)
pygame.display.set_caption("Cybathlon Motor Imagery")

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


def get_image_index(prob):
    """Determine the image index based on probability"""
    return min(int(prob * len(left_bicep_images)), 38)


def scale_image(image, prob):
    """Scale an image based on probability"""
    scale_factor = prob * 1.5 + 0.5  # between 0.5 and 2
    width, height = image.get_size()
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return pygame.transform.scale(image, new_size)


symbols = ["Left", "Right"]
nb_calib = trial_n * block_number

# Balance the two classes
trial_list = nb_calib // 2 * [symbols[0]] + nb_calib // 2 * [symbols[1]]
while len(trial_list) < nb_calib:
    trial_list += [
        symbols[np.random.randint(2)]
    ]  # Add symbols randomly for odd block number setups
random.shuffle(trial_list)

# Ensure max three consecutive identical trials
i = 0
while i < len(trial_list) - 3:
    if trial_list[i] == trial_list[i + 1] == trial_list[i + 2] == trial_list[i + 3]:
        for j in range(i + 4, len(trial_list)):
            if trial_list[j] != trial_list[i]:
                trial_list[i + 3], trial_list[j] = trial_list[j], trial_list[i + 3]
                break
    i += 1


trial_list = [
    trial_list[i : i + trial_n] for i in range(0, len(trial_list), trial_n)
]  # Reshape trial_list into a list of lists

time.sleep(3)  # Wait to have enough data in the EEG buffer


multilines_surface(calib_text_surfaces, start_y_calib, line_spacing, width, screen)
pygame.display.flip()  # Start presentation
pause()
screen.fill(BLACK)

thickness = 20  # Fixation cross thickness

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for idx_block, sequence in enumerate(trial_list):
        txt = f"Bloc {idx_block+1} sur {len(trial_list)}. \n Espace pour continuer et échap pour quitter."
        txt_surfaces, start_y, _ = str_to_surface(txt, font, height)
        multilines_surface(txt_surfaces, start_y, line_spacing, width, screen)
        pygame.display.flip()
        out = pause()
        screen.fill(BLACK)
        if out == "Skip":
            pygame.quit()

        # For each cue in our sequence
        for idx_target, target in enumerate(sequence):
            if target == "Left":
                pics = left_bicep_images
                x_pos = (width - size) // 2 - 3.5 * size
                label = 0
            elif target == "Right":
                pics = right_bicep_images
                x_pos = (width - size) // 2 + 1.5 * size
                label = 1
            # ITI presentation
            if (
                idx_target == 0
            ):  # If it is the first trial of the block we halve ITI to just have baseline before
                for n in range(iti_frames // 2):
                    draw_cross(screen, WHITE, width // 2, height // 2)
                    pygame.display.flip()
                    screen.fill(BLACK)
                    clock.tick(60)

            else:  # Else we have  ITI + baseline
                for n in range(iti_frames):
                    draw_cross(screen, WHITE, width // 2, height // 2)
                    pygame.display.flip()
                    screen.fill(BLACK)
                    clock.tick(60)

            frames = 0
            t0 = time.perf_counter()  # Retrieve time at trial onset

            outlet.push_sample([str(label)])  # Send marker
            for frame in range(epoch_frames):  # Cue presentation
                draw_cross(screen, WHITE, width // 2, height // 2)
                idx_pic = min(int(frame * len(pics) / epoch_frames), len(pics))
                screen.blit(pics[idx_pic], (x_pos, 100))
                pygame.display.flip()
                screen.fill(BLACK)
                clock.tick(60)

                frames += 1

            t1 = time.perf_counter()
            elapsed = (
                t1 - t0
            )  # At the end of the trial, calculate real duration and amount of frames
            print(f"Time elapsed: {elapsed}")
            print("")

            # If it is last trial of the block we need to finish with ITI only and not ITI + baseline
            # for next trial
            if idx_target == len(sequence) - 1:
                for n in range(iti_frames):
                    draw_cross(screen, WHITE, width // 2, height // 2)
                    pygame.display.flip()
                    screen.fill(BLACK)
                    clock.tick(60)

    txt = f"Calibration finie. Merci d'attendre que le classifieur s'entraine."
    txt_surface = font.render(txt, True, (255, 255, 255))
    text_rect = txt_surface.get_rect(centerx=width // 2, centery=height // 2)
    screen.blit(txt_surface, text_rect)
    pygame.display.flip()
    checkCalibration()
    pygame.quit()
