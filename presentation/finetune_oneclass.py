#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Animated GIF of noding to collect data for FT.
Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import os
import json
import platform
import argparse
import pygame
from pylsl import StreamInfo, StreamOutlet
from utils_presentation import pause, get_screen_settings, checkCalibration
from utils_pygame import str_to_surface, multilines_surface, draw_cross
import time

# Set up colors
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

parser.add_argument(
    "-c",
    "--command",
    metavar="command",
    type=str,
    default="forward",
    help="Run the forward or backward GUI. Default: %(default)s.",
)

args = parser.parse_args()
config_path = os.path.join(path, args.file)

with open(config_path, "r") as config_file:
    params = json.load(config_file)

# Experimental params
size = params["arrow_size"]
trial_n = params["trial_number"]
block_number = params["ft_number"]
epoch_duration = params["epoch_duration"]
iti_duration = params["iti_duration"]

system = platform.system()  # Window parameters
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

cmd = str(args.command).strip()
if cmd == "forward":
    mov = "pousser les 2 jambes."
    img_desc = "leg"
    size = 120
elif cmd == "backward":
    mov = "le pouce"
    img_desc = "thumb"
    size = 150

calib_text_start = f"L'expérience va commencer. \n Eviter les mouvements parasites et de la machoire pendant les blocs. \n \n  \
Suivre l'indication à l'écran pour imaginer bouger {mov}. \n \
"
calib_text_surfaces, start_y_calib, line_spacing = str_to_surface(
    calib_text_start, font, height
)

try:  # If we have multpile screens
    screen = pygame.display.set_mode((width, height), display=1)
except:  # Single screen
    screen = pygame.display.set_mode((width, height), display=0)

black = (0, 0, 0)  # Set the background color (white)
screen.fill(black)  # Fill the screen with white
pygame.display.set_caption("Cybathlon Motor Imagery")

# Load images (Ensure you have these images prepared and available in the correct path)
anim = pygame.image.load(
    f"C:\\Users\\ludov\\Documents\\repos\\MIonline\\presentation\\img\\{img_desc}\\frame_0.png"
)
x, y = anim.get_size()
animes = [
    pygame.image.load(
        f"C:\\Users\\ludov\\Documents\\repos\\MIonline\\presentation\\img\\{img_desc}\\frame_{i}.png"
    )
    for i in range(0, 18)
]

symbols = ["Tongue"]
nb_calib = trial_n * block_number

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

    for idx_block in range(block_number):
        txt = f"Bloc {idx_block+1} sur {block_number}. \n Espace pour continuer et échap pour quitter."
        txt_surfaces, start_y, _ = str_to_surface(txt, font, height)
        multilines_surface(txt_surfaces, start_y, line_spacing, width, screen)
        pygame.display.flip()
        out = pause()
        screen.fill(BLACK)
        if out == "Skip":
            pygame.quit()

        # For each cue in our sequence
        for idx_target in range(trial_n):
            pics = animes
            y_pos = (height - size) // 2 - y // 2
            label = 1

            # ITI presentation
            if (
                idx_target == 0
            ):  # If it is the first trial of the block we halve ITI to just have baseline before
                for n in range(iti_frames // 2):
                    draw_cross(screen, WHITE, width // 2, height // 2 + 100)
                    pygame.display.flip()
                    screen.fill(BLACK)
                    clock.tick(60)

            else:  # Else we have  ITI + baseline
                for n in range(iti_frames):
                    draw_cross(screen, WHITE, width // 2, height // 2 + 100)
                    pygame.display.flip()
                    screen.fill(BLACK)
                    clock.tick(60)

            frames = 0
            t0 = time.perf_counter()  # Retrieve time at trial onset

            # Cue presentation
            outlet.push_sample([str(label)])  # Send marker
            for frame in range(epoch_frames):
                draw_cross(screen, WHITE, width // 2, height // 2 + 100)
                idx_pic = min(int(frame * len(pics) / epoch_frames), len(pics))
                screen.blit(pics[idx_pic], ((width) // 2 - x // 2, y_pos))
                pygame.display.flip()
                screen.fill(BLACK)
                clock.tick(60)
                frames += 1

            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"Time elapsed: {elapsed}")
            print("")

            # If it is last trial of the block we need to finish with ITI only and not ITI + baseline
            # for next trial
            if idx_target == trial_n - 1:
                for n in range(iti_frames):
                    draw_cross(screen, WHITE, width // 2, height // 2 + 100)
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
