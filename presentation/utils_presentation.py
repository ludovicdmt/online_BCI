"""Define helper functions for the GUI.

Author: Ludovic Darmet
Mail: ludovic.darmet@gmail.com
"""

import platform
import time
import os
import pygame

if platform.system() == "Linux":
    import subprocess
elif platform.system() == "Windows":
    import win32api
    import winsound
else:
    raise RuntimeError(
        "It looks like you are running this on a system that is not Windows or Linux. \
           Your system is not supported for running the experiment, sorry!"
    )


def pause() -> str:
    """Pause execution until a key is pressed. If escape is pressed, a 'skip' signal is sent."""
    paused = True
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "Skip"
                else:
                    return "Continue"


def checkCalibration() -> None:
    """Pause execution until a key is pressed. If escape is pressed, a 'skip' signal is sent."""
    notFinished = True
    while notFinished:
        if os.path.exists("saved_models\\calibration_done.txt"):
            os.remove("saved_models\\calibration_done.txt")
            notFinished = False
        else:
            time.sleep(1)  # wait for 1 second before checking again


def get_screen_settings(platform):
    """Get screen resolution and refresh rate of the monitor.

    Args:
        platform: str, ['Linux', 'Ubuntu']
              output of platform.system, determines the OS running this script

    Returns:
        height: int
            Monitor height
        width: int
           Monitor width
    """
    if platform not in ["Linux", "Windows"]:
        raise RuntimeError("Unsupported OS! How did you arrive here?")

    if platform == "Linux":
        cmd = ["xrandr"]
        cmd2 = ["grep", "*"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)

        p.stdout.close()

        info, _ = p2.communicate()
        screen_info = info.decode("utf-8").split()[
            :2
        ]  # xrandr gives bytes, for some reason

        width, height = list(map(int, screen_info[0].split("x")))  # Convert to Int

    elif platform == "Windows":

        width, height = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)

    return width, height


def beep():
    """Make a beep sound."""
    if platform.system() == "Windows":
        frequency = 1000  # Set Frequency To 2500 Hertz
        duration = 1000  # Set Duration To 1000 ms == 1 second
        winsound.Beep(frequency, duration)
    else:
        print("\a")
