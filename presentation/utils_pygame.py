"""Define helper functions for Pygame.

Author: Ludovic Darmet
Mail: ludovic.darmet@gmail.com
"""

import pygame


def str_to_surface(txt, font, height):
    """Transform a multi lines text in Pygame surface.

    Args:
    ========
    txt: (str) Multi line text
    font: Font object from Pygame
    heigh: (int) Height of the screen in pixels

    Return:
    ========
    txt_surface: (list) List of Pygame Objects
    start_y: (int) Where to start verticaly to center text
    line_spacing: (int) Number of pixels between lines
    """
    pygame.init()
    txt_lines = txt.split("\n")
    txt_surfaces = [
        font.render(line, True, (255, 255, 255)) for line in txt_lines
    ]  # Render the text
    total_text_height = sum(surface.get_height() for surface in txt_surfaces)
    line_spacing = font.get_linesize()
    start_y = (
        height - total_text_height - line_spacing * (len(txt_lines) - 1)
    ) // 2  # Compute the vertical position to center the text

    return txt_surfaces, start_y, line_spacing


def multilines_surface(surface, start_y, line_spacing, width, screen) -> None:
    """Blit a multi-lines surface.

    Args:
    ========
    surface: (list) List of Pygame Objects
    start_y: (int) Where to start verticaly to center text
    line_spacing: (int) Number of pixels between lines
    width: (int) Number of pixels in the width of the screen
    screen: Pygame object
    """
    pygame.init()
    y = start_y
    for i, text_surface in enumerate(surface):
        text_rect = text_surface.get_rect(centerx=width // 2, centery=y)
        screen.blit(text_surface, text_rect)
        y += text_surface.get_height() + line_spacing


def draw_cross(surface, color, x, y):
    """Draw a cross of defined size in the middle of the screen."""
    # Define the size and position of the cross
    CROSS_WIDTH = 150
    CROSS_HEIGHT = 150
    CROSS_THICKNESS = 20
    # Horizontal line
    pygame.draw.rect(
        surface,
        color,
        (
            x - CROSS_HEIGHT // 2,
            y - CROSS_THICKNESS // 2,
            CROSS_HEIGHT,
            CROSS_THICKNESS,
        ),
    )
    # Vertical line
    pygame.draw.rect(
        surface,
        color,
        (x - CROSS_THICKNESS // 2, y - CROSS_WIDTH // 2, CROSS_THICKNESS, CROSS_WIDTH),
    )
