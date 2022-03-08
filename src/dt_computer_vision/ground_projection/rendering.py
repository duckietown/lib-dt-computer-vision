#!/usr/bin/env python3

from typing import Optional

import cv2
import numpy as np

grid_size = 10
# dimensions of the image are 1m x 1m so, 1px = (1 / size_x) meters
size_x, size_y = 1600, 1600
grid_size_x = int(size_x / grid_size)
grid_size_y = int(size_y / grid_size)
origin_x, origin_y = int(size_x / 2), size_y - grid_size_y
grid_thickness = 5
segment_thickness = 12
font_size = 3.0
font_thickness = 3


def draw_grid_image():
    """
    Generates a grid image with distances from the robot's origin.

    Returns:
        :obj:`numpy array`: an OpenCV image

    """
    # initialize gray image
    grid_image = np.ones((size_y, size_x, 3), np.uint8) * 128

    # draw vertical lines of the grid
    for vline in np.arange(grid_size_x, size_x - grid_size_x + 1, grid_size_x):
        cv2.line(
            grid_image,
            pt1=(vline, grid_size_y),
            pt2=(vline, origin_y),
            color=(255, 255, 0),
            thickness=grid_thickness
        )

    h_text_centering_offset = int(size_x / 20)
    v_text_centering_offset = int(size_y / 200)

    # draw the horizontal coordinates
    for i in range(-4, 5, 2):
        text = f"{i * 10}cm"
        text = " " * max(0, 4 - len(text)) + text
        cv2.putText(
            grid_image,
            text,
            (origin_x + i * grid_size_x - h_text_centering_offset,
             origin_y + h_text_centering_offset),
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            (255, 255, 0),
            font_thickness,
        )

    # draw horizontal lines of the grid
    for hline in np.arange(grid_size_y, size_y, grid_size_y):
        cv2.line(
            grid_image,
            pt1=(grid_size_x, hline),
            pt2=(size_x - grid_size_x, hline),
            color=(255, 255, 0),
            thickness=grid_thickness
        )

    # draw the vertical coordinates
    for i in range(0, 9, 2):
        cv2.putText(
            grid_image,
            f"{i * 10}cm",
            (10, origin_y - i * grid_size_y + v_text_centering_offset),
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            (255, 255, 0),
            font_thickness
        )

    return grid_image


def debug_image(segments, background_image: Optional[np.ndarray] = None):
    """
    Generates a debug image with all the projected segments plotted with respect to the robot's origin.

    Args:
        segments (:obj:`np.ndarray`): Line segments in the ground plane relative to robot's origin
        background_image (:obj:`np.ndarray`): Optional background image

    Returns:
        :obj:`numpy array`: an OpenCV image

    """
    background_image = background_image or draw_grid_image()
    image = background_image.copy()
    # plot every segment if both ends are in the scope of the image (within 50cm from the origin)
    for color, lines in segments.items():
        for start, end in lines:
            # TODO: not sure what this IF is for
            # if not np.any(np.abs([start.x, start.y, end.x, end.y]) > 0.50 ):
            cv2.line(
                image,
                pt1=(int(start.y * -size_x) + origin_x, int(start.x * -size_y) + origin_y),
                pt2=(int(end.y * -size_x) + origin_x, int(end.x * -size_y) + origin_y),
                color=color,
                thickness=segment_thickness,
            )
    # ---
    return image
