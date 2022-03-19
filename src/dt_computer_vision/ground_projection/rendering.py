#!/usr/bin/env python3

from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from dt_computer_vision.ground_projection.types import GroundPoint

Color = Tuple[int, int, int]
Segment = Tuple[GroundPoint, GroundPoint]

grid_size = 8
# everything here is calibrated for a 1000x1000 image
scale = 1000
s_segment_thickness = 10
s_padding = 80
s_grid_thickness = 3
s_font_size = 3
s_font_thickness = 3


def draw_grid_image(size: Tuple[int, int]):
    """
    Generates a grid image with distances from the robot's origin.

    Returns:
        :obj:`numpy array`: an OpenCV image

    """
    size_x, size_y = size

    s = max(size_x, size_y) / scale
    padding = int(s_padding * s)
    grid_thickness = max(1, int(s_grid_thickness * s))
    font_size = s_font_size * s
    font_thickness = max(1, int(s_font_thickness * s))

    half_grid_size = int(grid_size / 2)
    cell_size_x = int((size_x - 3 * padding) / grid_size)
    cell_size_y = int((size_y - 3 * padding) / grid_size)
    origin_x, origin_y = 2 * padding + int(grid_size / 2) * cell_size_x, size_y - 2 * padding

    # initialize gray image
    grid_image = np.ones((size_y, size_x, 3), np.uint8) * 128

    # draw vertical lines of the grid
    for col in range(grid_size + 1):
        col_x = col * cell_size_x
        cv2.line(
            grid_image,
            pt1=(2 * padding + col_x, padding),
            pt2=(2 * padding + col_x, origin_y),
            color=(255, 255, 0),
            thickness=grid_thickness
        )

    h_text_centering_offset = int(size_x / 20)
    v_text_centering_offset = int(size_y / 200)

    # draw the horizontal coordinates
    for i in range(0, half_grid_size+1, 2):
        for v in {i, i * -1}:
            text = f"{v * 10}cm"
            text = " " * max(0, 4 - len(text)) + text
            cv2.putText(
                grid_image,
                text,
                (origin_x + v * cell_size_x - h_text_centering_offset,
                 origin_y + h_text_centering_offset),
                cv2.FONT_HERSHEY_PLAIN,
                font_size,
                (255, 255, 255),
                font_thickness,
            )

    # draw horizontal lines of the grid
    for row in range(grid_size + 1):
        row_y = row * cell_size_y
        cv2.line(
            grid_image,
            pt1=(2 * padding, padding + row_y),
            pt2=(size_x - padding, padding + row_y),
            color=(255, 255, 0),
            thickness=grid_thickness
        )

    # draw the vertical coordinates
    for i in range(0, grid_size + 1, 2):
        cv2.putText(
            grid_image,
            f"{i * 10}cm",
            (10, origin_y - i * cell_size_y + v_text_centering_offset),
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            (255, 255, 255),
            font_thickness
        )

    # draw central vertical line
    col_x = half_grid_size * cell_size_x
    cv2.line(
        grid_image,
        pt1=(2 * padding + col_x, padding),
        pt2=(2 * padding + col_x, origin_y),
        color=(255, 0, 0),
        thickness=grid_thickness
    )

    # draw wheel's axis
    cv2.line(
        grid_image,
        pt1=(2 * padding, origin_y),
        pt2=(2 * padding + grid_size * cell_size_x, origin_y),
        color=(255, 0, 0),
        thickness=grid_thickness
    )

    return grid_image


def debug_image(segments: Dict[Color, List[Segment]], size: Tuple[int, int],
                background_image: Optional[np.ndarray] = None):
    """
    Generates a debug image with all the projected segments plotted with respect to the
    robot's origin.

    Args:
        segments (:obj:`dict`): Line segments in the ground plane relative to robot's origin
        size (:obj:`tuple`): Image of the image to draw
        background_image (:obj:`np.ndarray`): Optional background image

    Returns:
        :obj:`numpy array`: an OpenCV image

    """
    size_x, size_y = size

    s = max(size_x, size_y) / scale
    segment_thickness = max(1, int(s_segment_thickness * s))
    padding = int(s_padding * s)

    half_grid_size = int(grid_size / 2)
    cell_size_x = int((size_x - 3 * padding) / grid_size)
    cell_size_y = int((size_y - 3 * padding) / grid_size)
    origin_x, origin_y = 2 * padding + int(grid_size / 2) * cell_size_x, size_y - 2 * padding

    background_image = background_image if background_image is not None else draw_grid_image(size)
    image = background_image.copy()
    # plot every segment if both ends are in the scope of the image (within 50cm from the origin)
    for color, lines in segments.items():
        for start, end in lines:
            # only draw segments that are within the grid
            if np.any(np.abs([start.x, start.y, end.x, end.y]) > (half_grid_size / 10)):
                continue
            # draw segment
            cv2.line(
                image,
                pt1=(
                    origin_x + int(-(start.y * 10) * cell_size_x),
                    origin_y - int((start.x * 10) * cell_size_y)
                ),
                pt2=(
                    origin_x + int(-(end.y * 10) * cell_size_x),
                    origin_y - int((end.x * 10) * cell_size_y)
                ),
                color=color,
                thickness=segment_thickness,
            )
    # ---
    return image
