#!/usr/bin/env python3

from typing import Optional, Dict, List, Tuple, Union

import cv2
import numpy as np

from dt_computer_vision.ground_projection.types import GroundPoint

Color = Tuple[int, int, int]
Segment = Tuple[GroundPoint, GroundPoint]

GRID_SIZE: int = 4
# resolution of each cell in meters
RESOLUTION: float = 0.1
# how often (number of cells) do we draw a tick with the value
TICKS_EVERY: int = 2
# everything here is calibrated for a 1000x1000 image
SCALE: int = 1000
S_SEGMENT_THICKNESS: int = 10
S_PADDING: int = 80
S_GRID_THICKNESS: int = 3
S_FONT_SIZE: int = 3
S_FONT_THICKNESS: int = 3


def draw_grid_image(
        size: Tuple[int, int],
        # default values
        grid_size: Union[int, Tuple[int, int]] = GRID_SIZE,
        scale: int = SCALE,
        s_padding: int = S_PADDING,
        s_grid_thickness: int = S_GRID_THICKNESS,
        s_font_size: int = S_FONT_SIZE,
        s_font_thickness: int = S_FONT_THICKNESS,
        ticks_every: Union[int, Tuple[int, int]] = TICKS_EVERY,
        resolution: Union[float, Tuple[float, float]] = RESOLUTION,
        start_y: float = 0.0
):
    """
    Generates a grid image with distances from the robot's origin.

    Args:
        size (tuple): Size of the image to draw

        grid_size (int):                     Number of cells to draw
        scale (int):                         Scale of the image
        s_padding (int):                     Padding of the grid
        s_grid_thickness (dict):             Grid lines thickness
        s_font_size (np.ndarray):            Font size
        s_font_thickness (int):              Font thickness
        ticks_every (Union[int, Tuple[int, int]]):

    Returns:
        :obj:`numpy array`: an OpenCV image

    """
    # if grid_size is an integer, it means that the grid is square
    grid_size = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)

    # if resolution is an integer, it means that the grid is linear
    resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)

    # if ticks_every is an integer, it means that we use the same value on both x and y
    ticks_every = ticks_every if isinstance(ticks_every, tuple) else (ticks_every, ticks_every)

    # unpack
    resolution_x, resolution_y = resolution
    ticks_every_x, ticks_every_y = ticks_every

    # image is square
    size_x, size_y = size

    s = max(size_x, size_y) / scale
    padding = int(s_padding * s)
    grid_thickness = max(1, int(s_grid_thickness * s))
    font_size = s_font_size * s
    font_thickness = max(1, int(s_font_thickness * s))

    grid_size_x, grid_size_y = grid_size

    half_grid_size_horizontal = int(grid_size_x / 2)
    cell_size_x = int((size_x - 3 * padding) / grid_size_x)
    cell_size_y = int((size_y - 3 * padding) / grid_size_y)
    origin_x, origin_y = 2 * padding + half_grid_size_horizontal * cell_size_x, size_y - 2 * padding

    # initialize gray image
    grid_image = np.ones((size_y, size_x, 3), np.uint8) * 128

    # draw vertical lines of the grid
    for col in range(grid_size_x + 1):
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

    # write the horizontal coordinates
    for i in range(0, half_grid_size_horizontal+1, ticks_every_x):
        for v in {i, i * -1}:
            text = f"{v * int(resolution_x * 100)}cm"
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
    for row in range(grid_size_y + 1):
        row_y = row * cell_size_y
        cv2.line(
            grid_image,
            pt1=(2 * padding, padding + row_y),
            pt2=(size_x - padding, padding + row_y),
            color=(255, 255, 0),
            thickness=grid_thickness
        )

    # write the vertical coordinates
    for i in range(0, grid_size_y + 1, ticks_every_y):
        cv2.putText(
            grid_image,
            f"{int(start_y * 100) + i * int(resolution_y * 100)}cm",
            (10, origin_y - i * cell_size_y + v_text_centering_offset),
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            (255, 255, 255),
            font_thickness
        )

    # draw central vertical line
    col_x = half_grid_size_horizontal * cell_size_x
    cv2.line(
        grid_image,
        pt1=(2 * padding + col_x, padding),
        pt2=(2 * padding + col_x, origin_y),
        color=(255, 0, 0),
        thickness=grid_thickness
    )

    # draw wheel's axis
    if start_y == 0.0:
        cv2.line(
            grid_image,
            pt1=(2 * padding, origin_y),
            pt2=(2 * padding + grid_size_x * cell_size_x, origin_y),
            color=(255, 0, 0),
            thickness=grid_thickness
        )

    return grid_image


def debug_image(
        segments: Dict[Color, List[Segment]],
        size: Tuple[int, int],
        background_image: Optional[np.ndarray] = None,
        # default values
        grid_size: Union[int, Tuple[int, int]] = GRID_SIZE,
        scale: int = SCALE,
        s_segment_thickness: int = S_SEGMENT_THICKNESS,
        s_padding: int = S_PADDING,
        resolution: Union[float, Tuple[float, float]] = RESOLUTION,
        start_y: float = 0.0
):
    """
    Generates a debug image with all the projected segments plotted with respect to the
    robot's origin.

    Args:
        segments (:obj:`dict`): Line segments in the ground plane relative to robot's origin
        size (:obj:`tuple`): Size of the image to draw
        background_image (:obj:`np.ndarray`): Optional background image

        grid_size (:obj:`int`): Number of cells to draw
        scale (:obj:`int`): Scale of the image
        s_segment_thickness (:obj:`int`): Thickness of the segments drawn
        s_padding (:obj:`int`): Padding of the grid

    Returns:
        :obj:`numpy array`: an OpenCV image

    """
    background_image = background_image if background_image is not None else draw_grid_image(
        size,
        grid_size=grid_size,
        scale=scale,
        s_padding=s_padding,
        resolution=resolution,
        start_y=start_y,
    )

    # if grid_size is an integer, it means that the grid is square
    grid_size = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)

    # if resolution is an integer, it means that the grid is linear
    resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)

    # unpack
    resolution_x, resolution_y = resolution

    grid_size_x, grid_size_y = grid_size
    size_x, size_y = size

    s = max(size_x, size_y) / scale
    segment_thickness = max(1, int(s_segment_thickness * s))
    padding = int(s_padding * s)

    half_grid_size_horizontal = int(grid_size_x / 2)
    cell_size_x = int((size_x - 3 * padding) / grid_size_x)
    cell_size_y = int((size_y - 3 * padding) / grid_size_y)
    origin_x, origin_y = 2 * padding + half_grid_size_horizontal * cell_size_x, size_y - 2 * padding

    image = background_image.copy()
    # plot every segment if both ends are in the scope of the image (within 50cm from the origin)
    for color, lines in segments.items():
        for start, end in lines:
            # only draw segments that are within the grid
            if np.any(np.abs([start.y, end.y]) > (half_grid_size_horizontal * resolution_x)):
                continue
            if np.any(np.abs([start.x, end.x]) > (grid_size_y * resolution_y)):
                continue
            # draw segment
            cv2.line(
                image,
                pt1=(
                    origin_x + int((-start.y / resolution_x) * cell_size_x),
                    origin_y - int(((start.x - start_y) / resolution_y) * cell_size_y)
                ),
                pt2=(
                    origin_x + int((-end.y / resolution_x) * cell_size_x),
                    origin_y - int(((end.x - start_y) / resolution_y) * cell_size_y)
                ),
                color=color,
                thickness=segment_thickness,
            )
    # ---
    return image
