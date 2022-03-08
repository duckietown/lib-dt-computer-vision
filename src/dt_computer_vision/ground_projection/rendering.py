#!/usr/bin/env python3

from typing import Optional

import cv2
import numpy as np


def draw_grid_image():
    """
    Generates a grid image with distances from the robot's origin.

    Returns:
        :obj:`numpy array`: an OpenCV image

    """
    # dimensions of the image are 1m x 1m so, 1px = 2.5mm
    # the origin is at x=200 and y=300

    # initialize gray image
    grid_image = np.ones((400, 400, 3), np.uint8) * 128

    # draw vertical lines of the grid
    for vline in np.arange(40, 361, 40):
        cv2.line(
            grid_image, pt1=(vline, 20), pt2=(vline, 300), color=(255, 255, 0), thickness=1
        )

    # draw the coordinates
    cv2.putText(
        grid_image,
        "-20cm",
        (120 - 25, 300 + 15),
        cv2.FONT_HERSHEY_PLAIN,
        0.8,
        (255, 255, 0),
        1,
    )
    cv2.putText(
        grid_image,
        "  0cm",
        (200 - 25, 300 + 15),
        cv2.FONT_HERSHEY_PLAIN,
        0.8,
        (255, 255, 0),
        1,
    )
    cv2.putText(
        grid_image,
        "+20cm",
        (280 - 25, 300 + 15),
        cv2.FONT_HERSHEY_PLAIN,
        0.8,
        (255, 255, 0),
        1,
    )

    # draw horizontal lines of the grid
    for hline in np.arange(20, 301, 40):
        cv2.line(
            grid_image, pt1=(40, hline), pt2=(360, hline), color=(255, 255, 0), thickness=1
        )

    # draw the coordinates
    cv2.putText(
        grid_image, "20cm", (2, 220 + 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1
    )
    cv2.putText(
        grid_image, " 0cm", (2, 300 + 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1
    )

    # draw robot marker at the center
    cv2.line(
        grid_image,
        pt1=(200 + 0, 300 - 20),
        pt2=(200 + 0, 300 + 0),
        color=(255, 0, 0),
        thickness=1,
    )

    cv2.line(
        grid_image,
        pt1=(200 + 20, 300 - 20),
        pt2=(200 + 0, 300 + 0),
        color=(255, 0, 0),
        thickness=1,
    )

    cv2.line(
        grid_image,
        pt1=(200 - 20, 300 - 20),
        pt2=(200 + 0, 300 + 0),
        color=(255, 0, 0),
        thickness=1,
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
    # dimensions of the image are 1m x 1m so, 1px = 2.5mm
    # the origin is at x=200 and y=300

    background_image = background_image or draw_grid_image()

    # map segment color variables to BGR colors
    color_map = {Segment.WHITE: (255, 255, 255), Segment.RED: (0, 0, 255), Segment.YELLOW: (0, 255, 255)}

    image = background_image.copy()

    # plot every segment if both ends are in the scope of the image (within 50cm from the origin)
    for segment in segments:
        if not np.any(
                np.abs([segment.points[0].x, segment.points[0].y, segment.points[1].x, segment.points[1].y])
                > 0.50
        ):
            cv2.line(
                image,
                pt1=(int(segment.points[0].y * -400) + 200, int(segment.points[0].x * -400) + 300),
                pt2=(int(segment.points[1].y * -400) + 200, int(segment.points[1].x * -400) + 300),
                color=color_map.get(segment.color, (0, 0, 0)),
                thickness=1,
            )

    return image
