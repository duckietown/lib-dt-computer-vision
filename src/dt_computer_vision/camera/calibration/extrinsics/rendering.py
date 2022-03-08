from typing import List

import numpy as np
import cv2

from .boards import CalibrationBoard
from ... import BGRImage, Pixel


def draw_corners(image: BGRImage, board: CalibrationBoard, corners: List[Pixel]) -> BGRImage:
    """
    Draws the given corners on the given image.

    Note:
        The provided image should be rectified.

    Args:
        image (:obj:``numpy array``): A color (3-channel) OpenCV image
        board (:obj:``dt_computer_vision.camera.calibration.extrinsics.boards.CalibrationBoard``):
            The calibration board to look for
        corners: (:obj:``list[Pixel]``): Corners as returned by
        ``dt_computer_vision.camera.calibration.extrinsics.chessboard.find_corners``

    Returns:
        :obj:``numpy array``: The estimated homography

    Raises:
        RuntimeError: If no corners were found in image, or the corners couldn't be rearranged

    """
    grayscale = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    corners_num = (board.columns - 1, board.rows - 1)
    # opencv wants corners in an array of shape [N, 1, 2]
    corners = np.expand_dims([c.as_array() for c in corners], axis=1)
    # draw chessboard on image
    return cv2.drawChessboardCorners(grayscale, corners_num, corners, True)
