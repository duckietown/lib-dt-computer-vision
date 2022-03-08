from typing import List

import cv2

from .boards import CalibrationBoard
from ... import BGRImage, Pixel


def find_corners(image: BGRImage, board: CalibrationBoard) -> List[Pixel]:
    """
    Finds the corners of the given board in the given image.

    Note:
        The provided image should be rectified.

    Args:
        image (:obj:``numpy array``): A color (3-channel) OpenCV image
        board (:obj:``dt_computer_vision.camera.calibration.extrinsics.boards.CalibrationBoard``):
            The calibration board to look for

    Returns:
        :obj:``list[dt_computer_vision.camera.Pixel]``: Corners detected in the given image,
        ordered left-right, top-bottom.

    Raises:
        RuntimeError: If no corners were found in image, or the corners couldn't be rearranged

    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_num = (board.columns - 1, board.rows - 1)

    # find corners in the image
    ret, corners = cv2.findChessboardCorners(grayscale, corners_num, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not ret:
        raise RuntimeError("No corners found in image, or the corners couldn't be rearranged. "
                           "Make sure that the camera is positioned correctly.")

    # refine corners' position
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(grayscale,
                               corners,
                               winSize=(11, 11),
                               zeroZone=(-1, -1),
                               criteria=criteria)

    # return corners as Pixels
    corners = [Pixel(c[0], c[1]) for c in corners[:, 0]]
    # ---
    return corners
