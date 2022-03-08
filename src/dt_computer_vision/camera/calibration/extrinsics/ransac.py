from typing import List

import numpy as np

import cv2

from .boards import CalibrationBoard
from ... import CameraModel, Pixel


def estimate_homography(corners: List[Pixel], board: CalibrationBoard,
                        camera: CameraModel) -> np.ndarray:
    """
    Estimates the homography matrix from a list of detected corners and a given known board.

    For more information on the called to perform these operations, consult the OpenCV reference for
    `findChessboardCorners <https://docs.opencv.org/2.4/modules/calib3d/doc
    /camera_calibration_and_3d_reconstruction.html?highlight=findhomography#findchessboardcorners>`_,
    `cornerSubPix <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#cornersubpix>`_,
    `findHomography <https://docs.opencv.org/2.4/modules/calib3d/doc
    /camera_calibration_and_3d_reconstruction.html?#findhomographyi>`_.

    Args:
        corners (:obj:``list[Pixel]``): Corners as returned by
            ``dt_computer_vision.camera.calibration.extrinsics.chessboard.find_corners``
        board (:obj:``dt_computer_vision.camera.calibration.extrinsics.boards.CalibrationBoard``):
            The calibration board to use
        camera (:obj:``dt_computer_vision.camera.CameraModel``): Camera model of the camera used

    Returns:
        :obj:``numpy array``: The estimated homography

    Raises:
        RuntimeError: If no corners were found in image, or the corners couldn't be rearranged

    """
    # source corners are on the ground, easily reconstructable given a known board
    src_corners = []
    board_offset = np.array([board.x_offset, board.y_offset])
    square_size = board.square_size
    for r in range(board.rows - 1):
        for c in range(board.columns - 1):
            src_corner = np.array([(r + 1) * square_size, (c + 1) * square_size]) + board_offset
            src_corners.append(src_corner)

    # OpenCV labels corners left-to-right, top-to-bottom, let's do the same
    src_corners = src_corners[::-1]
    # opencv wants corners in an array of shape [N, 1, 2]
    src_corners = np.expand_dims(src_corners, axis=1)

    # NOTE: We're having a problem with our pattern since it's not rotation-invariant

    # destination corners are on the image, in normalized image coordinates
    dst_corners = []
    for corner in corners:
        dst_corner = camera.pixel2vector(corner).as_array()
        dst_corners.append(dst_corner)
    # opencv wants corners in an array of shape [N, 1, 2]
    dst_corners = np.expand_dims(dst_corners, axis=1)

    # compute homography from image to ground using RANSAC
    H, _ = cv2.findHomography(dst_corners, src_corners, cv2.RANSAC)
    # ---
    return H
