from typing import List

import numpy as np

import cv2

from dt_computer_vision.camera.homography import Homography
from dt_computer_vision.camera.types import NormalizedImagePoint

from .boards import CalibrationBoard, ReferenceFrame
from ... import CameraModel, Pixel


def estimate_homography(
    corners: List[Pixel],
    board: CalibrationBoard,
    camera: CameraModel,
    ref_frame: ReferenceFrame = ReferenceFrame.BOARD,
    enforce_orientation: bool = True,
) -> Homography:
    """
    Estimates the homography matrix from a list of detected corners and a given known board.

    For more information on the called to perform these operations, consult the OpenCV reference for
    `findChessboardCorners <https://docs.opencv.org/2.4/modules/calib3d/doc
    /camera_calibration_and_3d_reconstruction.html?highlight=findhomography#findchessboardcorners>`_,
    `cornerSubPix <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#cornersubpix>`_,
    `findHomography <https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?#findhomographyi>`.

    Args:
        corners (:obj:``List[Pixel]``): Corners in pixel coordinates as returned by
            ``dt_computer_vision.camera.calibration.extrinsics.chessboard.find_corners``
        board (:obj:``dt_computer_vision.camera.calibration.extrinsics.boards.CalibrationBoard``):
            The calibration board to use
        camera (:obj:``dt_computer_vision.camera.CameraModel``): Camera model of the camera used
        ref_frame (:obj:``ReferenceFrame``): The reference frame of the calibration board

    Returns:
        :obj:``Homography``: The estimated homography, mapping normalized coordinates in the image
        to the ground plane (metric space).

    Raises:
        RuntimeError: If no corners were found in image, or the corners couldn't be rearranged

    """
    # Get the ground corners from the board
    board_corners = board.corners(reference_frame=ref_frame)

    # opencv wants corners in an array of shape [N, 1, 2]
    board_corners = np.expand_dims(
            np.array([corner.as_array() for corner in board_corners]),
        axis=1
        )

    # NOTE: We're having a problem with our pattern since it's not rotation-invariant

    # destination corners are on the image, in normalized image coordinates
    image_corners : List [NormalizedImagePoint] = []
    
    if enforce_orientation is True:
        # re-orient corners if necessary
        p0, p_1 = corners[0], corners[-1]
        cx, cy = camera.cx, camera.cy

        # we want the first point (red) to be to the left of the principal point
        if p0.x > cx and p_1.x < cx:
            corners = corners[::-1]

    for corner in corners:
        dst_corner = camera.pixel2vector(corner).as_array()
        image_corners.append(dst_corner)
    # opencv wants corners in an array of shape [N, 1, 2]
    image_corners = np.expand_dims(image_corners, axis=1)

    # compute homography from image to ground using RANSAC
    H, _ = cv2.findHomography(image_corners, board_corners, cv2.RANSAC)
    # ---
    return np.array(H).view(Homography)
