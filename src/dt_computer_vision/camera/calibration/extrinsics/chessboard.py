from typing import List, Tuple

import cv2
import numpy as np

from ....ground_projection.types import GroundPoint
from .boards import CalibrationBoard
from .exceptions import NoCornersFoundException
from ... import BGRImage, Pixel, CameraModel

from ... import NormalizedImagePoint


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
        raise NoCornersFoundException(
            "No corners found in image, or the corners couldn't be "
            "rearranged. Make sure the camera is positioned correctly."
        )

    # refine corners' position
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(grayscale, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)

    # return corners as Pixels
    corners = [Pixel(c[0], c[1]) for c in corners[:, 0]]
    # ---
    return corners


ImagePoints = List[NormalizedImagePoint]
GroundPointsComputed = List[GroundPoint]
GroundPointsProjected = List[GroundPoint]


def get_ground_corners_and_error(
    camera: CameraModel, corners: List[Pixel], board: CalibrationBoard, H: np.ndarray
) -> Tuple[ImagePoints, GroundPointsComputed, GroundPointsProjected, List[float]]:
    # image corners, detected above
    image_corners: List[NormalizedImagePoint] = [camera.pixel2vector(c) for c in corners]

    # ground points, easily reconstructable given a known board
    ground_corners: List[GroundPoint] = []
    board_offset = np.array([board.x_offset, board.y_offset])
    square_size = board.square_size
    for r in range(board.rows - 1):
        for c in range(board.columns - 1):
            src_corner = np.array([(r + 1) * square_size, (c + 1) * square_size]) + board_offset
            ground_corners.append(GroundPoint(*src_corner))
    # OpenCV labels corners left-to-right, top-to-bottom, let's do the same
    ground_corners = ground_corners[::-1]

    # make sure the corners match in size
    assert len(image_corners) == len(ground_corners)

    errors: List[float] = []
    ground_corners_projected: List[GroundPoint] = []
    for i, (image_corner, ground_corner) in enumerate(zip(image_corners, ground_corners)):
        # project image point onto the ground plane
        ground_corner_projected = np.dot(H, [image_corner.x, image_corner.y, 1])
        # remove homogeneous coordinate
        ground_corner_projected = (ground_corner_projected / ground_corner_projected[2])[:2]
        # wrap x,y into a GroundPoint object
        ground_corner_projected_p = GroundPoint(
            round(ground_corner_projected[0], 4), round(ground_corner_projected[1], 4)
        )
        ground_corners_projected.append(ground_corner_projected_p)
        # compute error
        error = np.linalg.norm(ground_corner_projected - ground_corner.as_array())
        errors.append(error)
        # store corner

    return image_corners, ground_corners, ground_corners_projected, errors


def compute_placement_error(
    corners: List[Pixel], board: CalibrationBoard, errors: List[float]
) -> float:
    # compute vertical line of corners
    half_width: int = int(np.floor((board.columns - 1) / 2))
    vertical_principal_corners = np.arange(0, len(corners), board.columns - 1) + half_width
    vertical_principal_errors = [errors[i] for i in vertical_principal_corners]
    # compute horizontal line of corners
    half_height: int = int(np.floor((board.rows - 1) / 2))
    horizontal_principal_corners = half_height * (board.columns - 1) + np.arange(0, board.columns - 1)
    horizontal_principal_errors = [errors[i] for i in horizontal_principal_corners]
    # placement error is the average of all these point's reprojection errors
    placement_error = np.average(np.concatenate((horizontal_principal_errors, vertical_principal_errors)))
    # ---
    return placement_error
