import os
from typing import List

import cv2
import numpy as np

from dt_computer_vision.camera import NormalizedImagePoint
from dt_computer_vision.camera.calibration.extrinsics.boards import CalibrationBoard8by6, \
    CalibrationBoard
from dt_computer_vision.camera.calibration.extrinsics.chessboard import find_corners
from dt_computer_vision.camera.types import CameraModel
from dt_computer_vision.ground_projection.ground_projector import GroundProjector
from dt_computer_vision.ground_projection.types import GroundPoint
from dt_computer_vision_tests.line_detection_tests.test_detection import assets_dir


# NOTE: this is from the real `myrobot` duckiebot at TTIC, March 2022
test_camera = {
    "width": 640,
    "height": 480,
    "K": [[295.79606866959824, 0.0, 321.2621599038631],
          [0.0, 299.5389048862878, 241.73616515312332],
          [0.0, 0.0, 1.0]],
    "D": [-0.23543978771661125,
          0.03637781479419574,
          -0.0033069818601306755,
          -0.0012140708179525926,
          0.0],
    "P": [[201.14027404785156, 0.0, 319.5586620845679, 0.0],
          [0.0, 239.74398803710938, 237.60151004037834, 0.0],
          [0.0, 0.0, 1.0, 0.0]],
    # NOTE: this homography is computed in the 20-entrinsics-calibration jupyter notebook
    "H": [[-2.42749970e-02, 9.46389079e-02, 3.81909422e-01],
          [-4.55028567e-01, -1.17673909e-03, -1.87813039e-02],
          [-1.46006785e-01, 3.29784838e+00, 1]]
}

camera = CameraModel(**test_camera)
projector = GroundProjector(camera)
board: CalibrationBoard = CalibrationBoard8by6


def test_reprojection_error():
    # load image with chessboard
    image_fpath: str = os.path.join(assets_dir, "image1.jpg")
    image = cv2.imread(image_fpath)

    # rectify image
    image_rect = camera.rectifier.rectify(image)

    # find corners
    corners = find_corners(image_rect, board)
    print(f"Found {len(corners)} corners.")
    assert len(corners) == (board.columns - 1) * (board.rows - 1)

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

    errors = []
    for i, (image_corner, ground_corner) in enumerate(zip(image_corners, ground_corners)):
        # project image point onto the ground plane
        ground_corner_computed = projector.vector2ground(image_corner)
        # compute error
        error = np.linalg.norm(np.abs(
            ground_corner_computed.as_array() - ground_corner.as_array()
        ))
        # make sure the error is below 5.6mm
        print(f"Corner {i} has a reprojection error of {error}m")
        assert error <= 0.0056
        # store error
        errors.append(error)

    # compute average error
    avg_error = np.average(errors)
    std_error = np.std(errors)
    # make sure the average error is below 2.3mm
    print(f"Average reprojection error is {avg_error}m")
    assert avg_error < 0.0023
    # make sure the std error is below 1.4mm
    print(f"Std reprojection error is {std_error}m")
    assert std_error < 0.0014
