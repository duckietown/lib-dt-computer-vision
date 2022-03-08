import os

import cv2
import numpy as np

from dt_computer_vision.camera.calibration.extrinsics.boards import CalibrationBoard8by6
from dt_computer_vision.camera.calibration.extrinsics.chessboard import find_corners
from dt_computer_vision.camera.calibration.extrinsics.ransac import estimate_homography
from dt_computer_vision.camera.types import CameraModel

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
          [0.0, 0.0, 1.0, 0.0]]
}
# principal point projected onto the ground plane (manually measured when image1 was taken)
expected_pp = [0.4, 0]
board = CalibrationBoard8by6
camera = CameraModel(**test_camera)


def test_extrinsics_calibration_image1():
    image1_fpath: str = os.path.join(assets_dir, "image1.jpg")
    image1 = cv2.imread(image1_fpath)
    # rectify image
    image1 = camera.rectifier.rectify(image1)
    # find corners
    corners = find_corners(image1, board)
    print(f"Found {len(corners)} corners.")
    assert len(corners) == (board.columns - 1) * (board.rows - 1)
    # estimate homography
    H = estimate_homography(corners, board, camera)
    # project the principal point onto the plane
    ground_pp = np.dot(H, [0, 0, 1])[:2]
    # compute error estimate
    error = np.abs(ground_pp - expected_pp).sum()
    print(f"Error is ~{error * 100:.2f}cm.")
    # make sure the error is within 4cm
    assert error <= 0.04
