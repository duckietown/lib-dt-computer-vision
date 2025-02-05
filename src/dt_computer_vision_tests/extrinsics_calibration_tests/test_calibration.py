import os

import cv2
import numpy as np

from dt_computer_vision.camera.calibration.extrinsics.boards import CalibrationBoard8by6, ReferenceFrame
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

def test_calibration_board_corners():
    board = CalibrationBoard8by6
    corners = board.corners(reference_frame=ReferenceFrame.ROBOT)
    top_left = corners[0]
    bottom_right = corners[-1]
    
    # Check that the top left corner has positive x and y coordinates
    assert top_left.x >= 0, f"Expected x >= 0, got {top_left.x}."
    assert top_left.y >= 0, f"Expected y >= 0, got {top_left.y}."

    # Check that the bottom right corner has positive x and negative y coordinates
    assert bottom_right.x >= 0, f"Expected x >= 0, got {bottom_right.x}."
    assert bottom_right.y <= 0, f"Expected y >= 0, got {bottom_right.y}."
    
    # Check that the top left corner has x coordinate board.square_size*number of rows + x offset
    # and y coordinate board.y_offset-board.square_size
    assert top_left.x == board.square_size * board.rows + board.x_offset, f"Expected x == {board.square_size * board.rows + board.x_offset}, got {top_left.x}."
    assert top_left.y == board.y_offset - board.square_size, f"Expected y == {board.y_offset - board.square_size}, got {top_left.y}."
    
    # Check that the bottom right corner has x coordinate x offset and y coordinate y_offset - board.square_size*(number of columns+1)
    assert bottom_right.x == board.x_offset + board.square_size, f"Expected x == {board.x_offset + board.square_size}, got {bottom_right.x}."
    assert bottom_right.y == board.y_offset - board.square_size * board.columns, f"Expected y == {board.y_offset - board.square_size * (board.columns)}, got {bottom_right.y}."
    
def test_extrinsics_calibration_image1():
    image1_fpath: str = os.path.join(assets_dir, "image1.jpg")
    image1 = cv2.imread(image1_fpath)
    # rectify image
    image1 = camera.rectifier.rectify(image1)
    # find corners
    corners = find_corners(image1, board)
    print(f"Found {len(corners)} corners.")
    assert len(corners) == board.columns * board.rows, f"Expected {len(corners)} corners."
    # estimate homography
    H = estimate_homography(corners, board, camera, ref_frame=ReferenceFrame.ROBOT)
    # project the principal point onto the plane
    ground_pp = np.dot(H, [0, 0, 1])
    ground_pp = (ground_pp / ground_pp[2])[:2]
    # compute error estimate
    error = np.linalg.norm(ground_pp - expected_pp)
    print(f"Error is ~{error * 100:.2f}cm.")
    # make sure the error is within 3cm
    assert error <= 0.03, f"Error is {error * 100:.2f}cm, expected less than 3cm."

if __name__ == "__main__":
    test_extrinsics_calibration_image1()