import os

import cv2

from dt_computer_vision.ground_projection.ground_projector import GroundProjector
from dt_computer_vision.camera.types import CameraModel, Pixel

from dt_computer_vision_tests.line_detector_tests.test_detection import assets_dir, detect_color

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


def test_example1():
    color: str = "yellow"
    image0_fpath: str = os.path.join(assets_dir, "image0.jpg")
    image0 = cv2.imread(image0_fpath)
    # ---
    detections, image0_dets = detect_color(image0, color)
    # ---
    for detection in detections.lines:
        pixel0 = Pixel(float(detection[0]), float(detection[1]))
        pixel1 = Pixel(float(detection[2]), float(detection[3]))

        point0 = camera.pixel2vector(pixel0)
        point1 = camera.pixel2vector(pixel1)

        point0_ground = projector.distorted_to_ground(point0)
        point1_ground = projector.distorted_to_ground(point1)

        # print(pixel0, pixel1)
        # print(point0, point1)
        # print(point0_ground, point1_ground)
        # print()

    assert False
