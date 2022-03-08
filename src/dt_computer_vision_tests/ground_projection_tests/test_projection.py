import os

import cv2

from dt_computer_vision.ground_projection.ground_projector import GroundProjector
from dt_computer_vision.ground_projection.types import CameraModel, Point, Pixel

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
    "R": [[1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]],
      # TODO: this homography is wrong
    "H": [
        [6.466493116650289e-06, -0.00020595599434109174, -0.24445439714590136],
        [0.0011641323888359127, -1.4519582718982904e-05, -0.37033417682986924],
        [0.0003286437429322083, -0.00786476462833761, 1.0]
    ]
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
        print(pixel0, pixel1)

        point0 = camera.pixel2vector(pixel0)
        point1 = camera.pixel2vector(pixel1)
        print(point0, point1)

        point0_ground = projector.project_to_ground(point0)
        point1_ground = projector.project_to_ground(point1)
        print(point0_ground, point1_ground)

        print()

    assert False
