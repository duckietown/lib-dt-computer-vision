import os
from typing import Tuple

import cv2
import numpy as np

from dt_computer_vision.line_detection import LineDetector, ColorRange, Detections
from dt_computer_vision.line_detection.rendering import draw_segments

this_dir: str = os.path.dirname(os.path.realpath(__file__))
this_lib: str = os.path.basename(this_dir)
assets_dir: str = os.path.join(this_dir, "..", "..", "..", "assets")
output_dir: str = os.path.join(this_dir, "..", "..", "..", "out", "test-results", this_lib)
colors = {
    "red": {
        "low_1": [0, 140, 100],
        "high_1": [15, 255, 255],
        "low_2": [165, 140, 100],
        "high_2": [180, 255, 255],
    },
    "white": {
        "low": [0, 0, 150],
        "high": [180, 100, 255]
    },
    "yellow": {
        "low": [25, 140, 100],
        "high": [45, 255, 255]
    }
}

os.makedirs(output_dir, exist_ok=True)


def detect_color(image: np.ndarray, color: str) -> Tuple[Detections, np.ndarray]:
    detector = LineDetector()
    color_range: ColorRange = ColorRange.fromDict(colors[color])
    detections: Detections = detector.detect(image, [color_range])[0]
    image0_dets = draw_segments(image, {color_range: detections})
    return detections, image0_dets


def test_image_0_white():
    color: str = "white"
    expected_detections: int = 766
    image0_fpath: str = os.path.join(assets_dir, "image0.jpg")
    image0 = cv2.imread(image0_fpath)
    # ---
    detections, image0_dets = detect_color(image0, color)
    # ---
    assert len(detections.lines) == expected_detections
    image0_dets_fpath: str = os.path.join(output_dir, f"image0_{color}.jpg")
    cv2.imwrite(image0_dets_fpath, image0_dets)


def test_image_0_yellow():
    color: str = "yellow"
    expected_detections: int = 9
    image0_fpath: str = os.path.join(assets_dir, "image0.jpg")
    image0 = cv2.imread(image0_fpath)
    # ---
    detections, image0_dets = detect_color(image0, color)
    # ---
    assert len(detections.lines) == expected_detections
    image0_dets_fpath: str = os.path.join(output_dir, f"image0_{color}.jpg")
    cv2.imwrite(image0_dets_fpath, image0_dets)


def test_image_0_red():
    color: str = "red"
    expected_detections: int = 6
    image0_fpath: str = os.path.join(assets_dir, "image0.jpg")
    image0 = cv2.imread(image0_fpath)
    # ---
    detections, image0_dets = detect_color(image0, color)
    # ---
    assert len(detections.lines) == expected_detections
    image0_dets_fpath: str = os.path.join(output_dir, f"image0_{color}.jpg")
    cv2.imwrite(image0_dets_fpath, image0_dets)
