import os
from typing import List

import cv2

from dt_computer_vision.anti_instagram import AntiInstagram
from dt_computer_vision.camera import BGRImage

Color = List[int]

this_dir: str = os.path.dirname(os.path.realpath(__file__))
this_lib: str = os.path.basename(this_dir)
assets_dir: str = os.path.join(this_dir, "..", "..", "..", "assets")
output_dir: str = os.path.join(this_dir, "..", "..", "..", "out", "test-results", this_lib)

os.makedirs(output_dir, exist_ok=True)


def _load_images(num: int) -> List[BGRImage]:
    out = []
    for filter in ["_darker", "", "_brighter"]:
        image_fpath: str = os.path.join(assets_dir, f"image{num}{filter}.jpg")
        out.append(cv2.imread(image_fpath))
    return out


def _perform_test(name: str, image: int, original: BGRImage, update: BGRImage, expected: Color):
    ai = AntiInstagram()
    test_output_dir = os.path.join(output_dir, f"test_image{image}_{name}")
    os.makedirs(test_output_dir, exist_ok=True)
    # ---
    # before update
    assert ai.lower_threshold == [0] * 3
    assert ai.higher_threshold == [255] * 3
    # update
    ai.update(update)
    image_corrected = ai.apply(original)
    # after update
    print("ai.lower_threshold", ai.lower_threshold)
    print("ai.higher_threshold", ai.higher_threshold)
    assert ai.lower_threshold == expected
    assert ai.higher_threshold == [255] * 3
    # ---
    image_original_fpath: str = os.path.join(test_output_dir, f"original.jpg")
    cv2.imwrite(image_original_fpath, original)
    image_original_fpath: str = os.path.join(test_output_dir, f"update.jpg")
    cv2.imwrite(image_original_fpath, update)
    image_corrected_fpath: str = os.path.join(test_output_dir, f"corrected.jpg")
    cv2.imwrite(image_corrected_fpath, image_corrected)


def test_image_0_original():
    image_num = 0
    test_name = "original"
    _, original, _ = _load_images(image_num)
    _perform_test(test_name, image_num, original, original, [66, 69, 73])


def test_image_0_darker():
    image_num = 0
    test_name = "darker"
    darker, original, _ = _load_images(image_num)
    _perform_test(test_name, image_num, original, darker, [37, 38, 41])


def test_image_0_brighter():
    image_num = 0
    test_name = "brighter"
    _, original, brighter = _load_images(image_num)
    _perform_test(test_name, image_num, original, brighter, [99, 103, 110])
