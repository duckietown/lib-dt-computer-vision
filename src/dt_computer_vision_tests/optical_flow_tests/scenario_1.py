import os
from typing import List

import cv2
import numpy as np

this_dir: str = os.path.dirname(os.path.realpath(__file__))
this_lib: str = os.path.basename(this_dir)
assets_dir: str = os.path.join(this_dir, "..", "..", "..", "assets")

image_fpaths = [
    os.path.join(assets_dir, f"extrinsics/dd24/real-world/scenario1/{distance}cm.png") for distance in [20, 50]
    ]

distance_cm_to_image_fpath_map = {
    20: image_fpaths[0],
    50: image_fpaths[1]
}

yaml_homography_fpath = os.path.join(
    assets_dir,
    "extrinsics",
    "dd24",
    "real-world",
    "scenario1",
    "homography.yaml",
)

yaml_intrinsics_fpath = os.path.join(
    assets_dir,
    "extrinsics",
    "dd24",
    "real-world",
    "scenario1",
    "calibration-intrinsic-dd24.yaml",
)

def generate_translated_images_sequence(base_image_fpath: str, frame_count: int = 30) -> List[np.ndarray]:
    image = cv2.imread(base_image_fpath)

    # Get the image width and height
    height, width = image.shape[:2]

    # Define the translation amount
    translation_amount = int(width / 2)

    # Create a black image with the same size as the original image
    black_image = np.zeros_like(image)

    # Generate the sequence of translated images
    sequence = []

    for frame in range(frame_count):
        # Calculate the translation amount for the current frame
        translation = int((frame / frame_count) * translation_amount)
        
        # Create a translation matrix
        M = np.float32([[1, 0, translation], [0, 1, 0]])
        
        # Apply the translation to the original image
        translated_image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Combine the translated image with the black image to fill the missing pixels
        final_image = cv2.bitwise_or(translated_image, black_image)
        
        # Add the final image to the sequence
        sequence.append(final_image)
    
    return sequence
