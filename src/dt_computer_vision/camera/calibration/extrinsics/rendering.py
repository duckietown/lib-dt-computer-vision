import os
from typing import List, Tuple, Optional, Union

import numpy as np
import cv2

from dt_computer_vision.ground_projection.types import GroundPoint
from .boards import CalibrationBoard
from ... import BGRImage, Pixel

from dt_computer_vision.ground_projection.rendering import draw_grid_image

assets_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
gui_active_background_image_fpath: str = os.path.join(assets_dir, "gui_active.png")
gui_active_background_image: BGRImage = cv2.imread(gui_active_background_image_fpath)
gui_inactive_background_image_fpath: str = os.path.join(assets_dir, "gui_inactive.png")
gui_inactive_background_image: BGRImage = cv2.imread(gui_inactive_background_image_fpath)
validation_gui_active_background_image_fpath: str = os.path.join(assets_dir, "validation_gui.png")
validation_gui_active_background_image: BGRImage = cv2.imread(validation_gui_active_background_image_fpath)

GUI_LEFT_IMAGE_ROI = (20, 20, 640, 480)
GUI_RIGHT_IMAGE_ROI = (700, 20, 560, 480)
GUI_BOX1_ROI = (20, 575, 300, 95)
GUI_BOX2_ROI = (355, 575, 300, 95)
GUI_BOX3_ROI = (700, 575, 300, 95)

GUI_BTN1_ROI = (1080, 630, 145, 68)
GUI_BTN2_ROI = (1080, 530, 145, 68)

VALIDATION_GUI_LEFT_IMAGE_ROI = (20, 20, 640, 480)
VALIDATION_GUI_RIGHT_IMAGE_ROI = (700, 20, 560, 480)
VALIDATION_GUI_BTN1_ROI = (1080, 550, 145, 120)

GUI_SIZE: Tuple[int, int] = gui_active_background_image.shape[:2][::-1]

GRID_SIZE: int = 2
# dimension in meters of each cell
RESOLUTION: float = 0.1
# how often (number of cells) do we draw a tick with the value
TICKS_EVERY: int = 1
# everything here is calibrated for a 1000x1000 image
SCALE: int = 1000
S_PADDING: int = 60
S_GRID_THICKNESS: int = 1
S_FONT_SIZE: int = 2
S_FONT_THICKNESS: int = 2


def draw_corners(image: BGRImage, board: CalibrationBoard, corners: List[Pixel]) -> BGRImage:
    """
    Draws the given corners on the given image.

    Note:
        The provided image should be rectified.

    Args:
        image (np.ndarray):         A color (3-channel) OpenCV image
        board (dt_computer_vision.camera.calibration.extrinsics.boards.CalibrationBoard):
                                    The calibration board to look for
        corners: (List[Pixel]): Corners as returned by
        ``dt_computer_vision.camera.calibration.extrinsics.chessboard.find_corners``

    Returns:
        np.ndarray: The estimated homography

    Raises:
        RuntimeError: If no corners were found in image, or the corners couldn't be rearranged

    """
    grayscale = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    corners_num = (board.columns - 1, board.rows - 1)
    # opencv wants corners in an array of shape [N, 1, 2]
    corners = np.expand_dims([c.as_array() for c in corners], axis=1)
    # draw chessboard on image
    return cv2.drawChessboardCorners(grayscale, corners_num, corners, True)


def draw_gui(active: bool, left: Optional[BGRImage], right: Optional[BGRImage], corners: int,
             board: CalibrationBoard, current_error: Optional[float], best_error: Optional[float]) \
        -> BGRImage:
    # start with a fresh copy of the background
    image = (gui_active_background_image if active else gui_inactive_background_image).copy()
    # draw left image
    if left is not None:
        x, y, w, h = GUI_LEFT_IMAGE_ROI
        image[y:y+h, x:x+w] = left
    # draw right image
    if right is not None:
        x, y, w, h = GUI_RIGHT_IMAGE_ROI
        image[y:y+h, x:x+w] = right
    # draw number of corners detected
    tot_corners: int = (board.columns - 1) * (board.rows - 1)
    text = {
        "text": f"{corners}/{tot_corners}",
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 2,
        "thickness": 2
    }
    text_width, text_height = cv2.getTextSize(**text)[0]
    x, y, w, h = GUI_BOX1_ROI
    position = (x + int(w / 2) - int(text_width / 2), y + int(h / 2) + int(text_height / 2))
    cv2.putText(image, org=position, color=(200, 200, 200), **text)
    # write error
    text = {
        "text": f"{current_error * 100:.2f}cm" if current_error is not None else "Infty",
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 2,
        "thickness": 2
    }
    text_width, text_height = cv2.getTextSize(**text)[0]
    x, y, w, h = GUI_BOX2_ROI
    position = (x + int(w / 2) - int(text_width / 2), y + int(h / 2) + int(text_height / 2))
    cv2.putText(image, org=position, color=(200, 200, 200), **text)
    # write best error so far
    text = {
        "text": f"{best_error * 100:.2f}cm" if best_error is not None else "Infty",
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 2,
        "thickness": 2
    }
    text_width, text_height = cv2.getTextSize(**text)[0]
    x, y, w, h = GUI_BOX3_ROI
    position = (x + int(w / 2) - int(text_width / 2), y + int(h / 2) + int(text_height / 2))
    cv2.putText(image, org=position, color=(200, 200, 200), **text)
    # ---
    return image


def top_view_projected_corners(
        corners: List[GroundPoint],
        errors: List[float],
        size: Tuple[int, int],
        background_image: Optional[np.ndarray] = None,
        # default values
        grid_size: Union[int, Tuple[int, int]] = GRID_SIZE,
        scale: int = SCALE,
        s_padding: int = S_PADDING,
        resolution: Union[float, Tuple[float, float]] = RESOLUTION,
        start_y: float = 0.0
):
    """
    Generates a debug image with all the detected corners projected onto the gound plane and
    plotted with respect to the robot's origin.

    Args:
        corners (List[GroundPoint]):    True corners computed from knowledge of the board and robot location
        errors (List[float]):           Error in meters for each reprojected ground point
        size (Tuple[int, int]):         Size of the image to draw (width, height)
        background_image (np.ndarray):  Optional background image to draw on
        grid_size (Union[int, Tuple[int, int]]):    Number of rows and columns to show in the grid
        scale (int):                    Scale of the image
        s_padding (int):                Amount of padding (scaled) to leave around the grid plot
        resolution (Union[float, Tuple[float, float]]):
                                        Resolution of the grid (size of each row/column in meters)
        start_y (float):                Offset of the y-axis (the smallest value along the y-axis to show)

    Returns:
        np.ndarray:                    OpenCV image drawn

    """
    background_image = background_image if background_image is not None else draw_grid_image(
        size,
        grid_size=grid_size,
        scale=scale,
        s_padding=s_padding,
        resolution=resolution,
        start_y=start_y,
        s_grid_thickness=S_GRID_THICKNESS,
        s_font_size=S_FONT_SIZE,
        s_font_thickness=S_FONT_THICKNESS,
        ticks_every=TICKS_EVERY,
    )

    size_x, size_y = size

    # if grid_size is an integer, it means that the grid is square
    grid_size = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)

    # if resolution is an integer, it means that the grid is linear
    resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)

    # unpack
    resolution_x, resolution_y = resolution

    s = max(size_x, size_y) / scale
    padding = int(s_padding * s)

    grid_size_x, grid_size_y = grid_size

    half_grid_size_horizontal = int(grid_size_x / 2)
    cell_size_x = int((size_x - 3 * padding) / grid_size_x)
    cell_size_y = int((size_y - 3 * padding) / grid_size_y)
    origin_x, origin_y = 2 * padding + half_grid_size_horizontal * cell_size_x, size_y - 2 * padding

    image = background_image.copy()

    # plot known corners
    for i, corner in enumerate(corners):
        # draw point
        cv2.circle(
            image,
            center=(
                origin_x + int((-corner.y / resolution_x) * cell_size_x),
                origin_y - int(((corner.x - start_y) / resolution_y) * cell_size_y)
            ),
            radius=2,
            color=(30, 30, 30),
            thickness=-1
        )
        error: float = errors[i]
        # draw error as a circle
        cv2.circle(
            image,
            center=(
                origin_x + int((-corner.y / resolution_x) * cell_size_x),
                origin_y - int(((corner.x - start_y) / resolution_y) * cell_size_y)
            ),
            radius=int((error / resolution_x) * cell_size_x),
            color=(0, 0, 255),
            thickness=1
        )
    # ---
    return image


def draw_validation_gui(left: Optional[BGRImage] = None, right: Optional[BGRImage] = None) -> BGRImage:
    # start with a fresh copy of the background
    image = validation_gui_active_background_image.copy()
    # draw left image
    if left is not None:
        x, y, w, h = VALIDATION_GUI_LEFT_IMAGE_ROI
        image[y:y+h, x:x+w] = left
    # draw right image
    if right is not None:
        x, y, w, h = VALIDATION_GUI_RIGHT_IMAGE_ROI
        image[y:y+h, x:x+w] = right
    # ---
    return image
