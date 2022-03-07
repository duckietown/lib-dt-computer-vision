import dataclasses
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass
class Point:
    """
    A generic (up to 3D) point.
    It is used for 2D, 3D, pixel points.

    """
    x: float
    y: float
    z: Optional[float] = None

    def __repr__(self):
        return f"P({self.x}, {self.y}, {self.z})"


ImageSpacePixel = Point
ImageSpaceNormalizedPoint = Point
GroundPoint = Point


class GroundProjectionGeometry:
    """
    Handles the Ground Projection operations.

    Note:
        All pixel and image operations in this class assume that the pixels and images are *already
        rectified*. If
        unrectified pixels or images are supplied, the outputs of these operations will be incorrect.

    Args:
        width (``int``): Width of the rectified image
        height (``int``): Height of the rectified image
        homography (``np.ndarray``): The 3x3 Homography matrix

    """

    def __init__(self, width: int, height: int, homography: np.ndarray):
        self.width: int = width
        self.height: int = height

        H = homography.copy()
        if H.shape != (3, 3):
            H = H.reshape((3, 3))

        self.H: np.ndarray = H
        self.Hinv: np.ndarray = np.linalg.inv(self.H)

    def get_shape(self) -> Tuple[int, int]:
        """returns (height, width) of image"""
        return self.height, self.width

    def vector2pixel(self, vec: ImageSpaceNormalizedPoint) -> ImageSpacePixel:
        """
        Converts a ``[0,1] X [0,1]`` representation to ``[0, W] X [0, H]``
        (from normalized to image coordinates).

        Args:
            vec (:py:class:`Point`): A :py:class:`Point` object in normalized coordinates.
            Only the ``x`` and ``y`` values are used.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object in image coordinates.
            Only the ``x`` and ``y`` values are used.

        """
        x = self.width * vec.x
        y = self.height * vec.y
        return ImageSpacePixel(x, y)

    def pixel2vector(self, pixel: ImageSpacePixel) -> ImageSpaceNormalizedPoint:
        """
        Converts a ``[0,W] X [0,H]`` representation to ``[0, 1] X [0, 1]``
        (from image to normalized coordinates).

        Args:
            pixel (:py:class:`Point`): A :py:class:`Point` object in image coordinates.
            Only the ``x`` and ``y`` values are used.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object in normalized coordinates.
            Only the ``x`` and ``y`` values are used.

        """
        x = pixel.x / self.width
        y = pixel.y / self.height
        return ImageSpaceNormalizedPoint(x, y)

    def pixel2ground(self, pixel: ImageSpaceNormalizedPoint) -> GroundPoint:
        """
        Projects a normalized pixel (``[0, 1] X [0, 1]``) to the ground
        plane using the homography matrix.

        Args:
            pixel (:py:class:`Point`): A :py:class:`Point` object in normalized coordinates.
            Only the ``x``and ``y`` values are used.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object on the ground plane.
            Only the ``x`` and ``y`` values are used.

        """
        uv_raw = np.array([pixel.x, pixel.y, 1.0])
        ground_point = np.dot(self.H, uv_raw)
        x = ground_point[0] / ground_point[2]
        y = ground_point[1] / ground_point[2]
        return GroundPoint(x, y, 0.0)

    def vector2ground(self, vec: ImageSpaceNormalizedPoint) -> GroundPoint:
        pixel: ImageSpacePixel = self.vector2pixel(vec)
        return self.pixel2ground(pixel)

    def ground2pixel(self, point: GroundPoint) -> ImageSpaceNormalizedPoint:
        """
        Projects a point on the ground plane to a normalized pixel (``[0, 1] X [0, 1]``) using the
        homography matrix.

        Args:
            point (:py:class:`Point`): A :py:class:`Point` object on the ground plane.
            Only the ``x`` and ``y`` values are used.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object in normalized coordinates.
            Only the ``x`` and ``y`` values are used.

        Raises:
            ValueError: If the input point's ``z`` attribute is non-zero.
            The point must be on the ground (``z=0``).

        """
        if point.z != 0:
            msg = "This method assumes that the point is a ground point (z=0). "
            msg += f"However, the point is ({point.x},{point.y},{point.z})"
            raise ValueError(msg)

        ground_point = np.array([point.x, point.y, 1.0])
        image_point = np.dot(self.Hinv, ground_point)
        image_point = image_point / image_point[2]

        x = image_point[0]
        y = image_point[1]

        return ImageSpaceNormalizedPoint(x, y)
