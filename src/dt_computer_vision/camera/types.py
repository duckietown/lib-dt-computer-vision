import dataclasses
from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import invert_map, ensure_ndarray

BGRImage = np.ndarray
RGBImage = np.ndarray
HSVImage = np.ndarray


@dataclasses.dataclass
class Point:
    """
    A generic 2D point.
    """
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def __repr__(self):
        return f"P({round(self.x, 4)}, {round(self.y, 4)})"


class Pixel(Point):

    def as_integers(self) -> np.ndarray:
        return np.array([int(self.x), int(self.y)], dtype=int)


class NormalizedImagePoint(Point):
    pass


@dataclasses.dataclass
class ResolutionIndependentImagePoint(NormalizedImagePoint):

    def __post_init__(self):
        if self.x < 0 or self.x > 1 or self.y < 0 or self.y > 1:
            raise ValueError("Values of x and y must be within the range [0, 1] for objects of type "
                             "ResolutionIndependentImagePoint.")


class Size(Point):

    def __repr__(self):
        return f"S({round(self.x, 4)}, {round(self.y, 4)})"


@dataclasses.dataclass
class RegionOfInterest:
    """
    A generic 2D region.
    """
    origin: Point
    size: Size


class Rectifier:
    """
    Handles the Rectification operations.

    """

    def __init__(self, camera: 'CameraModel'):
        self.camera: CameraModel = camera
        self._rectify_inited: bool = False
        self._distort_inited: bool = False
        # forward maps (lazily initialized)
        self.mapx: Optional[np.ndarray] = None
        self.mapy: Optional[np.ndarray] = None
        # inverse maps (lazily initialized)
        self.rmapx: Optional[np.ndarray] = None
        self.rmapy: Optional[np.ndarray] = None

    def _init_rectify_maps(self):
        W = self.camera.width
        H = self.camera.height
        mapx = np.ndarray(shape=(H, W, 1), dtype="float32")
        mapy = np.ndarray(shape=(H, W, 1), dtype="float32")
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera.K, self.camera.D, self.camera.R, self.camera.P,
            (W, H), cv2.CV_32FC1, mapx, mapy
        )
        self._rectify_inited = True

    def rectify_pixel(self, point: Pixel) -> Pixel:
        """
        Args:
            point (:py:class:`Pixel`): A point in pixel coordinates.

        Applies the rectification specified by camera parameters
        :math:`K` and and :math:`D` to the given pixel (u, v) and returns the
        pixel coordinates of the rectified point.
        """
        src = point.as_array().reshape((1, 1, 2)).astype(float)
        dst = cv2.undistortPoints(src,
                                  self.camera.K,
                                  self.camera.D,
                                  R=self.camera.R,
                                  P=self.camera.P)
        return Pixel(*dst[0, 0])

    def rectify(self, image: BGRImage, interpolation=cv2.INTER_NEAREST) -> BGRImage:
        """
        Undistorts an image.
        While cv2.INTER_NEAREST is faster, use cv2.INTER_CUBIC for higher quality.
        """
        if not self._rectify_inited:
            self._init_rectify_maps()
        # rectify distorted image
        return cv2.remap(image, self.mapx, self.mapy, interpolation)

    def distort(self, rectified: BGRImage, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
        """
        Distorts an image.
        While cv2.INTER_NEAREST is faster, use cv2.INTER_CUBIC for higher quality.
        """
        # initialize forward map (if not done yet)
        if not self._rectify_inited:
            self._init_rectify_maps()
        # initialize inverse map (if not done yet)
        if not self._distort_inited:
            self.rmapx, self.rmapy = invert_map(self.mapx, self.mapy)
            self._distort_inited = True
        # distort rectified image
        return cv2.remap(rectified, self.rmapx, self.rmapy, interpolation)


@dataclasses.dataclass
class CameraModel:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    P: np.ndarray
    R: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None

    rectifier: Rectifier = dataclasses.field(init=False)

    def __post_init__(self):
        self.K = ensure_ndarray(self.K)
        self.D = ensure_ndarray(self.D)
        self.P = ensure_ndarray(self.P)
        self.R = np.eye(3) if self.R is None else ensure_ndarray(self.R)
        self.H = None if self.H is None else ensure_ndarray(self.H)
        self.rectifier = Rectifier(self)

    @property
    def fx(self) -> float:
        return self.K[0, 0]

    @property
    def fy(self) -> float:
        return self.K[1, 1]

    @property
    def cx(self) -> float:
        return self.K[0, 2]

    @property
    def cy(self) -> float:
        return self.K[1, 2]

    @property
    def principal_point(self) -> Pixel:
        return Pixel(self.cx, self.cy)

    def get_shape(self) -> Tuple[int, int]:
        """returns (height, width) of image"""
        return self.height, self.width

    def vector2pixel(self, vec: NormalizedImagePoint) -> Pixel:
        """
        Converts a ``[-1,1] X [-1,1]`` representation to ``[0, W] X [0, H]``
        (from normalized to image coordinates).

        Args:
            vec (:py:class:`Point`): A :py:class:`Point` object in normalized coordinates.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object in image coordinates.

        """
        x = self.cx + vec.x * self.fx
        y = self.cy + vec.y * self.fy
        return Pixel(x, y)

    def pixel2vector(self, pixel: Pixel) -> NormalizedImagePoint:
        """
        Converts a ``[0,W] X [0,H]`` representation to ``[-1, 1] X [-1, 1]``
        (from image to normalized coordinates).

        Args:
            pixel (:py:class:`Point`): A :py:class:`Point` object in image coordinates.

        Returns:
            :py:class:`NormalizedImagePoint` : Corresponding normalized coordinates.

        """
        x = (pixel.x - self.cx) / self.fx
        y = (pixel.y - self.cy) / self.fy
        return NormalizedImagePoint(x, y)

    def cropped(self, top: int = 0, right: int = 0, bottom: int = 0, left: int = 0) -> 'CameraModel':
        K1: np.ndarray = np.array([
            [self.fx,        0,   self.cx - left],
            [0,        self.fy,    self.cy - top],
            [0,              0,                1],
        ])
        shape: Tuple[int, int] = (self.width - left - right, self.height - top - bottom)
        K1rect, _ = cv2.getOptimalNewCameraMatrix(K1, self.D, shape, 0, shape)
        P1rect = np.hstack((K1rect, [[0], [0], [1]]))
        return CameraModel(
            width=shape[0],
            height=shape[1],
            K=K1,
            D=self.D,
            # TODO: we are not testing this rigorously (e.g., unit tests)
            P=P1rect,
            H=self.H
        )

    def scaled(self, s: float) -> 'CameraModel':
        K1: np.ndarray = self.K * s
        K1[2, 2] = 1.0
        w, h = round(self.width * s), round(self.height * s)
        shape: Tuple[int, int] = (w, h)
        K1rect, _ = cv2.getOptimalNewCameraMatrix(K1, self.D, shape, 0, shape)
        P1rect = np.hstack((K1rect, [[0], [0], [1]]))
        return CameraModel(
            width=w,
            height=h,
            K=K1,
            D=self.D,
            # TODO: we are not testing this rigorously (e.g., unit tests)
            P=P1rect
        )

    def pixel2independent(self, pixel: Pixel) -> ResolutionIndependentImagePoint:
        """
        Converts a ``[0,W] X [0,H]`` representation to ``[0, 1] X [0, 1]``
        (from image to resolution-independent coordinates).

        Args:
            pixel (:py:class:`Pixel`): A :py:class:`Pixel` object in image coordinates.

        Returns:
            :py:class:`ResolutionIndependentImagePoint` : Corresponding resolution-independent coordinates.

        """
        x = pixel.x / (self.width - 1)
        y = pixel.y / (self.height - 1)
        return ResolutionIndependentImagePoint(x, y)

    def homography_vector2independent(self) -> np.ndarray:
        """
        Homography converting normalized coordinates ``[-∞,+∞] X [-∞,+∞]`` to resolution-independent
        coordinates ``[0, 1] X [0, 1]``.

        Returns:
            :py:class:`np.ndarray` : A 3-by-3 matrix implementing the coordinate transformation operation.
        """
        # top-left and bottom-right corner of the image in normalized coordinates
        tl: NormalizedImagePoint = self.pixel2vector(Pixel(0, 0))
        br: NormalizedImagePoint = self.pixel2vector(Pixel(self.width - 1, self.height - 1))
        # scaling factors along X and Y
        sx = 1.0 / (br.x - tl.x)
        sy = 1.0 / (br.y - tl.y)
        # ---
        return np.array([
            [sx, 0, -tl.x * sx],
            [0, sy, -tl.y * sy],
            [0,  0,          1]
        ])

    def homography_independent2vector(self) -> np.ndarray:
        """
        Homography converting resolution-independent coordinates ``[0, 1] X [0, 1]`` to normalized
        coordinates ``[-∞,+∞] X [-∞,+∞]``.

        Returns:
            :py:class:`np.ndarray` : A 3-by-3 matrix implementing the coordinate transformation operation.
        """
        return np.linalg.inv(self.homography_vector2independent())
