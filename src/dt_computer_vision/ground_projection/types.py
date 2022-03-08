import dataclasses
from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import invert_map, ensure_ndarray

BGRImage = np.ndarray


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
        return f"P({self.x}, {self.y})"


class Pixel(Point):

    def as_integers(self) -> np.ndarray:
        return np.array([int(self.x), int(self.y)], dtype=int)


class NormalizedImagePoint(Point):
    pass


class GroundPoint(Point):
    pass


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

    def rectify_point(self, point: NormalizedImagePoint) -> NormalizedImagePoint:
        """
        Args:
            point (:py:class:`NormalizedImagePoint`): A point in normalized image coordinates.

        Applies the rectification specified by camera parameters
        :math:`K` and and :math:`D` to point (u, v) and returns the
        pixel coordinates of the rectified point.
        """

        # TODO: use numpy instead
        # src = cv2.CreateMat(1, 2, cv2.CV_64FC1)
        # cv2.SetData(src, array.array('d', list(uv_raw)), 8 * 2)
        # src = cv2.Reshape(src, 2)
        # dst = cv2.CloneMat(src)

        src = point.as_array().reshape((1, 1, 2))
        dst = cv2.undistortPoints(src,
                                  self.camera.K,
                                  self.camera.D,
                                  R=self.camera.R,
                                  P=self.camera.P)
        return NormalizedImagePoint(*dst[0, 0])

    def rectify(self, image: BGRImage, interpolation=cv2.INTER_NEAREST):
        """
        Undistorts an image.
        While cv2.INTER_NEAREST is faster, use cv2.INTER_CUBIC for higher quality.
        """
        if not self._rectify_inited:
            self._init_rectify_maps()
        # TODO: this "dst" might not be needed
        cv_image_rectified = np.empty_like(image)
        res = cv2.remap(image, self.mapx, self.mapy, interpolation, cv_image_rectified)
        return res

    def distort(self, rectified: BGRImage, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
        # initialize forward map (if not done yet)
        if not self._rectify_inited:
            self._init_rectify_maps()
        # initialize inverse map (if not done yet)
        if not self._distort_inited:
            self.rmapx, self.rmapy = invert_map(self.mapx, self.mapy)
            self._distort_inited = True
        # distort rectified image
        # TODO: this "dst" might not be needed
        distorted = np.zeros(np.shape(rectified))
        res = cv2.remap(rectified, self.rmapx, self.rmapy, interpolation, distorted)
        # ---
        return res

    def rectify_full(self,
                     image: BGRImage,
                     interpolation: int = cv2.INTER_NEAREST,
                     ratio: float = 1.0):
        """
        Undistort an image by maintaining the proportions.
        While cv2.INTER_NEAREST is faster, use cv2.INTER_CUBIC for higher quality.
        Returns the new camera matrix as well.

        Args:
            image:  np.ndarray      Distorted image to rectify
            interpolation:  int     Interpolation strategy used to scale the image
            ratio:  float           Scaling factor
        """
        W = int(self.camera.width * ratio)
        H = int(self.camera.height * ratio)

        print(f"K: {self.camera.K}")
        print(f"P: {self.camera.P}")

        # Use the same camera matrix
        # TODO: this should be P instead of K
        new_camera_matrix = self.camera.K.copy()

        # TODO: this is wrong, it assumes that the principal point is at (W/2, H/2)
        #       these should be scaled as well, not replaced
        new_camera_matrix[0, 2] = W / 2
        new_camera_matrix[1, 2] = H / 2

        print(f"new_camera_matrix: {new_camera_matrix}")

        mapx, mapy = cv2.initUndistortRectifyMap(
            self.camera.K, self.camera.D, self.camera.R, new_camera_matrix, (W, H), cv2.CV_32FC1
        )

        # TODO: we might not need to pass this empty stuff, test it out without it
        cv_image_rectified = np.empty_like(image)
        res = cv2.remap(image, mapx, mapy, interpolation, cv_image_rectified)
        return new_camera_matrix, res


@dataclasses.dataclass
class CameraModel:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    P: np.ndarray
    H: Optional[np.ndarray] = None

    rectifier: Rectifier = dataclasses.field(init=False)

    def __post_init__(self):
        self.K = ensure_ndarray(self.K)
        self.D = ensure_ndarray(self.D)
        self.R = ensure_ndarray(self.R)
        self.P = ensure_ndarray(self.P)
        self.H = None if self.H is None else ensure_ndarray(self.H)
        self.rectifier = Rectifier(self)

    def get_shape(self) -> Tuple[int, int]:
        """returns (height, width) of image"""
        return self.height, self.width

    def vector2pixel(self, vec: NormalizedImagePoint) -> Pixel:
        """
        Converts a ``[0,1] X [0,1]`` representation to ``[0, W] X [0, H]``
        (from normalized to image coordinates).

        Args:
            vec (:py:class:`Point`): A :py:class:`Point` object in normalized coordinates.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object in image coordinates.

        """
        x = self.width * vec.x
        y = self.height * vec.y
        return Pixel(x, y)

    def pixel2vector(self, pixel: Pixel) -> NormalizedImagePoint:
        """
        Converts a ``[0,W] X [0,H]`` representation to ``[0, 1] X [0, 1]``
        (from image to normalized coordinates).

        Args:
            pixel (:py:class:`Point`): A :py:class:`Point` object in image coordinates.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object in normalized coordinates.

        """
        x = pixel.x / self.width
        y = pixel.y / self.height
        return NormalizedImagePoint(x, y)
