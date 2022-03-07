import array
import dataclasses
from typing import Optional

import cv2
import numpy as np

from .ground_projection_geometry import Point
from .utils import invert_map

BGRImage = np.ndarray


@dataclasses.dataclass
class CameraModel:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    P: np.ndarray

    def rectify_point(self, uv_raw):
        """
        :param uv_raw:    pixel coordinates
        :type uv_raw:     (u, v)
        Applies the rectification specified by camera parameters
        :math:`K` and and :math:`D` to point (u, v) and returns the
        pixel coordinates of the rectified point.

        NOTE: This is based on
              https://github.com/strawlab/vision_opencv/blob/master/image_geometry/src/image_geometry/cameramodels.py
        """
        src = cv2.CreateMat(1, 2, cv2.CV_64FC1)
        cv2.SetData(src, array.array('d', list(uv_raw)), 8 * 2)
        src = cv2.Reshape(src, 2)
        dst = cv2.CloneMat(src)
        cv2.UndistortPoints(src, dst, self.K, self.D, self.R, self.P)
        return dst[0, 0]


class Rectify:
    """
    Handles the Rectification operations.

    """

    def __init__(self,
                 width: int,
                 height: int,
                 K: np.ndarray,
                 D: np.ndarray,
                 R: np.ndarray,
                 P: np.ndarray,
                 ):
        self.camera: CameraModel = CameraModel(width, height, K, D, R, P)
        self._rectify_inited: bool = False
        self._distort_inited: bool = False
        # forward maps (lazily initialized)
        self.mapx: Optional[np.ndarray] = None
        self.mapy: Optional[np.ndarray] = None
        # inverse maps (lazily initialized)
        self.rmapx: Optional[np.ndarray] = None
        self.rmapy: Optional[np.ndarray] = None

    def rectify_point(self, pixel: Point) -> Point:
        p = (pixel.x, pixel.y)
        return Point(*list(self.camera.rectify_point(p)))

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

    def rectify(self, cv_image_raw: np.ndarray, interpolation=cv2.INTER_NEAREST):
        """
        Undistorts an image.
        While cv2.INTER_NEAREST is faster, use cv2.INTER_CUBIC for higher quality.
        """
        if not self._rectify_inited:
            self._init_rectify_maps()
        cv_image_rectified = np.empty_like(cv_image_raw)
        res = cv2.remap(cv_image_raw, self.mapx, self.mapy, interpolation, cv_image_rectified)
        return res

    def distort(self, rectified: BGRImage) -> np.ndarray:
        # initialize forward map (if not done yet)
        if not self._rectify_inited:
            self._init_rectify_maps()
        # initialize inverse map (if not done yet)
        if not self._distort_inited:
            self.rmapx, self.rmapy = invert_map(self.mapx, self.mapy)
            self._distort_inited = True
        # distort rectified image
        distorted = np.zeros(np.shape(rectified))
        res = cv2.remap(rectified, self.rmapx, self.rmapy, cv2.INTER_NEAREST, distorted)
        # ---
        return res

    def rectify_full(self, image: BGRImage, interpolation: int = cv2.INTER_NEAREST,
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
