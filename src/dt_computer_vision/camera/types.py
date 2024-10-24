import dataclasses
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from dt_computer_vision.camera.homography import Homography

from .utils import invert_map, ensure_ndarray
import yaml

BGRImage = np.ndarray
RGBImage = np.ndarray
HSVImage = np.ndarray
GRAYImage = np.ndarray


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
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __truediv__(self, scalar: float) -> 'Point':
        return Point(self.x / scalar, self.y / scalar)
    
    def __mul__(self, scalar: float) -> 'Pixel':
        return Pixel(self.x * scalar, self.y * scalar)



class Pixel(Point):

    def as_integers(self) -> np.ndarray:
        return np.array([int(self.x), int(self.y)], dtype=int)

    def __repr__(self):
        return f"P({round(self.x, 4)}, {round(self.y, 4)})"

    def __truediv__(self, scalar: float) -> 'Pixel':
        return Pixel(self.x / scalar, self.y / scalar)
    
    def __sub__(self, other: 'Pixel') -> 'Pixel':
        return Pixel(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Pixel':
        return Pixel(self.x * scalar, self.y * scalar)

class NormalizedImagePoint(Point):
    """
    Image point normalized using the camera intrinsics.
    The normalized coordinates are defined as follows:
        u=(x-cx)/fx, v=(y-cy)/fy
    """
    pass


@dataclasses.dataclass
class ResolutionIndependentImagePoint(Point):

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
        # noinspection PyUnresolvedReferences
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera.K, self.camera.D, self.camera.R, self.camera.P, # type: ignore
            (W, H), cv2.CV_32FC1, mapx, mapy
        ) # type: ignore
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
    """
    Class describing a camera model.
    
    The following conventions for coordinates are used (W: image width, H: image height):
    
    - resolution-dependent: [0, W] x [0, H]      (or "image coordinates")
    - resolution-independent: [0, 1] x [0, 1]
    - normalized coordinates: [-inf, +inf] x [-inf, +inf]
    """
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    P: np.ndarray
    R: Optional[np.ndarray] = dataclasses.field(default_factory=lambda: np.eye(3))
    H: Optional[np.ndarray] = None

    rectifier: Rectifier = dataclasses.field(init=False)

    def __post_init__(self):
        self.K = ensure_ndarray(self.K, shape=(3, 3))
        self.D = ensure_ndarray(self.D, shape=(5,))
        self.P = ensure_ndarray(self.P, shape=(3, 4))
        self.R = ensure_ndarray(self.R, shape=(3, 3)) if self.R is not None else np.eye(3)
        self.H = None if self.H is None else ensure_ndarray(self.H)
        self._H_inv = None if self.H is None else np.linalg.inv(self.H)
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
        Converts a ``[-inf,+inf] X [-inf,+inf]`` representation to ``[0, W] X [0, H]``
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
        Converts a ``[0,W] X [0,H]`` representation to ``[-inf,+inf] X [-inf,+inf]``
        (from resolution-dependent to normalized coordinates).

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
        
        scaling_matrix = np.eye(3)
        scaling_matrix[0, 0] = s
        scaling_matrix[1, 1] = s
        
        
        return CameraModel(
            width=w,
            height=h,
            K=K1,
            D=self.D,
            # TODO: we are not testing this rigorously (e.g., unit tests)
            P=P1rect,
            H=self.H @ scaling_matrix @ self._H_inv if self.H is not None else None
        )
        
    def downsample(self, binning: int) -> 'CameraModel':
        return self.scaled(1.0 / binning)
        
    def pixel2independent(self, pixel: Pixel) -> ResolutionIndependentImagePoint:
        """
        Converts a ``[0,W] X [0,H]`` representation to ``[0, 1] X [0, 1]``
        (from resolution-dependent to resolution-independent coordinates).

        Args:
            pixel (:py:class:`Pixel`): A :py:class:`Pixel` object in image coordinates.

        Returns:
            :py:class:`ResolutionIndependentImagePoint` : Corresponding resolution-independent coordinates.

        """
        x = pixel.x / (self.width - 1)
        y = pixel.y / (self.height - 1)
        return ResolutionIndependentImagePoint(x, y)

    def independent2pixel(self, riip: ResolutionIndependentImagePoint) -> Pixel:
        x = riip.x * (self.width - 1)
        y = riip.y * (self.height - 1)
        return Pixel(x,y)

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
        
    def homography_pixel2vector(self) -> np.ndarray:
        """
        Homography converting a pixel coordinates ``[0,W] X [0,H]`` representation to normalized coordinates
        ``[-∞,+∞] X [-∞,+∞]``

        Returns:
            :py:class:`np.ndarray` : A 3-by-3 matrix implementing the coordinate transformation operation.
        """        
        return np.array([
            [1/self.fx,         0, -self.cx/self.fx],
            [0,         1/self.fy, -self.cy/self.fy],
            [0,                 0,                1]
        ])

    def homography_vector2pixel(self) -> np.ndarray:
        """
        Homography converting normalized coordinates ``[-∞,+∞] X [-∞,+∞]`` to pixel coordinates ``[0,W] X [0,H]``

        Returns:
            :py:class:`np.ndarray
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def homography_independent2vector(self) -> np.ndarray:
        """
        Homography converting resolution-independent coordinates ``[0, 1] X [0, 1]`` to normalized
        coordinates ``[-∞,+∞] X [-∞,+∞]``.

        Returns:
            :py:class:`np.ndarray` : A 3-by-3 matrix implementing the coordinate transformation operation.
        """
        return np.linalg.inv(self.homography_vector2independent())

    def to_native_objects(self) -> dict:
        return {
            'width': self.width,
            'height': self.height,
            'K': self.K.tolist(),
            'D': self.D.tolist(),
            'P': self.P.tolist(),
            'R': self.R.tolist() if self.R is not None else None,
            'H': self.H.tolist() if self.H is not None else None
        }

    @classmethod
    def from_native_objects(cls, data) -> 'CameraModel':
        return CameraModel(
            width=data['width'],
            height=data['height'],
            K=np.array(data['K']),
            D=np.array(data['D']),
            P=np.array(data['P']),
            R=np.array(data['R']) if 'R' in data and data['R'] is not None else None,
            H=np.array(data['H']) if data['H'] is not None else None
        )
        
    @classmethod
    def from_ros_calibration(cls, filestream, alpha = 0.0):
        """
        Import the camera calibration parameters from a ROS calibration file.
        """
        data = yaml.safe_load(filestream)
        K = np.array(data['camera_matrix']['data']).reshape(3, 3)
        D = np.array(data['distortion_coefficients']['data'])
        P = np.array(data['projection_matrix']['data']).reshape(3, 4)
        R = np.array(data['rectification_matrix']['data']).reshape(3, 3)
        width = data['image_width']
        height = data['image_height']
        
        if alpha > 0.0:
            # Compute the new camera matrix
            K, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), alpha)
            P = np.hstack((K, [[0], [0], [0]]))
    
        return CameraModel(width, height, K, D, P, R)
    
    def to_ros_calibration(self, filestream, camera_name = 'camera'):
        """
        Export the camera calibration parameters to a ROS calibration file.
        """
        data = {
            'image_width': self.width,
            'image_height': self.height,
            'camera_name': camera_name,
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': self.K.flatten().tolist()
            },
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': self.D.flatten().tolist()
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': self.R.flatten().tolist()
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': self.P.flatten().tolist()
            }
        }
        
        yaml.dump(data, filestream)
