import os.path
import typing
from datetime import datetime
from typing import List, Tuple, Optional, Union

import numpy as np
import requests
import yaml

import dt_computer_vision.camera


if typing.TYPE_CHECKING:
    from dt_computer_vision.camera.types import CameraModel
    import dt_computer_vision


class Homography(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def inverse(self) -> 'Homography':
        """
        Inverse matrix, normalized to have the [2,2] element equal to 1.
        
        H_inv = inv(H)/inv(H)[2,2]
        """
        H_inv = np.linalg.inv(self)
        H_inv /= H_inv[2, 2]
        
        # Use __new__ to create an instance of Homography
        return Homography(H_inv)

def pose_from_homography(H: Homography ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the pose (translation and rotation) from a homography matrix.

    Parameters:
        H (Homography): The homography matrix (as returned by 
        `dt_computer_vision.camera.calibration.extrinsics.ransac.estimate_homography`).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the translation vector and rotation matrix.

    """

    H = H.inverse

    norm = np.linalg.norm(H[:, 0])
    H /= norm

    c1 = H[:, 0]
    c2 = H[:, 1]
    c3 = np.cross(c1, c2)

    tvec = H[:, 2]
    R = np.zeros((3, 3))
    R[:, 0] = c1
    R[:, 1] = c2
    R[:, 2] = c3

    # Orthogonalize rotation matrix using SVD
    U, _, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[2, 0] *= -1
        Vt[2, 1] *= -1
        Vt[2, 2] *= -1
        R = np.dot(U, Vt)
    return tvec, R


def __compute_homography_from_poses(R1, tvec1, R2, tvec2, d_inv, normal):
    homography = np.dot(R2, R1.T) + d_inv * (np.dot(-R2, np.dot(R1.T, tvec1)) + tvec2) * normal.T
    return homography


def interpolate_homography(H : Homography, tvec : np.ndarray, R : np.ndarray, camera : 'dt_computer_vision.camera.CameraModel') -> Homography:
    """Compute a new homography from a given homography and a goal pose for the camera, expressed in the board reference frame.
    
    Args:
        H (Homography): The given homography.
        tvec (np.ndarray): The goal pose for the camera, expressed in the board reference frame.
        R (np.ndarray): The rotation matrix of the goal pose.
        camera ('dt_computer_vision.camera.CameraModel'): The camera model.
    Returns:
        Homography: The computed new homography.
    """
    
    tvec1, R1 = pose_from_homography(H)
    tvec1 = tvec1.reshape(3, 1)
    
    assert tvec.shape == (3, 1)
    assert R.shape == (3, 3)
    assert R1.shape == (3, 3)

    # Compute plane normal at camera pose 1
    normal = np.array([[0], [0], [1]])
    normal1 : np.ndarray = np.dot(R1, normal)

    # Compute plane distance to the camera frame 1
    origin = np.zeros((3, 1))
    origin1 = np.dot(R1, origin) + tvec1
    d_inv1 = 1.0 / np.dot(normal1.T, origin1)
    
    homography_euclidean = __compute_homography_from_poses(R1, tvec1, R, tvec, d_inv1, normal1)

    # Compute the full homography using the camera matrix
    homography = np.dot(np.dot(camera.K, homography_euclidean), np.linalg.inv(camera.K))

    # Normalize the homography matrices
    homography /= homography[2, 2]
    
    return Homography(homography)


class ResolutionDependentHomography(Homography):

    # noinspection PyMissingConstructor
    def __init__(self):
        raise RuntimeError("Do not instantiate the class 'ResolutionDependentHomography' directly, use the "
                           "method ResolutionDependentHomography.read() instead.")

    @classmethod
    def read(cls, value: np.ndarray):
        return value.view(ResolutionDependentHomography)

    def camera_independent(
            self, camera: 'CameraModel'
    ) -> "ResolutionIndependentHomography":
        Hi2v: Homography = camera.homography_independent2vector()
        Hv2x: Homography = self
        Hi2x: ResolutionIndependentHomography = ResolutionIndependentHomography.read(
            np.dot(Hi2v, Hv2x)
        )
        return Hi2x

class ResolutionIndependentHomography(Homography):

    # noinspection PyMissingConstructor
    def __init__(self, shape):
        raise RuntimeError("Do not instantiate the class 'ResolutionIndependentHomography' directly, use the "
                           "method ResolutionIndependentHomography.read() instead.")

    @classmethod
    def read(cls, value: np.ndarray):
        return value.view(ResolutionIndependentHomography)

    def camera_specific(
            self, camera: 'CameraModel'
    ) -> ResolutionDependentHomography:
        Hv2i: Homography = camera.homography_vector2independent()
        Hi2x: Homography = self
        Hv2x: ResolutionDependentHomography = ResolutionDependentHomography.read(
            np.dot(Hv2i, Hi2x)
        )
        return Hv2x


class HomographyToolkit:
    FILE_CONTENT = """
# IMPORTANT: Do not use this homography directly with your images. The homography in this file is stored
#            in a resolution-independent format. Use the proper APIs to read this file and extract a 
#            resolution-dependent homography that is specific to your camera. 
# ------------------------------------------------------------------------------------------------------

{yaml}
"""

    DEFAULT_VERSION: str = "2"
    SUPPORTED_VERSIONS: List[str] = [DEFAULT_VERSION]

    @staticmethod
    def load_from_disk(
            fpath: str,
            return_date: bool = False
    ) -> Union[Homography, Tuple[Homography, Optional[datetime]]]:
        fpath = os.path.abspath(fpath)
        # make sure the file exists
        if not os.path.exists(fpath) or not os.path.isfile(fpath):
            raise FileNotFoundError(fpath)
        # read file
        with open(fpath, "rt") as fin:
            fcontent: dict = yaml.safe_load(fin)
        # interpret file content
        # - check version
        if "version" not in fcontent:
            raise ValueError(
                f"Invalid homography file '{fpath}'. The required root key 'version' was not found."
            )
        version: str = fcontent["version"].strip()
        if version not in HomographyToolkit.SUPPORTED_VERSIONS:
            raise ValueError(
                f"The homography file '{fpath}' uses the version '{version}' which is not supported. "
                f"Supported versions are: {HomographyToolkit.SUPPORTED_VERSIONS}"
            )
        # - check homography
        if "homography" not in fcontent:
            raise ValueError(
                f"Invalid homography file '{fpath}'. The required root key 'homography' was not found."
            )
        Hraw: List[float] = fcontent["homography"]
        if not isinstance(Hraw, list):
            raise ValueError(
                f"Invalid homography file '{fpath}'. The root key 'homography' must contain a list of floats."
            )
        # read homography
        H: Homography = Homography(
            np.array(fcontent["homography"]).reshape((3, 3))
        )
        # load time
        date: Optional[datetime] = None
        if "date" in fcontent:
            date = datetime.fromisoformat(fcontent["date"])
        if return_date:
            return H, date
        return H

    @staticmethod
    def save_to_disk(
            H: Homography,
            fpath: str,
            exist_ok: bool = False,
            date: datetime = None,
    ):
        if not isinstance(H, Homography):
            raise ValueError()

        fpath = os.path.abspath(fpath)
        # make sure the file does not exists
        if os.path.exists(fpath):
            if not os.path.isfile(fpath):
                raise ValueError(f"The path '{fpath}' exists and is not a file.")
            if not exist_ok:
                raise FileExistsError(
                    f"The file '{fpath}' already exists and 'exist_ok' is set to False."
                )
        # create file content
        fcontent: str = HomographyToolkit.FILE_CONTENT.format(
            yaml=yaml.safe_dump(
                {
                    "version": HomographyToolkit.DEFAULT_VERSION,
                    "date": (date or datetime.today()).isoformat(),
                    "homography": H.tolist(),
                },
                sort_keys=False
            )
        )
        # write file
        with open(fpath, "wt") as fout:
            fout.write(fcontent)

    @staticmethod
    def save_to_http(
            H: ResolutionIndependentHomography,
            url: str,
            date: datetime = None,
    ):
        """
        Uploads a resolution-independent homography to a remote server using a HTTP POST request.
        :param H: the homography to upload
        :param url: the URL to upload the homography to
        :param date: the date to associate with the homography (if None, the current date is used)
        
        The format of the yaml file is:
        
        ```
        version: "2"
        date: "2021-01-01T12:00:00"
        homography: [1, 0, 0, 0, 1, 0, 0, 0, 1]
        ```
        """
        if not isinstance(H, ResolutionIndependentHomography):
            raise ValueError(
                "Only resolution-independent homographies can be stored to disk. Please, "
                "convert your homography to a ResolutionIndependentHomography first."
            )
        # create file content
        fcontent: str = HomographyToolkit.FILE_CONTENT.format(
            yaml=yaml.safe_dump(
                {
                    "version": HomographyToolkit.DEFAULT_VERSION,
                    "date": (date or datetime.today()).isoformat(),
                    "homography": H.tolist(),
                },
                sort_keys=False
            )
        )
        # send POST request
        response = requests.post(url, data=fcontent)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to upload homography to '{url}'. The server returned the following error:\n\n"
                f"{response.text}"
            )
