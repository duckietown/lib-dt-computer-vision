import os.path
import typing
from datetime import datetime
from typing import List, Tuple, Optional, Union

import numpy as np
import yaml

if typing.TYPE_CHECKING:
    import dt_computer_vision

Homography = np.ndarray


class ResolutionDependentHomography(Homography):

    # noinspection PyMissingConstructor
    def __init__(self):
        raise RuntimeError("Do not instantiate the class 'ResolutionDependentHomography' directly, use the "
                           "method ResolutionDependentHomography.read() instead.")

    @classmethod
    def read(cls, value: np.ndarray):
        return value.view(ResolutionDependentHomography)

    def camera_independent(
            self, camera: "dt_computer_vision.camera.CameraModel"
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
            self, camera: "dt_computer_vision.camera.CameraModel"
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
    DEFAULT_FORMAT: str = "resolution-independent"
    SUPPORTED_VERSIONS: List[str] = [DEFAULT_VERSION]
    SUPPORTED_FORMATS: List[str] = [DEFAULT_FORMAT]

    @staticmethod
    def load_from_disk(
            fpath: str,
            return_date: bool = False
    ) -> Union[ResolutionIndependentHomography, Tuple[ResolutionIndependentHomography, Optional[datetime]]]:
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
        # - check format
        if "format" not in fcontent:
            raise ValueError(
                f"Invalid homography file '{fpath}'. The required root key 'format' was not found."
            )
        format: str = fcontent["format"].lower().strip()
        if format not in HomographyToolkit.SUPPORTED_FORMATS:
            raise ValueError(
                f"The homography file '{fpath}' uses the format '{format}' which is not supported. "
                f"Supported formats are: {HomographyToolkit.SUPPORTED_FORMATS}"
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
        H: ResolutionIndependentHomography = ResolutionIndependentHomography.read(
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
            H: ResolutionIndependentHomography,
            fpath: str,
            exist_ok: bool = False,
            date: datetime = None,
    ):
        if not isinstance(H, ResolutionIndependentHomography):
            raise ValueError(
                "Only resolution-independent homographies can be stored to disk. Please, "
                "convert your homography to a ResolutionIndependentHomography first."
            )
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
                    "format": HomographyToolkit.DEFAULT_FORMAT,
                    "date": (date or datetime.today()).isoformat(),
                    "homography": H.tolist(),
                },
                sort_keys=False
            )
        )
        # write file
        with open(fpath, "wt") as fout:
            fout.write(fcontent)
