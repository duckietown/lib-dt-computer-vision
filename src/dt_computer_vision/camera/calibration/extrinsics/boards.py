import dataclasses
from typing import List
import numpy as np

from ...types import Size, Point

from dt_computer_vision.ground_projection import GroundPoint


@dataclasses.dataclass
class CalibrationBoard:
    rows: int
    columns: int
    square_size: float
    x_offset: float
    y_offset: float

    @property
    def size(self) -> Size:
        return Size(x=self.square_size * self.rows, y=self.square_size * self.columns)

    @property
    def chessboard_offset(self) -> Point:
        return Point(self.x_offset, self.y_offset)

    def corners(self) -> List[GroundPoint]:
        # ground points, easily reconstructable given a known board
        ground_corners: List[GroundPoint] = []
        board_offset = np.array([self.x_offset, self.y_offset])
        square_size = self.square_size
        for r in range(self.rows - 1):
            for c in range(self.columns - 1):
                src_corner = np.array([(r + 1) * square_size, (c + 1) * square_size]) + board_offset
                ground_corners.append(GroundPoint(*src_corner))
        # OpenCV labels corners left-to-right, top-to-bottom, let's do the same
        ground_corners = ground_corners[::-1]
        # ---
        return ground_corners


CalibrationBoard8by6 = CalibrationBoard(
    rows=6,
    columns=8,
    square_size=0.031,
    x_offset=0.16,
    y_offset=-0.124,
)
