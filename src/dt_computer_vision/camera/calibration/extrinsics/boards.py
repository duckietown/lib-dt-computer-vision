import dataclasses
from enum import Enum
from typing import List
import numpy as np

from ...types import Size, Point

from dt_computer_vision.ground_projection import GroundPoint

class ReferenceFrame(Enum):
    BOARD=0
    ROBOT=1

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

    def corners(self, reference_frame : ReferenceFrame = ReferenceFrame.ROBOT) -> List[GroundPoint]:
        # ground points, easily reconstructable given a known board
        ground_corners: List[GroundPoint] = []
        
        if reference_frame == ReferenceFrame.BOARD:
            board_offset = np.array([0, 0])
        elif reference_frame == ReferenceFrame.ROBOT:
            board_offset = np.array([-self.square_size * (self.columns - 1) / 2, 0])

        square_size = self.square_size
        
        # OpenCV labels corners left-to-right, top-to-bottom, let's do the same
        for i in range(self.rows):
            for j in range(self.columns):
                object_point = np.array([j * square_size, i * square_size]) + board_offset
                ground_corners.append(GroundPoint(*object_point))
        # ---
        return ground_corners


CalibrationBoard8by6 = CalibrationBoard(
    rows=5,
    columns=7,
    square_size=0.031,
    x_offset=0.16,
    y_offset=-0.124,
)

CalibrationBoardDD24 = CalibrationBoard(
    rows=5,
    columns=7,
    square_size=0.017,
    x_offset=0.25,
    y_offset=-0.080,
)
