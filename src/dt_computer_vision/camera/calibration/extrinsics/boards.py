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
        """Obtain a list of ground points of the interior corners of the calibration board.

        Args:
            reference_frame (ReferenceFrame, optional): Reference frame in which to express the board points. Defaults to ReferenceFrame.ROBOT.

        Returns:
            List[GroundPoint]: _description_
        """
        # ground points, easily reconstructable given a known board
        ground_corners: List[GroundPoint] = []
        square_size = self.square_size
        
        if reference_frame == ReferenceFrame.BOARD:
            # In this case we express the board coordinates in the board's frame
            # OpenCV labels corners left-to-right, top-to-bottom, let's do the same
            for i in range(self.rows):
                for j in range(self.columns):
                    object_point = np.array([j * square_size, i * square_size])
                    ground_corners.append(GroundPoint(*object_point))
            # ---
            return ground_corners

        elif reference_frame == ReferenceFrame.ROBOT:
            # In this case we express the board coordinates in the robot's frame
            board_offset = np.array([self.x_offset, self.y_offset])

            # OpenCV labels corners left-to-right, top-to-bottom, let's do the same
            for i in range(self.rows):
                for j in range(self.columns):
                    object_point = np.array([(self.rows-i) * square_size, - (j+1) * square_size]) + board_offset
                    ground_corners.append(GroundPoint(*object_point))
            # ---
            return ground_corners


CalibrationBoard8by6 = CalibrationBoard(
    rows=5,
    columns=7,
    square_size=0.031,
    x_offset=0.16,
    y_offset=0.124,
)

CalibrationBoardDD24 = CalibrationBoard(
    rows=5,
    columns=7,
    square_size=0.017,
    x_offset=0.25,
    y_offset=-0.080,
)
