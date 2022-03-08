import dataclasses


@dataclasses.dataclass
class CalibrationBoard:
    rows: int
    columns: int
    square_size: float
    x_offset: float
    y_offset: float


CalibrationBoard8by6 = CalibrationBoard(
    rows=6,
    columns=8,
    square_size=0.031,
    x_offset=0.16,
    y_offset=-0.124,
)
