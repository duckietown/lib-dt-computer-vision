import itertools
from typing import Tuple, Any, Optional

import numpy as np


def invert_map(mapx: np.ndarray, mapy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = mapx.shape[0:2]
    rmapx = np.empty_like(mapx)
    rmapx.fill(np.nan)
    rmapy = np.empty_like(mapx)
    rmapy.fill(np.nan)

    for y, x in itertools.product(list(range(H)), list(range(W))):
        tx = mapx[y, x]
        ty = mapy[y, x]

        tx = int(np.round(tx))
        ty = int(np.round(ty))

        if (0 <= tx < W) and (0 <= ty < H):
            rmapx[ty, tx] = x
            rmapy[ty, tx] = y

    fill_holes(rmapx, rmapy)
    return rmapx, rmapy


def fill_holes(rmapx, rmapy):
    H, W = rmapx.shape[0:2]

    R = 2
    F = R * 2 + 1

    def norm(x):
        return np.hypot(x[0], x[1])

    deltas0 = [(i - R - 1, j - R - 1) for i, j in itertools.product(list(range(F)), list(range(F)))]
    deltas0 = [x for x in deltas0 if norm(x) <= R]
    deltas0.sort(key=norm)

    holes = set()

    for i, j in itertools.product(list(range(H)), list(range(W))):
        if np.isnan(rmapx[i, j]):
            holes.add((i, j))

    while holes:
        nholes = len(holes)
        nholes_filled = 0

        for i, j in list(holes):
            # there is nan
            nholes += 1
            for di, dj in deltas0:
                u = i + di
                v = j + dj
                if (0 <= u < H) and (0 <= v < W):
                    if not np.isnan(rmapx[u, v]):
                        rmapx[i, j] = rmapx[u, v]
                        rmapy[i, j] = rmapy[u, v]
                        nholes_filled += 1
                        holes.remove((i, j))
                        break

        if nholes_filled == 0:
            break


def ensure_ndarray(obj: Any, shape: Optional[Tuple[int, ...]] = None):
    obj = obj if isinstance(obj, np.ndarray) else np.array(obj)
    if shape is not None:
        assert obj.shape == shape, f"Expected shape {shape}, received {obj.shape} instead."
    return obj
