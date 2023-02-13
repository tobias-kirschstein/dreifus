from typing import Union, Optional

import numpy as np


class Intrinsics(np.ndarray):
    def __new__(cls,
                matrix_or_fx: Union[np.ndarray, float] = np.eye(3),
                fy: Optional[float] = None,
                cx: Optional[float] = None,
                cy: Optional[float] = None,
                s: Optional[float] = None) -> 'Intrinsics':
        intrinsics = super().__new__(cls, (3, 3), dtype=np.float32)
        if not isinstance(matrix_or_fx, np.ndarray) and not isinstance(matrix_or_fx, (float, int)):
            matrix_or_fx = np.asarray(matrix_or_fx)

        if isinstance(matrix_or_fx, np.ndarray) and matrix_or_fx.shape == (3, 3):
            assert fy is None and cx is None and cy is None and s is None, \
                "If a full intrinsics matrix is given, no other parameters should be specified!"
            intrinsics[:] = matrix_or_fx
        elif isinstance(matrix_or_fx, (float, int)):
            assert not (cx is None or cy is None), \
                "If a focal length is given, cx and cy have to be specified!"

            s = 0 if s is None else s
            fy = matrix_or_fx if fy is None else fy

            intrinsics.fill(0)
            intrinsics[0, 0] = matrix_or_fx
            intrinsics[0, 1] = s
            intrinsics[0, 2] = cx
            intrinsics[1, 1] = fy
            intrinsics[1, 2] = cy
            intrinsics[2, 2] = 1
        else:
            print(matrix_or_fx, type(matrix_or_fx))
            raise ValueError("Either a full intrinsics matrix has to be given or fx, cx and cy")

        return intrinsics

    @property
    def fx(self) -> float:
        return self[0, 0].item()

    @property
    def fy(self) -> float:
        return self[1, 1].item()

    @property
    def cx(self) -> float:
        return self[0, 2].item()

    @property
    def cy(self) -> float:
        return self[1, 2].item()

    @property
    def s(self) -> float:
        return self[0, 1].item()

    def rescale(self, scale_factor: float):
        self[0, 0] *= scale_factor
        self[1, 1] *= scale_factor
        self[0, 2] *= scale_factor
        self[1, 2] *= scale_factor
        # What about s?

    def homogenize(self, invert: bool = False) -> np.ndarray:
        homogenized = np.eye(4)
        homogenized[:3, :3] = self

        if invert:
            homogenized = np.linalg.inv(homogenized)

        return homogenized

    def invert(self) -> np.ndarray:
        return np.linalg.inv(self)

    def __rmatmul__(self, other):
        if isinstance(other, Intrinsics):
            return super(Intrinsics, self).__rmatmul__(other)
        else:
            return other @ np.array(self)

    def __matmul__(self, other):
        # TODO: figure out why numpy operations automatically cast to Pose again
        if isinstance(other, Intrinsics):
            return super(Intrinsics, self).__matmul__(other)
        else:
            return np.array(self) @ other
