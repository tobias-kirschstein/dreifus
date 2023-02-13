from typing import Optional, Union, List

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dreifus.matrix.pose_base import is_rotation_matrix
from dreifus.vector.vector_base import Vec3TypeX, FloatType, unpack_3d_params, Vec3Type
from dreifus.vector.vector_numpy import Vec3, Vec4


class Pose(np.ndarray):

    def __new__(cls,
                matrix_or_rotation: Union[np.ndarray, List] = np.eye(4),
                translation: Optional[Vec3Type] = None):
        pose = super().__new__(cls, (4, 4), dtype=np.float32)
        if not isinstance(matrix_or_rotation, np.ndarray):
            matrix_or_rotation = np.asarray(matrix_or_rotation)

        if matrix_or_rotation.shape == (4, 4):
            # Full 4x4 Pose
            assert translation is None, "If a full pose is given, no translation should be specified!"
            assert (matrix_or_rotation[3, :] == [0, 0, 0, 1]).all(), \
                f"Last row of pose must be [0, 0, 0, 1]. Got {matrix_or_rotation[3, :]}"
            assert is_rotation_matrix(matrix_or_rotation[:3, :3]), \
                f"Specified matrix does not contain a valid rotation matrix! {matrix_or_rotation[:3, :3]}"

            pose[:] = matrix_or_rotation

        elif matrix_or_rotation.shape == (3, 3):
            # 3x3 rotation matrix + 3(x1) translation vector
            translation = np.asarray(translation)
            assert translation.squeeze().shape == (3,), \
                "If a rotation matrix is given, the translation vector has to be 3 dimensional!"

            pose[:3, :3] = matrix_or_rotation
            pose[:3, 3] = translation.squeeze()
            pose[3, :] = [0, 0, 0, 1]
        elif matrix_or_rotation.shape == (3, 1) or matrix_or_rotation.shape == (3,):
            # 3(x1) Rodriguez vector + 3(x1) translation vector
            assert translation.squeeze().shape == (3,), \
                "If a Rodriguez vector is given, the translation vector has to be 3 dimensional!"

            pose[:3, :3] = cv2.Rodrigues(matrix_or_rotation)[0]
            pose[:3, 3] = translation.squeeze()
            pose[3, :] = [0, 0, 0, 1]
        else:
            raise ValueError("Either a full pose has to be given or a 3x3 rotation + 3x1 translation!")

        return pose

    @staticmethod
    def from_euler(euler_angles: Vec3Type, translation: Vec3Type = Vec3(), euler_mode: str = 'XYZ') -> 'Pose':
        return Pose(R.from_euler(euler_mode, euler_angles).as_matrix(), translation)

    @staticmethod
    def from_rodriguez(rodriguez_vector: Vec3Type, translation: Vec3Type = Vec3()) -> 'Pose':
        return Pose(cv2.Rodrigues(rodriguez_vector)[0], translation)

    def get_rotation_matrix(self) -> np.ndarray:
        return self[:3, :3]

    def get_euler_angles(self, order: str) -> Vec3:
        return Vec3(R.from_matrix(self.get_rotation_matrix()).as_euler(order))

    def get_rodriguez_vector(self) -> Vec3:
        return Vec3(cv2.Rodrigues(self.get_rotation_matrix())[0].squeeze())

    def get_quaternion(self) -> Vec4:
        return Vec4(R.from_matrix(self.get_rotation_matrix()).as_quat())

    def get_translation(self) -> Vec3:
        return Vec3(self[:3, 3])

    def set_translation(self, x: Vec3TypeX, y: Optional[FloatType] = None, z: Optional[FloatType] = None):
        x, y, z = unpack_3d_params(x, y, z)
        if x is not None:
            self[0, 3] = x
        if y is not None:
            self[1, 3] = y
        if z is not None:
            self[2, 3] = z

    def move(self, x: Optional[Vec3TypeX] = None, y: Optional[FloatType] = None, z: Optional[FloatType] = None):
        x, y, z = unpack_3d_params(x, y, z, 0)
        self[0, 3] += x
        self[1, 3] += y
        self[2, 3] += z

    def scale(self, scale: float) -> 'Pose':
        self[:3, 3] *= scale
        return self

    def set_rotation_matrix(self, rotation_matrix: np.ndarray):
        assert is_rotation_matrix(rotation_matrix), \
            f"Specified matrix does not contain a valid rotation matrix! {rotation_matrix}"
        self[:3, :3] = rotation_matrix

    def set_rotation_euler(self, order: str,
                           euler_x: Vec3TypeX = 0,
                           euler_y: Optional[float] = None,
                           euler_z: Optional[float] = None):
        euler_x, euler_y, euler_z = unpack_3d_params(euler_x, euler_y, euler_z, default=0)
        self[:3, :3] = R.from_euler(order, [euler_x, euler_y, euler_z]).as_matrix()

    def rotate_euler(self,
                     order: str,
                     euler_x: Vec3TypeX = 0,
                     euler_y: Optional[float] = None,
                     euler_z: Optional[float] = None):
        euler_x, euler_y, euler_z = unpack_3d_params(euler_x, euler_y, euler_z, default=0)
        euler_rotation = Vec3(euler_x, euler_y, euler_z)
        current_euler_angles = self.get_euler_angles(order)
        self.set_rotation_euler(order, current_euler_angles + euler_rotation)

    def invert(self) -> 'Pose':
        inverted_rotation = self.get_rotation_matrix().T
        inverted_translation = -inverted_rotation @ self.get_translation()
        inverted_pose = Pose(inverted_rotation, inverted_translation)
        return inverted_pose

    def negate_orientation_axis(self, axis: int):
        """
        Changes the coordinate convention of the camera space.
        E.g., if the pose is assumed to be in OpenCV camera space (x -> right, y -> down, z -> forward), then
        negate_orientation_axis(1) will make it DirectX (x -> right, y -> up, z -> forward).
        The function negates the column of the rotation matrix that corresponds to the given axis.
        The camera's position in world space is NOT affected by this transformation.
        Only the orientation of the camera frustum is changed.
        Assumes the current pose is cam2world.

        Parameters
        ----------
            axis: which axis to negate.
        """

        # negates the column of the rotation matrix
        self[:3, axis] *= -1

    def swap_axes(self, permutation: List[Union[int, str]]):
        """
        Negates/swaps entire rows of the pose matrix.
        Essentially, applies a rotation / flip operation around the world origin to the camera object.
        E.g., swap_axes(['-x', 'y', 'z']) will move an origin-facing camera that was at (3, 1, 1) to (-3, 1, 1).
        The moved camera will still face the origin afterwards (i.e., the camera is not just moved but also rotated).
        Alternatively, instead of flipping the camera along axis-aligned planes, swap_axes() can be interpreted as
        flipping the actual scene in world space while keeping the camera where it was.
        I.e., after applying the camera transformation the scene will be rendered as if it was flipped.
        Assumes the current pose is cam2world

        Parameters
        ----------
            permutation: 3-tuple of axis indicators (either 0, 1, 2 or x, y, z with optional '-' signs)
        """

        axis_switcher = np.zeros((4, 4))
        axis_order = ['x', 'y', 'z']
        for idx, a in enumerate(permutation):
            v = 1
            if isinstance(a, int):
                ax = a
            else:
                # Possibility to also flip an axis via -x, -y etc.
                ax = axis_order.index(a[-1])  # Map x -> 0, y -> 1, z -> 2
                if a[0] == '-':
                    # Axis shall be flipped
                    v = -1

            axis_switcher[idx, ax] = v
        axis_switcher[3, 3] = 1

        # Negates / Flips rows
        self[:, :] = axis_switcher @ self

    def get_look_direction(self) -> 'Vec3':
        # Assumes the current pose is cam2world
        # Assigns meaning to the coordinate axes. Hence, the coordinate system convention is important
        # Assumes an OpenCV camera coordinate system convention (x -> right, y -> down, z -> forward/look)

        look_direction = self[:3, 2]

        return look_direction

    def get_up_direction(self) -> 'Vec3':
        # Assumes the current pose is cam2world
        # Assigns meaning to the coordinate axes. Hence, the coordinate system convention is important
        # Assumes an OpenCV camera coordinate system convention (x -> right, y -> down, z -> forward)
        up_direction = -self[:3, 1]

        return up_direction

    def look_at(self, at: Vec3, up: Vec3 = Vec3(0, 0, 1)):
        # This method assigns meaning to the coordinate axes. Hence, the coordinate system convention is important
        # We use the OpenCV camera coordinate system convention

        # Poses are always assumed to be cam2world
        # That way the translation part of the pose matrix is the location of the object in world space

        eye = self.get_translation()
        z_axis = (at - eye).normalize()  # Assumes z-axis is forward
        x_axis = z_axis.cross(up).normalize()  # Assumes y-axis is up
        y_axis = x_axis.cross(z_axis).normalize()

        # Important as otherwise rotation matrix has negative determinant (would be left-handed).
        # Makes it a [x, -y, z] OpenCV camera coordinate system
        # [x, y, -z] would be a Blender/OpenGL camera coordinate system
        y_axis = - y_axis

        self.set_rotation_matrix(np.array([x_axis, y_axis, z_axis]).T)
        # self.set_translation(np.dot(x_axis, eye), np.dot(y_axis, eye), np.dot(z_axis, eye))
        self.set_translation(eye)

    def __rmatmul__(self, other):
        if isinstance(other, Pose):
            return super(Pose, self).__rmatmul__(other)
        else:
            return other @ np.array(self)

    def __matmul__(self, other):
        # TODO: figure out why numpy operations automatically cast to Pose again
        if isinstance(other, Pose):
            return super(Pose, self).__matmul__(other)
        else:
            return np.array(self) @ other