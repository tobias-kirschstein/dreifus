from typing import Union, Type
from unittest import TestCase

import numpy as np
import torch
from numpy.ma.testutils import assert_array_almost_equal, assert_array_equal
from scipy.spatial.transform import Rotation as R

from dreifus.matrix.intrinsics_numpy import Intrinsics
from dreifus.matrix.intrinsics_torch import TorchIntrinsics
from dreifus.matrix.pose_numpy import Pose
from dreifus.matrix.pose_torch import TorchPose


class PoseTest(TestCase):

    def _test_pose(self, cls: Type[Union[Pose, TorchPose]]):
        pose = cls()
        self.assertEqual(pose[0, 0], 1)
        self.assertEqual(pose[1, 1], 1)
        self.assertEqual(pose[2, 2], 1)
        self.assertEqual(pose[3, 3], 1)

        rotation_matrix = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
        pose = cls(rotation_matrix, [1, 2, 3])

        translation = pose.get_translation()
        assert_array_equal(translation, [1, 2, 3])

        euler_angles = pose.get_euler_angles('xyz')
        assert_array_almost_equal(euler_angles, [np.pi, 0, 0])

        pose = cls.from_euler([np.pi, 0, 0])
        euler_angles = pose.get_euler_angles('xyz')
        assert_array_almost_equal(euler_angles, [np.pi, 0, 0])

        rodriguez = R.from_euler('xyz', [np.pi, 0, 0]).as_rotvec()
        pose = cls.from_rodriguez(rodriguez)
        euler_angles = pose.get_euler_angles('xyz')
        assert_array_almost_equal(euler_angles, [np.pi, 0, 0])

        pose = pose.invert()
        euler_angles = pose.get_euler_angles('xyz')
        assert_array_almost_equal(euler_angles, [-np.pi, 0, 0])

        if cls == TorchPose:
            matrix = torch.arange(4 * 4).reshape(4, 4).float()
        else:
            matrix = np.arange(4 * 4).reshape(4, 4)

        pose_l = pose @ matrix
        pose_r = matrix @ pose

        self.assertNotIsInstance(pose_l, cls)
        self.assertNotIsInstance(pose_r, cls)

        pose_other = cls()

        pose_l = pose @ pose_other
        pose_r = pose_other @ pose

        self.assertIsInstance(pose_l, cls)
        self.assertIsInstance(pose_r, cls)

        if cls == TorchPose:
            intrinsics = TorchIntrinsics()
        else:
            intrinsics = Intrinsics()

        pose_l = pose @ intrinsics.homogenize()
        pose_r = intrinsics.homogenize() @ pose

        self.assertNotIsInstance(pose_l, cls)
        self.assertNotIsInstance(pose_r, cls)

    def test_pose(self):
        self._test_pose(Pose)

    def test_torch_pose(self):
        self._test_pose(TorchPose)
