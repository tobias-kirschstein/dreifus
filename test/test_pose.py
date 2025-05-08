import pickle
from typing import Union, Type, List
from unittest import TestCase

import numpy as np
import torch
from numpy.ma.testutils import assert_array_almost_equal, assert_array_equal
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset

from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix.intrinsics_numpy import Intrinsics
from dreifus.matrix.intrinsics_torch import TorchIntrinsics
from dreifus.matrix.pose_numpy import Pose
from dreifus.matrix.pose_torch import TorchPose
from dreifus.vector import Vec3


class PoseDataset(Dataset):
    def __getitem__(self, item):
        return Pose()

    def __len__(self):
        return 2

    def collate_fn(self, samples: List[Pose]):
        return samples


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

        # pose is CAM_2_WORLD due to invert()
        pose_l = pose @ pose_other
        pose_r = pose_other.invert() @ pose.invert()

        with self.assertRaises(ValueError):
            # Should not work as pose_other is WORLD_2_CAM and pose is CAM_2_WORLD. WORLD -> WORLD doesn't exist
            pose_other @ pose

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

        pose = Pose(camera_coordinate_convention=CameraCoordinateConvention.DIRECT_X, pose_type=PoseType.CAM_2_WORLD)
        pose_copy = pose.copy()

        self.assertEqual(pose.camera_coordinate_convention, pose_copy.camera_coordinate_convention)
        self.assertEqual(pose.pose_type, pose_copy.pose_type)

        pose_change = pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV)
        # Changing the camera coordinate convention should affect some of the entries in the matrix
        self.assertFalse((pose == pose_change).all())

    def test_pose(self):
        self._test_pose(Pose)

    def test_torch_pose(self):
        self._test_pose(TorchPose)

    def test_serialize_pose(self):
        pose = Pose.from_euler((np.pi, 0, 0), Vec3(0, 0, 0), camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL, pose_type=PoseType.CAM_2_WORLD)
        pose_serialized = pickle.dumps(pose)
        pose_deserialized = pickle.loads(pose_serialized)

        self.assertTrue((pose == pose_deserialized).all())
        for k in pose.__dict__.keys():
            self.assertEqual(getattr(pose, k), getattr(pose_deserialized, k))

        pose_deserialized.change_camera_coordinate_convention(CameraCoordinateConvention.PYTORCH_3D)
        pose_deserialized.change_pose_type(PoseType.WORLD_2_CAM)

        dataset = PoseDataset()
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, num_workers=2)
        iter_dataloader = iter(dataloader)
        batch = next(iter_dataloader)
        pose = batch[0]
        pose.change_pose_type(PoseType.CAM_2_WORLD)
        pose.change_camera_coordinate_convention(CameraCoordinateConvention.PYTORCH_3D)
