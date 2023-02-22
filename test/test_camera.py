from unittest import TestCase

from dreifus.camera import CameraCoordinateConvention, AxisDirection, Handedness, PoseType


class CameraTest(TestCase):

    def test_coordinate_convention(self):
        ccc = CameraCoordinateConvention.OPEN_CV
        self.assertEqual(ccc.up_direction, AxisDirection.NEG_Y)
        self.assertEqual(ccc.forward_direction, AxisDirection.Z)
        self.assertEqual(ccc.handedness, Handedness.RIGHT_HANDED)

        ccc = CameraCoordinateConvention.OPEN_GL
        self.assertEqual(ccc.up_direction, AxisDirection.Y)
        self.assertEqual(ccc.forward_direction, AxisDirection.NEG_Z)
        self.assertEqual(ccc.handedness, Handedness.RIGHT_HANDED)

        ccc = CameraCoordinateConvention.DIRECT_X
        self.assertEqual(ccc.up_direction, AxisDirection.Y)
        self.assertEqual(ccc.forward_direction, AxisDirection.Z)
        self.assertEqual(ccc.handedness, Handedness.LEFT_HANDED)

    def test_pose_type(self):
        pose_type = PoseType.CAM_2_WORLD
        self.assertEqual(pose_type.invert(), PoseType.WORLD_2_CAM)
        self.assertEqual(pose_type.invert().invert(), PoseType.CAM_2_WORLD)