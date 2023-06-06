from unittest import TestCase

import numpy as np

from dreifus.camera_bundle import align_poses
from dreifus.matrix import Pose, Intrinsics
import pyvista as pv
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum


class CameraBundleTest(TestCase):

    def test_align_poses(self):
        camera_1 = Pose.from_euler([np.pi / 4, np.pi / 4, 0], translation=[1, 1, -1])
        camera_2 = Pose.from_euler([np.pi / 4, np.pi / 4, 0], translation=[1, -1, -1])
        camera_3 = Pose.from_euler([np.pi / 4, np.pi / 8, 0], translation=[-1, 1, -1])

        world_to_cam_poses = [camera_1, camera_2, camera_3]
        cam_to_world_poses, transformation = align_poses(world_to_cam_poses, return_transformation=True)

        p = pv.Plotter()
        add_coordinate_axes(p)
        for camera, old_camera in zip(cam_to_world_poses, world_to_cam_poses):

            add_camera_frustum(p, camera, Intrinsics(500, 500, 500, 500))

            cam_to_world_pose = old_camera.invert()
            cam_to_world_pose = transformation @ cam_to_world_pose
            add_camera_frustum(p, cam_to_world_pose, Intrinsics(500, 500, 500, 500), color='red')

            for val_aligned, val_transformed in zip(camera.numpy().flatten(), cam_to_world_pose.numpy().flatten()):
                self.assertAlmostEqual(val_aligned, val_transformed, places=3)
        p.show()