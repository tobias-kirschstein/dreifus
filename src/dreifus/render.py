import numpy as np

from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose, Intrinsics


def back_project(points: np.ndarray, depths: np.ndarray, cam_to_world_pose: Pose, intrinsics: Intrinsics) -> np.ndarray:
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert len(depths.shape) == 1
    N = points.shape[0]
    assert depths.shape[0] == N, "Need to have one depth value per point"
    assert cam_to_world_pose.camera_coordinate_convention == CameraCoordinateConvention.OPEN_CV
    assert cam_to_world_pose.pose_type == PoseType.CAM_2_WORLD

    p_screen = np.hstack([points, np.ones((points.shape[0], 1))])
    p_screen_canonical = p_screen @ intrinsics.invert().T
    p_cam = p_screen_canonical * np.expand_dims(depths, 1)
    p_cam_hom = np.hstack([p_cam, np.ones((p_cam.shape[0], 1))])
    p_world = p_cam_hom @ cam_to_world_pose.T

    return p_world[:, :3]
