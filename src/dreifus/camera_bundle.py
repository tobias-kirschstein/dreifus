from typing import List, Optional

import numpy as np
from dreifus.camera import PoseType

from dreifus.matrix import Pose
from dreifus.vector import Vec3, rotation_matrix_between_vectors, offset_vector_between_line_and_point, \
    angle_between_vectors


def calculate_look_center(cam_to_world_poses: List[Pose], distance_to_center: float) -> Vec3:
    look_centers = []
    for cam_to_world in cam_to_world_poses:
        look_direction = -cam_to_world.get_look_direction()  # TODO: get_look_direction() is negated now, can we remove the minus?
        position = cam_to_world.get_translation()
        # TODO: This is not a very sophisticated approach
        look_center = position + distance_to_center * look_direction
        look_centers.append(look_center)

    return Vec3(np.mean(look_centers, axis=0))


def align_poses(world_to_cam_poses: List[Pose],
                up: Optional[Vec3] = Vec3(0, 1, 0),
                look: Optional[Vec3] = Vec3(0, 0, -1),
                look_center: Optional[Vec3] = Vec3(0, 0, 0),
                cam_to_world: bool = False) -> List[Pose]:
    """
    Calibration poses can be arbitrarily aligned. This method provides a utility to transform a set of camera poses
    such that their up/look directions and look center correspond to the specified values.
    calibration poses are expected in world_to_cam format and OpenCV coordinate convention
    (i.e., x -> right, y -> down, z -> forward).
    Per default, the set of camera poses will be transformed to look at the center in an OpenGL world
    (i.e., x -> right, y -> up, z -> backward).

    Parameters
    ----------
        world_to_cam_poses: the poses to transform
        up: where the up direction should point to
        look: where the look direction should point to
        look_center: where the look center of all cameras should fall into
        cam_to_world: whether the provided poses are already cam_to_world

    Returns
    -------
        the re-aligned camera poses
    """

    if cam_to_world:
        cam_to_world_poses = world_to_cam_poses
    else:
        cam_to_world_poses = [cam_pose.invert() for cam_pose in world_to_cam_poses]

    # Align up direction
    if up is not None:
        up_directions = [cam_pose.get_up_direction() for cam_pose in cam_to_world_poses]
        average_up_direction = np.mean(up_directions, axis=0)
        align_up_rotation = rotation_matrix_between_vectors(average_up_direction, up)
        rotator_up = Pose(align_up_rotation, Vec3(), pose_type=PoseType.CAM_2_CAM)
        cam_to_world_poses = [rotator_up @ cam_pose for cam_pose in cam_to_world_poses]

    # Align the look direction
    if look is not None:
        look_directions = [cam_pose.get_look_direction() for cam_pose in cam_to_world_poses]
        average_look_direction = np.mean(look_directions, axis=0)
        align_look_rotation = rotation_matrix_between_vectors(average_look_direction, look)
        rotator_look = Pose(align_look_rotation, Vec3(), pose_type=PoseType.CAM_2_CAM)
        cam_to_world_poses = [rotator_look @ cam_pose for cam_pose in cam_to_world_poses]

    # Align the look center
    if look_center is not None:
        look_directions = [cam_pose.get_look_direction() for cam_pose in cam_to_world_poses]
        cameras_center = np.mean([cam_pose.get_translation() for cam_pose in cam_to_world_poses], axis=0)
        average_look_direction = np.mean(look_directions, axis=0)
        # TODO: This won't move cameras much if cameras_center is already at look_center
        #   Would have to somehow find the point that is closest to all camera rays
        offset_vector = offset_vector_between_line_and_point(cameras_center, average_look_direction, look_center)
        for cam_pose in cam_to_world_poses:
            cam_pose.move(offset_vector)

    if up is not None:
        # Aligning the look direction might mess up the up direction again
        up_directions = [cam_pose.get_up_direction() for cam_pose in cam_to_world_poses]
        average_up_direction = np.mean(up_directions, axis=0)
        angle = angle_between_vectors(average_up_direction, up)

        # TODO: Here we assume that look direction should be z axis. Correct would be to rotate around look direction
        rotator = Pose.from_euler(Vec3(0, 0, -angle), pose_type=PoseType.CAM_2_CAM)
        cam_to_world_poses = [rotator @ cam_pose for cam_pose in cam_to_world_poses]

    return cam_to_world_poses
