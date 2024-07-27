from typing import Tuple
import numpy as np
from numpy import ndarray
import math

def get_rot_matrix_from_rpy(rpy: Tuple[float, float, float]) -> np.ndarray:
    """
    Return the rotation matrix from roll, pitch, yaw.

    Args:
        rpy (Tuple[float, float, float]): roll, pitch, yaw

    Returns:
        np.ndarray: rotation matrix
    """
    assert isinstance(rpy, Tuple)
    assert len(rpy) == 3

    roll, pitch, yaw = rpy
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ]
    )
    rot_y = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )
    rot_z = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    return rot_z @ rot_y @ rot_x

def bearing_is_behind_robot(pitch: float, yaw: float, tolerance: float) -> bool:
    """Returns true if 3d bearing is behind robot.

    Args:
        pitch (float): pitch
        yaw (float): yaw
    """

    # behind: if pitch or yaw is greater than 90 degrees, but not both or neither (XOR)
    # in front: if pitch and yaw are both less than 90 degrees or both greater than 90 degrees (XNOR)
    pitch_greater = bool(abs(pitch) > math.pi / 2 + tolerance)
    yaw_greater = bool(abs(yaw) > math.pi / 2 + tolerance)
    return pitch_greater ^ yaw_greater