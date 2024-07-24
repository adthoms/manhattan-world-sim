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
    assert rpy.shape == (3,)

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