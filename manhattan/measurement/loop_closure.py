import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, overload
from manhattan.geometry.Elements import SEPose, SE2Pose, SE3Pose
from py_factor_graph.utils.matrix_utils import get_measurement_precisions_from_covariance_matrix


class LoopClosure(ABC):
    """
    represents a loop closure between poses

    pose_1 (SEPose): the pose the loop closure is measured from
    pose_2 (SEPose): the pose the loop closure is measured to
    measured_association (str): the measurement association (can be
        incorrect and differ from the true association)
    measured_relative_pose (SEPose): the measured relative pose
    timestamp (int): the timestamp of the measurement
    mean_offset (np.ndarray): the mean offset in the measurement model
    covariance (np.ndarray): the covariance of the measurement model
    """

    def __init__(
        self,
        pose_1: SEPose,
        pose_2: SEPose,
        measured_association: str,
        measured_rel_pose: SEPose,
        timestamp: int,
        mean_offset: np.ndarray,
        covariance: np.ndarray,
    ):
        self.pose_1 = pose_1
        self.pose_2 = pose_2
        self.measured_association = measured_association
        self.measured_rel_pose = measured_rel_pose
        self.timestamp = timestamp
        self.mean_offset = mean_offset
        self.covariance = covariance

    def __str__(self) -> str:
        return (
            f"LoopClosure (t={self.timestamp})\n"
            f"{self.pose_1} -> {self.pose_2}\n"
            f"measured association: {self.measured_association}\n"
            f"{self.measured_rel_pose}\n"
            f"offset: {self.mean_offset}\n"
            f"covariance:\n{self.covariance}"
        )

    @property
    def true_association(self) -> str:
        """
        the true association between poses
        """
        return self.pose_2.local_frame

    @property
    def true_transformation(self):
        return self.pose_1.transform_to(self.pose_2)

    @property
    def base_frame(self) -> str:
        return self.pose_1.local_frame

    @property
    def local_frame(self) -> str:
        return self.pose_2.local_frame

    @property
    def measurement(self) -> SEPose:
        """
        returns the noisy transformation between the poses
        """
        return self.measured_rel_pose

    @property
    def delta_x(self) -> float:
        """
        returns the delta x in the measurement model
        """
        return self.measurement.x

    @property
    def delta_y(self) -> float:
        """
        returns the delta y in the measurement model
        """
        return self.measurement.y
    
    @property
    def delta_z(self) -> float:
        """
        returns the delta z in the measurement model
        """
        return self.measurement.z

    @property
    def delta_angles(self) -> Union[float, Tuple[float, float, float]]:
        """
        returns the delta angles in the measurement model
        """
        return self.measurement.rot.angles()

    # Is the calculation of the translation/rotation precision from the covariance matrix 
    # the same between a Loop Closure Measurement and an Odometry Measurement?
    @property
    @abstractmethod
    def translation_precision(self) -> float:
        """
        returns the precision of the translation
        """
        pass

    @property
    @abstractmethod
    def rotation_precision(self) -> Union[float, Tuple[float, float, float]]:
        """
        returns the precision of the rotation
        """
        pass

class LoopClosure2D(LoopClosure):
    """
    represents a 2D loop closure between poses

    pose_1 (SE2Pose): the pose the loop closure is measured from
    pose_2 (SE2Pose): the pose the loop closure is measured to
    measured_association (str): the measurement association (can be
        incorrect and differ from the true association)
    measured_relative_pose (SE2Pose): the measured relative pose
    timestamp (int): the timestamp of the measurement
    mean_offset (np.ndarray): the mean offset in the measurement model
    covariance (np.ndarray): the covariance of the measurement model
    """
    def __init__(self,
        pose_1: SE2Pose,
        pose_2: SE2Pose,
        measured_association: str,
        measured_rel_pose: SE2Pose,
        timestamp: int,
        mean_offset: np.ndarray,
        covariance: np.ndarray) -> None:

        assert isinstance(pose_1, SE2Pose)
        assert isinstance(pose_2, SE2Pose)
        assert isinstance(measured_rel_pose, SE2Pose)
        assert mean_offset.shape == (3,)
        assert covariance.shape == (3, 3)
        super().__init__(pose_1, pose_2, measured_association, measured_rel_pose, timestamp, mean_offset, covariance)

    @property
    def translation_precision(self) -> float:
        return get_measurement_precisions_from_covariance_matrix(self.covariance, matrix_dim=2)[0]

    @property
    def rotation_precision(self) -> float:
        return get_measurement_precisions_from_covariance_matrix(self.covariance, matrix_dim=2)[0]

class LoopClosure3D(LoopClosure):
    """
    represents a 3D loop closure between poses

    pose_1 (SE3Pose): the pose the loop closure is measured from
    pose_2 (SE3Pose): the pose the loop closure is measured to
    measured_association (str): the measurement association (can be
        incorrect and differ from the true association)
    measured_relative_pose (SE3Pose): the measured relative pose
    timestamp (int): the timestamp of the measurement
    mean_offset (np.ndarray): the mean offset in the measurement model
    covariance (np.ndarray): the covariance of the measurement model
    """
    def __init__(self,
        pose_1: SE3Pose,
        pose_2: SE3Pose,
        measured_association: str,
        measured_rel_pose: SE3Pose,
        timestamp: int,
        mean_offset: np.ndarray,
        covariance: np.ndarray) -> None:

        assert isinstance(pose_1, SE3Pose)
        assert isinstance(pose_2, SE3Pose)
        assert isinstance(measured_rel_pose, SE3Pose)
        assert mean_offset.shape == (6,)
        assert covariance.shape == (6, 6)
        super().__init__(pose_1, pose_2, measured_association, measured_rel_pose, timestamp, mean_offset, covariance)
    
    @property
    def translation_precision(self) -> float:
        return get_measurement_precisions_from_covariance_matrix(self.covariance, matrix_dim=3)[0]

    @property
    def rotation_precision(self) -> float:
        return get_measurement_precisions_from_covariance_matrix(self.covariance, matrix_dim=3)[0]
