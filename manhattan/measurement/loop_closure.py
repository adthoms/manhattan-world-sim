import numpy as np

from manhattan.geometry.TwoDimension import SE2Pose


class LoopClosure:
    """
    represents a loop closure between poses

    pose_1 (SE2Pose): the pose the loop closure is measured from
    pose_2 (SE2Pose): the pose the loop closure is measured to
    measured_association (str): the measurement association (can be
        incorrect and differ from the true association)
    measured_relative_pose (SE2Pose): the measured relative pose
    timestamp (int): the timestamp of the measurement
    mean_offset (np.ndarray): the mean offset in the measurement model
    covariance (np.ndarray): the covariance of the measurement model
    """

    def __init__(
        self,
        pose_1: SE2Pose,
        pose_2: SE2Pose,
        measured_association: str,
        measured_rel_pose: SE2Pose,
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
    def measurement(self) -> SE2Pose:
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
    def delta_theta(self) -> float:
        """
        returns the delta theta in the measurement model
        """
        return self.measurement.theta

    @property
    def translation_precision(self) -> float:
        """
        returns the precision of the translation
        """
        return 1 / self.covariance[0, 0]

    @property
    def rotation_precision(self) -> float:
        """
        returns the precision of the rotation
        """
        return 1 / self.covariance[2, 2]
