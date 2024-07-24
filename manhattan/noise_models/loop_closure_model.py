from abc import abstractmethod
from typing import Union
import numpy as np

from manhattan.measurement.loop_closure import LoopClosure, LoopClosure2, LoopClosure3
from manhattan.geometry.Elements import SEPose, SE2Pose, SE3Pose, DIM

class LoopClosureModel:
    """
    A base noisy loop closure model.
    """

    def __init__(
        self,
    ):
        """
        Initialize this noise model.
        """
        self._covariance = None
        self._mean = None

    def __str__(self):
        return (
            f"Generic Loop Closure Model\n"
            + f"Covariance: {self._covariance.flatten()}\n"
            + f"Mean: {self._mean}\n"
        )

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def mean(self) -> np.ndarray:
        return self._mean
    
    @property
    @abstractmethod
    def dim(self) -> DIM:
        pass

    @abstractmethod
    def get_relative_pose_measurement(
        self, pose_1: SEPose, pose_2: SEPose, association: str, timestamp: int
    ) -> Union[LoopClosure2, LoopClosure3]:
        """Takes a two poses, and returns a loop closure measurement based on
        the relative pose from pose_1 to pose_2 and the determined sensor model.

        Note: we get a little hacky with the frame naming just to allow for
        multiplication of all of these noises

        Args:
            pose_1 (SE2Pose): the first pose
            pose_2 (SE2Pose): the second pose
            association (str): the measured association of the second pose
            timestamp (int): the timestamp of the measurement

        Returns:
            LoopClosure: A noisy measurement of the relative pose from pose_1 to
                pose_2
        """
        pass


class GaussianLoopClosureModel2(LoopClosureModel):
    """
    This is a simple Gaussian noise model for the robot odometry. This assumes
    that the noise is additive gaussian and constant in time and distance moved.
    """

    def __init__(
        self, mean: np.ndarray = np.zeros(3), covariance: np.ndarray = np.eye(3) / 50.0
    ) -> None:
        """Initializes the gaussian additive noise model

        Args:
            mean (np.ndarray, optional): The mean of the noise. Generally
                zero-mean. Entries are [x, y, theta]. Defaults to np.zeros(3).
            covariance (np.ndarray, optional): The covariance matrix
                with entries corresponding to the mean vector. Defaults to
                np.eye(3)/50.0.
        """
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (3,)
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (3, 3)
        self._mean = mean
        self._covariance = covariance

    def __str__(self):
        return (
            f"Gaussian Additive Loop Closure Model\n"
            + f"Covariance: {self._covariance.flatten()}\n"
            + f"Mean: {self._mean}\n"
        )

    @property
    def dim(self) -> DIM:
        return DIM.TWO

    def get_relative_pose_measurement(
        self, pose_1: SE2Pose, pose_2: SE2Pose, association: str, timestamp: int
    ) -> LoopClosure2:
        """Takes a two poses, gets the relative pose and then perturbs it by
        transformation randomly sampled from a Gaussian distribution and passed
        through the exponential map

        Note: we get a little hacky with the frame naming just to allow for
        multiplication of all of these noises

        Args:
            pose_1 (SE2Pose): the first pose
            pose_2 (SE2Pose): the second pose
            association (str): the measured association of the second pose
            timestamp (int): the timestamp of the measurement

        Returns:
            SE2Pose: A noisy measurement of the relative pose from pose_1 to
                pose_2
        """
        assert isinstance(pose_1, SE2Pose)
        assert isinstance(pose_2, SE2Pose)
        assert isinstance(timestamp, int)
        assert 0 <= timestamp

        rel_pose = pose_1.transform_to(pose_2)

        # this is the constant component from the gaussian noise
        mean_offset = SE2Pose.by_exp_map(
            self._mean, local_frame="temp", base_frame=rel_pose.local_frame
        )

        # this is the random component from the gaussian noise
        noise_sample = np.random.multivariate_normal(np.zeros(3), self._covariance)
        noise_offset = SE2Pose.by_exp_map(
            noise_sample, local_frame=rel_pose.local_frame, base_frame="temp"
        )

        # because we're in 2D rotations commute so we don't need to think about
        # the order of operations???
        noisy_pose_measurement = rel_pose * mean_offset * noise_offset
        return LoopClosure2(
            pose_1=pose_1,
            pose_2=pose_2,
            measured_association=association,
            measured_rel_pose=noisy_pose_measurement,
            timestamp=timestamp,
            mean_offset=self._mean,
            covariance=self._covariance,
        )

class GaussianLoopClosureModel3(LoopClosureModel):
    """
    This is a simple Gaussian noise model for the robot odometry. This assumes
    that the noise is additive gaussian and constant in time and distance moved.
    """

    def __init__(
        self, mean: np.ndarray = np.zeros(6), covariance: np.ndarray = np.eye(6) / 50.0
    ) -> None:
        """Initializes the gaussian additive noise model

        Args:
            mean (np.ndarray, optional): The mean of the noise. Generally
                zero-mean. Entries are [x, y, z, roll, pitch, yaw]. Defaults to np.zeros(6).
            covariance (np.ndarray, optional): The covariance matrix
                with entries corresponding to the mean vector. Defaults to
                np.eye(6)/50.0.
        """
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (6,)
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (6, 6)
        self._mean = mean
        self._covariance = covariance

    def __str__(self):
        return (
            f"Gaussian Additive Loop Closure Model\n"
            + f"Covariance: {self._covariance.flatten()}\n"
            + f"Mean: {self._mean}\n"
        )
    
    @property
    def dim(self) -> DIM:
        return DIM.THREE

    def get_relative_pose_measurement(
        self, pose_1: SE3Pose, pose_2: SE3Pose, association: str, timestamp: int
    ) -> LoopClosure3:
        """Takes a two poses, gets the relative pose and then perturbs it by
        transformation randomly sampled from a Gaussian distribution and passed
        through the exponential map

        Note: we get a little hacky with the frame naming just to allow for
        multiplication of all of these noises

        Args:
            pose_1 (SE3Pose): the first pose
            pose_2 (SE3Pose): the second pose
            association (str): the measured association of the second pose
            timestamp (int): the timestamp of the measurement

        Returns:
            SE3Pose: A noisy measurement of the relative pose from pose_1 to
                pose_2
        """
        assert isinstance(pose_1, SE3Pose)
        assert isinstance(pose_2, SE3Pose)
        assert isinstance(timestamp, int)
        assert 0 <= timestamp

        rel_pose = pose_1.transform_to(pose_2)

        # this is the constant component from the gaussian noise
        mean_offset = SE3Pose.by_exp_map(
            self._mean, local_frame="temp", base_frame=rel_pose.local_frame
        )

        # this is the random component from the gaussian noise
        noise_sample = np.random.multivariate_normal(np.zeros(6), self._covariance)
        noise_offset = SE3Pose.by_exp_map(
            noise_sample, local_frame=rel_pose.local_frame, base_frame="temp"
        )

        # What are the order of operations on the multiplication of these poses?
        noisy_pose_measurement = rel_pose * mean_offset * noise_offset
        return LoopClosure3(
            pose_1=pose_1,
            pose_2=pose_2,
            measured_association=association,
            measured_rel_pose=noisy_pose_measurement,
            timestamp=timestamp,
            mean_offset=self._mean,
            covariance=self._covariance,
        )
