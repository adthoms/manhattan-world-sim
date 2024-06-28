from abc import ABC, abstractmethod
import numpy as np

from manhattan.measurement.odom_measurement import OdomMeasurement, OdomMeasurement2D, OdomMeasurement3D
from manhattan.geometry.Elements import SEPose, SE2Pose, SE3Pose
from numpy import ndarray
from typing import List, Tuple, Union, Optional, overload


class OdomNoiseModel(ABC):
    """
    A base odometry noise model.
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
            f"Generic Odometry\n"
            + f"Covariance: {self._covariance.flatten()}\n"
            + f"Mean: {self._mean}\n"
        )

    @abstractmethod
    def get_odometry_measurement(self, movement: SEPose) -> Union[OdomMeasurement2D, OdomMeasurement3D]:
        """Get a noisy odometry measurement from the true odometry.

        Args:
            movement (SEPose): the movement performed by the robot

        Returns:
            OdomMeasurement: the noisy measurement of the movement
        """
        pass


class GaussianOdom2DNoiseModel(OdomNoiseModel):
    """
    This is a simple Gaussian noise model for the robot odometry. This assumes
    that the noise is additive gaussian and constant in time and distance moved.
    """

    def __init__(
        self, mean: np.ndarray = np.zeros(3), covariance: np.ndarray = np.eye(3) / 1e8
    ) -> None:
        """Initializes the 2D gaussian additive noise model

        Args:
            mean (np.ndarray, optional): The mean of the noise. Generally
                zero-mean. Entries are [x, y, theta]. Defaults to np.zeros(3).
            covariance (np.ndarray, optional): The covariance matrix
                with entries corresponding to the mean vector. Defaults to
                np.eye(3)/1e8.
        """
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (3, 3)
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (3,)
        self._mean = mean
        self._covariance = covariance

    def __str__(self):
        return (
            f"Gaussian Additive Odometry\n"
            + f"Covariance: {self._covariance.flatten()}\n"
            + f"Mean: {self._mean}\n"
        )

    @property
    def covariance(self):
        return self._covariance

    @property
    def mean(self):
        return self._mean

    def get_odometry_measurement(self, movement: SE2Pose) -> OdomMeasurement2D:
        """Takes the groundtruth movement performed and then perturbs it by
        transformation randomly sampled from a Gaussian distribution and passed
        through the exponential map

        Note: we get a little hacky with the frame naming just to allow for
        multiplication of all of these noises

        Args:
            movement (SE2Pose): the true movement performed by the robot

        Returns:
            SE2Pose: A noisy measurement of the movement passed in
        """
        assert isinstance(movement, SE2Pose)

        # this is the constant component from the gaussian noise
        mean_offset = SE2Pose.by_exp_map(
            self._mean, local_frame="temp", base_frame=movement.local_frame
        )

        # this is the random component from the gaussian noise
        noise_sample = np.random.multivariate_normal(np.zeros(3), self._covariance)
        noise_offset = SE2Pose.by_exp_map(
            noise_sample, local_frame=movement.local_frame, base_frame="temp"
        )

        # because we're in 2D rotations commute so we don't need to think about
        # the order of operations???
        noisy_odom_measurement = movement * mean_offset * noise_offset
        return OdomMeasurement2D(
            movement, noisy_odom_measurement, self._mean, self._covariance
        )

class GaussianOdom3DNoiseModel(OdomNoiseModel):
    """
    This is a simple Gaussian noise model for the robot odometry. This assumes
    that the noise is additive gaussian and constant in time and distance moved.
    """

    def __init__(
        self, mean: np.ndarray = np.zeros(6), covariance: np.ndarray = np.eye(6) / 1e8
    ) -> None:
        """Initializes the 3D gaussian additive noise model

        Args:
            mean (np.ndarray, optional): The mean of the noise. Generally
                zero-mean. Entries are [x, y, theta]. Defaults to np.zeros(3).
            covariance (np.ndarray, optional): The covariance matrix
                with entries corresponding to the mean vector. Defaults to
                np.eye(6)/1e8.
        """
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (6, 6)
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (6,)
        self._mean = mean
        self._covariance = covariance

    def __str__(self):
        return (
            f"Gaussian Additive Odometry\n"
            + f"Covariance: {self._covariance.flatten()}\n"
            + f"Mean: {self._mean}\n"
        )

    @property
    def covariance(self):
        return self._covariance

    @property
    def mean(self):
        return self._mean

    def get_odometry_measurement(self, movement: SE3Pose) -> OdomMeasurement3D:
        """Takes the groundtruth movement performed and then perturbs it by
        transformation randomly sampled from a Gaussian distribution and passed
        through the exponential map

        Note: we get a little hacky with the frame naming just to allow for
        multiplication of all of these noises

        Args:
            movement (SE2Pose): the true movement performed by the robot

        Returns:
            SE2Pose: A noisy measurement of the movement passed in
        """
        assert isinstance(movement, SE3Pose)

        # this is the constant component from the gaussian noise
        mean_offset = SE3Pose.by_exp_map(
            self._mean, local_frame="temp", base_frame=movement.local_frame
        )

        # this is the random component from the gaussian noise
        noise_sample = np.random.multivariate_normal(np.zeros(6), self._covariance)
        noise_offset = SE3Pose.by_exp_map(
            noise_sample, local_frame=movement.local_frame, base_frame="temp"
        )

        # What are the order of operations on the multiplication of these poses?
        noisy_odom_measurement = movement * mean_offset * noise_offset
        return OdomMeasurement3D(
            movement, noisy_odom_measurement, self._mean, self._covariance
        )