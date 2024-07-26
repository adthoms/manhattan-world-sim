from manhattan.geometry.Elements import SEPose, SE2Pose, SE3Pose
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, overload
from py_factor_graph.utils.matrix_utils import get_measurement_precisions_from_covariance_matrix


class OdomMeasurement(ABC):
    """
    This class represents an odometry measurement.
    """

    def __init__(
        self,
        true_odometry: SEPose,
        measured_odometry: SEPose,
        mean_offset: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """Construct an odometry measurement

        Args:
            true_odometry (SE2Pose): the true odometry
            measured_odometry (SE2Pose): the measured odometry
        """
        assert isinstance(true_odometry, SEPose)
        assert isinstance(measured_odometry, SEPose)
        assert isinstance(mean_offset, np.ndarray)
        assert isinstance(covariance, np.ndarray)

        self._true_odom = true_odometry
        self._measured_odom = measured_odometry
        self._mean_offset = mean_offset
        self._covariance = covariance

    def __str__(self):
        line = "Odom Measurement\n"
        line += f"True Odometry: {self._true_odom}\n"
        line += f"Measured Odometry: {self._measured_odom}\n"
        line += f"Mean Offset: {self._mean_offset}\n"
        line += f"Covariance: {self._covariance.flatten()}"
        return line

    @property
    def local_frame(self) -> str:
        return self._measured_odom.local_frame

    @property
    def base_frame(self) -> str:
        return self._measured_odom.base_frame

    @property
    def true_odom(self) -> SEPose:
        """Get the true odometry"""
        return self._true_odom

    @property
    def measured_odom(self) -> SEPose:
        """Get the measured odometry"""
        return self._measured_odom

    @property
    def mean_offset(self) -> np.ndarray:
        """Get the mean offset"""
        return self._mean_offset

    @property
    def covariance(self) -> np.ndarray:
        """Get the covariance"""
        return self._covariance

    @property
    def delta_x(self) -> float:
        """
        returns the delta x in the measurement model
        """
        return self.measured_odom.x

    @property
    def delta_y(self) -> float:
        """
        returns the delta y in the measurement model
        """
        return self.measured_odom.y
    
    @property
    def delta_z(self) -> float:
        """
        returns the delta z in the measurement model
        """
        return self.measured_odom.z

    @property
    def delta_angles(self) -> Union[float, Tuple[float, float, float]]:
        """
        returns the delta angles in the measurement model
        """
        return self.measured_odom.rot.angles

    # Is the calculation of the translation/rotation precision from the covariance matrix 
    # the same between a Loop Closure Measurement and an Odometry Measurement?
    
    @property
    @abstractmethod
    def translation_precision(self) -> float:
        pass

    @property
    @abstractmethod
    def rotation_precision(self) -> float:
        pass

class OdomMeasurement2(OdomMeasurement):
    """
    This class represents a 2D odometry measurement.
    """

    def __init__(
        self,
        true_odometry: SE2Pose,
        measured_odometry: SE2Pose,
        mean_offset: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """Construct an odometry measurement

        Args:
            true_odometry (SE2Pose): the true odometry
            measured_odometry (SE2Pose): the measured odometry
        """
        assert isinstance(true_odometry, SE2Pose)
        assert isinstance(measured_odometry, SE2Pose)
        assert isinstance(mean_offset, np.ndarray)
        assert mean_offset.shape == (3,)
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (3, 3)

        super().__init__(true_odometry, measured_odometry, mean_offset, covariance)

    @property
    def translation_precision(self) -> float:
        return get_measurement_precisions_from_covariance_matrix(self.covariance, matrix_dim=3)[0]

    @property
    def rotation_precision(self) -> float:
        return get_measurement_precisions_from_covariance_matrix(self.covariance, matrix_dim=3)[1]

class OdomMeasurement3(OdomMeasurement):
    """
    This class represents a 3D odometry measurement.
    """

    def __init__(
        self,
        true_odometry: SE3Pose,
        measured_odometry: SE3Pose,
        mean_offset: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """Construct an odometry measurement

        Args:
            true_odometry (SE2Pose): the true odometry
            measured_odometry (SE2Pose): the measured odometry
        """
        assert isinstance(true_odometry, SE3Pose)
        assert isinstance(measured_odometry, SE3Pose)
        assert isinstance(mean_offset, np.ndarray)
        assert mean_offset.shape == (6,)
        assert isinstance(covariance, np.ndarray)
        assert covariance.shape == (6, 6)

        super().__init__(true_odometry, measured_odometry, mean_offset, covariance)

    @property
    def translation_precision(self) -> float:
        return get_measurement_precisions_from_covariance_matrix(self.covariance, matrix_dim=6)[0]

    @property
    def rotation_precision(self) -> float:
        return get_measurement_precisions_from_covariance_matrix(self.covariance, matrix_dim=6)[1]