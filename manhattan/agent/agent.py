from abc import ABC, abstractmethod
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from typing import List, Tuple, Union, Optional, overload

from manhattan.noise_models.range_noise_model import RangeNoiseModel
from manhattan.noise_models.odom_noise_model import OdomNoiseModel
from manhattan.noise_models.loop_closure_model import LoopClosureModel
from manhattan.measurement.range_measurement import RangeMeasurement
from manhattan.measurement.odom_measurement import OdomMeasurement, OdomMeasurement2, OdomMeasurement3
from manhattan.measurement.loop_closure import LoopClosure, LoopClosure2, LoopClosure3
from manhattan.geometry.Elements import Point, SEPose, Point2, SE2Pose, Point3, SE3Pose


class Agent:
    """
    This class represents a general agent. In our simulator this is either a
    robot or a stationary beacon.
    """

    def __init__(self, name: str, range_model: RangeNoiseModel) -> None:
        assert isinstance(name, str)
        assert isinstance(range_model, RangeNoiseModel)

        self._name = name
        self._range_model = range_model
        self._timestep = 0

    def get_range_measurement_to_agent(self, other_agent: "Agent") -> RangeMeasurement:
        other_loc = other_agent.position
        assert isinstance(other_loc, Point)
        cur_loc = self.position
        assert isinstance(cur_loc, Point)
        dist = cur_loc.distance(other_loc)
        measurement = self._range_model.get_range_measurement(dist, self.timestep)
        assert isinstance(measurement, RangeMeasurement)
        return measurement

    @property
    def timestep(self) -> int:
        return self._timestep

    def _increment_timestep(self) -> None:
        self._timestep += 1

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> Point:
        raise NotImplementedError("position property must be implemented")

    @abstractmethod
    def plot(self) -> None:
        pass

    def distance_to_other_agent(self, other_agent: "Agent") -> float:
        """Returns the distance between this agent and the other agent

        Args:
            other_agent (Agent): The other agent

        Returns:
            float: The distance between the two agents
        """
        assert isinstance(other_agent, Agent)

        # get distance between the points of the two agents
        cur_position = self.position
        other_position = other_agent.position
        assert cur_position.frame == other_position.frame

        dist = cur_position.distance(other_position)

        assert isinstance(dist, float)
        return dist

    def range_measurement_from_dist(
        self, dist: float, gt_measure: bool = False
    ) -> RangeMeasurement:
        """Returns the range measurement generated from a given distance

        Args:
            dist (float): the distance
            gt_measure (bool): whether the measurement is the ground truth distance

        Returns:
            RangeMeasurement: the range measurement from this distance
        """
        if gt_measure:
            return RangeMeasurement(dist, dist, 0.0, 0.1, self.timestep)

        return self._range_model.get_range_measurement(dist, self.timestep)


class Robot(Agent, ABC):
    def __init__(
        self,
        name: str,
        start_pose: SEPose,
        range_model: RangeNoiseModel,
        odometry_model: OdomNoiseModel,
        loop_closure_model: LoopClosureModel,
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(start_pose, SEPose)
        assert isinstance(range_model, RangeNoiseModel)
        assert isinstance(odometry_model, OdomNoiseModel)
        assert isinstance(loop_closure_model, LoopClosureModel)

        Agent.__init__(self, name, range_model)
        self._odom_model = odometry_model
        self._loop_closure_model = loop_closure_model
        self._pose = start_pose

    def __str__(self) -> str:
        return (
            f"Robot: {self._name}\n"
            + f"Timestep: {self._timestep}"
            + f"Groundtruth pose: {self._pose}\n"
            + f"Range model: {self._range_model}"
            + f"Odometry model: {self._odom_model}\n"
        )

    @property
    def position(self) -> Point:
        assert isinstance(self._pose, SEPose)
        assert isinstance(self._pose.point, Point)
        return self._pose.point

    @property
    def pose(self) -> SEPose:
        assert isinstance(self._pose, SEPose)
        return self._pose

    @property
    def heading(self) -> Union[float, Tuple[float, float, float]]:
        """Returns the robot's current heading in radians"""
        return self.pose.rot.angles

    @abstractmethod
    def get_loop_closure_measurement(
        self, other_pose: SEPose, measure_association: str, gt_measure: bool = False
    ) -> Union[LoopClosure2, LoopClosure3]:
        """Gets a loop closure measurement to another pose based on the Robot's
        loop closure model

        Args:
            other_pose (SEPose): the pose to measure the loop closure to
            measure_association (str): the assumed data association for the loop
                closure (which pose the closure is thought to be measured to)
            gt_measure (bool): whether the measurement is the ground truth

        Returns:
            LoopClosure: the loop closure measurement
        """

        pass

    @abstractmethod
    def move(self, transform: SEPose, gt_measure: bool) -> Union[OdomMeasurement2, OdomMeasurement3]:
        """Moves the robot by the given transform and returns a noisy odometry
        measurement of the move

        Args:
            transform (SEPose): the transform to move the robot by
            gt_measure (bool): whether the odometry measurement should be
                the true movement

        Returns:
            SEPose: the noisy measurement of the transform
        """
        pass

    @abstractmethod
    def plot(self) -> None:
        """Plots the robot's groundtruth position"""
        pass
        
class Robot2(Robot):
    def __init__(
        self,
        name: str,
        start_pose: SE2Pose,
        range_model: RangeNoiseModel,
        odometry_model: OdomNoiseModel,
        loop_closure_model: LoopClosureModel,
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(start_pose, SE2Pose)
        assert isinstance(range_model, RangeNoiseModel)
        assert isinstance(odometry_model, OdomNoiseModel)
        assert isinstance(loop_closure_model, LoopClosureModel)

        super().__init__(name, start_pose, range_model, odometry_model, loop_closure_model)
        self._odom_model = odometry_model
        self._loop_closure_model = loop_closure_model
        self._pose = start_pose

    def get_loop_closure_measurement(
        self, other_pose: SE2Pose, measure_association: str, gt_measure: bool = False
    ) -> LoopClosure2:
        """Gets a loop closure measurement to another pose based on the Robot's
        loop closure model

        Args:
            other_pose (SE2Pose): the pose to measure the loop closure to
            measure_association (str): the assumed data association for the loop
                closure (which pose the closure is thought to be measured to)
            gt_measure (bool): whether the measurement is the ground truth

        Returns:
            LoopClosure: the loop closure measurement
        """

        assert isinstance(other_pose, SE2Pose)
        if gt_measure:
            true_transform = self.pose.transform_to(other_pose)

            return LoopClosure2(
                pose_1=self.pose,
                pose_2=other_pose,
                measured_association=measure_association,
                measured_rel_pose=true_transform,
                timestamp=self.timestep,
                mean_offset=self._loop_closure_model.mean,
                covariance=self._loop_closure_model.covariance,
            )
        return self._loop_closure_model.get_relative_pose_measurement(
            self.pose, other_pose, measure_association, self.timestep
        )

    def move(self, transform: SE2Pose, gt_measure: bool) -> OdomMeasurement2:
        """Moves the robot by the given transform and returns a noisy odometry
        measurement of the move

        Args:
            transform (SEPose): the transform to move the robot by
            gt_measure (bool): whether the odometry measurement should be
                the true movement

        Returns:
            SE2Pose: the noisy measurement of the transform
        """

        assert isinstance(transform, SE2Pose)

        # move the robot
        self._pose = self._pose * transform
        self._increment_timestep()

        # if gt measure then just fake the noise and return the true transform
        # as the measurement
        if gt_measure:
            return OdomMeasurement2(
                transform,
                transform,
                self._odom_model._mean,
                self._odom_model._covariance,
            )

        # get the odometry measurement
        odom_measurement = self._odom_model.get_odometry_measurement(transform)
        return odom_measurement

    def plot(self) -> None:
        """Plots the robot's groundtruth position"""

        cur_position = self.position

        heading_tol = 1e-8
        if abs(self.heading) < heading_tol:
            return plt.plot(cur_position.x, cur_position.y, "b>", markersize=10)
        elif abs(self.heading - (np.pi / 2.0)) < heading_tol:
            return plt.plot(cur_position.x, cur_position.y, "b^", markersize=10)
        elif (
            abs(self.heading + np.pi) < heading_tol
            or abs(self.heading - np.pi) < heading_tol
        ):
            return plt.plot(cur_position.x, cur_position.y, "b<", markersize=10)
        elif abs(self.heading + (np.pi / 2.0)) < heading_tol:
            return plt.plot(cur_position.x, cur_position.y, "bv", markersize=10)
        else:
            raise NotImplementedError(f"Unhandled heading: {self.heading}")

class Robot3(Robot):
    def __init__(
        self,
        name: str,
        start_pose: SE3Pose,
        range_model: RangeNoiseModel,
        odometry_model: OdomNoiseModel,
        loop_closure_model: LoopClosureModel,
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(start_pose, SE3Pose)
        assert isinstance(range_model, RangeNoiseModel)
        assert isinstance(odometry_model, OdomNoiseModel)
        assert isinstance(loop_closure_model, LoopClosureModel)

        super().__init__(name, start_pose, range_model, odometry_model, loop_closure_model)
        self._odom_model = odometry_model
        self._loop_closure_model = loop_closure_model
        self._pose = start_pose

    def get_loop_closure_measurement(
        self, other_pose: SE3Pose, measure_association: str, gt_measure: bool = False
    ) -> LoopClosure3:
        """Gets a loop closure measurement to another pose based on the Robot's
        loop closure model

        Args:
            other_pose (SE3Pose): the pose to measure the loop closure to
            measure_association (str): the assumed data association for the loop
                closure (which pose the closure is thought to be measured to)
            gt_measure (bool): whether the measurement is the ground truth

        Returns:
            LoopClosure: the loop closure measurement
        """

        assert isinstance(other_pose, SE3Pose)
        if gt_measure:
            true_transform = self.pose.transform_to(other_pose)

            return LoopClosure3(
                pose_1=self.pose,
                pose_2=other_pose,
                measured_association=measure_association,
                measured_rel_pose=true_transform,
                timestamp=self.timestep,
                mean_offset=self._loop_closure_model.mean,
                covariance=self._loop_closure_model.covariance,
            )
        return self._loop_closure_model.get_relative_pose_measurement(
            self.pose, other_pose, measure_association, self.timestep
        )

    def move(self, transform: SE3Pose, gt_measure: bool) -> OdomMeasurement3:
        """Moves the robot by the given transform and returns a noisy odometry
        measurement of the move

        Args:
            transform (SEPose): the transform to move the robot by
            gt_measure (bool): whether the odometry measurement should be
                the true movement

        Returns:
            SE3Pose: the noisy measurement of the transform
        """

        assert isinstance(transform, SE3Pose)

        # move the robot
        self._pose = self._pose * transform
        self._increment_timestep()

        # if gt measure then just fake the noise and return the true transform
        # as the measurement
        if gt_measure:
            return OdomMeasurement3(
                transform,
                transform,
                self._odom_model._mean,
                self._odom_model._covariance,
            )

        # get the odometry measurement
        odom_measurement = self._odom_model.get_odometry_measurement(transform)
        return odom_measurement

    def plot(self) -> None:
        """Plots the robot's groundtruth position"""

        cur_position = self.position

        heading_tol = 1e-8
        plt.axes(projection='3d')
        heading_yaw = self.heading[2]

        # Only handles yaw angles of pi/2, pi, -pi/2, and 0
        if abs(heading_yaw) < heading_tol:
            return plt.plot(cur_position.x, cur_position.y, cur_position.z, "b>", markersize=10)
        elif abs(heading_yaw - (np.pi / 2.0)) < heading_tol:
            return plt.plot(cur_position.x, cur_position.y, cur_position.z, "b^", markersize=10)
        elif (
            abs(heading_yaw + np.pi) < heading_tol
            or abs(heading_yaw - np.pi) < heading_tol
        ):
            return plt.plot(cur_position.x, cur_position.y, cur_position.z, "b<", markersize=10)
        elif abs(heading_yaw + (np.pi / 2.0)) < heading_tol:
            return plt.plot(cur_position.x, cur_position.y, cur_position.z, "bv", markersize=10)
        else:
            raise NotImplementedError(f"Unhandled heading: {self.heading}")


class Beacon(Agent, ABC):
    def __init__(
        self, name: str, position: Point, range_model: RangeNoiseModel
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(position, Point)
        assert isinstance(range_model, RangeNoiseModel)

        Agent.__init__(self, name, range_model)
        self._position = position

    def __str__(self):
        return (
            f"Beacon: {self._name}\n"
            + f"Groundtruth position: {self._position}\n"
            + f"Range model: {self._range_model}\n"
        )

    @property
    def position(self) -> Point:
        return self._position

    @abstractmethod
    def plot(self) -> None:
        """Plots the beacons's groundtruth position"""
        pass

class Beacon2(Beacon):
    def __init__(
        self, name: str, position: Point2, range_model: RangeNoiseModel
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(position, Point2)
        assert isinstance(range_model, RangeNoiseModel)

        super().__init__(name, position, range_model)
        self._position = position

    def plot(self) -> None:
        """Plots the beacons's groundtruth position"""
        cur_position = self.position
        return plt.plot(cur_position.x, cur_position.y, "g*", markersize=10)
    
class Beacon3(Beacon):
    def __init__(
        self, name: str, position: Point3, range_model: RangeNoiseModel
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(position, Point3)
        assert isinstance(range_model, RangeNoiseModel)

        super().__init__(name, position, range_model)
        self._position = position

    def plot(self) -> None:
        """Plots the beacons's groundtruth position"""
        cur_position = self.position
        plt.axes(projection='3d')
        return plt.plot(cur_position.x, cur_position.y, cur_position.z, "g*", markersize=10)