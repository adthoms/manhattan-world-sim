import numpy as np
import math
from abc import ABC, abstractmethod
from enum import Enum
from liegroups.numpy import SO2, SO3
from liegroups.numpy import SE2, SE3
from typing import List, Tuple, Union, Optional, overload


_TRANSLATION_TOLERANCE = 1e-6  # m
_ROTATION_TOLERANCE = 1e-9  # rad


class DIM(Enum):
    TWO = 2
    THREE = 3


def check_compatible_types(obj1, obj2):
    assert (
        (type(obj1), type(obj2)) == (SE2Pose, Point2)
        or (type(obj1), type(obj2)) == (SE3Pose, Point3)
        or (type(obj1), type(obj2)) == (Rot2, Point2)
        or (type(obj1), type(obj2)) == (Rot3, Point3)
    )


def check_same_types(obj1, obj2):
    assert (
        (type(obj1), type(obj2)) == (SE2Pose, SE2Pose)
        or (type(obj1), type(obj2)) == (SE3Pose, SE3Pose)
        or (type(obj1), type(obj2)) == (Rot2, Rot2)
        or (type(obj1), type(obj2)) == (Rot3, Rot3)
        or (type(obj1), type(obj2)) == (Point2, Point2)
        or (type(obj1), type(obj2)) == (Point3, Point3)
    )


def none_to_zero(x: Optional[float]) -> float:
    return 0.0 if x is None else x


def wrap_angle_to_pipi(theta: float):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class Point(ABC):
    def __init__(self, dim: DIM, x: float, y: float, z: float, frame: str) -> None:
        """
        Abstract base class for 2D and 3D Point classes.

        Args:
            dim (int): the point dimension.
            x (float): the x coordinate.
            y (float): the y coordinate.
            z (float): the z coordinate.
            frame (str): the frame of this point.
        """
        assert isinstance(dim, DIM)
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
        assert isinstance(frame, str)

        self._dim = dim
        self._x = x
        self._y = y
        self._z = z
        self._frame = frame

    @staticmethod
    def dist(x1: "Point", x2: "Point") -> float:
        """
        Calculates the Euclidean distance between two points.

        Args:
            x1 (Point): the coordinates of the first point.
            x2 (Point): the coordinates of the second point.

        Returns:
            float: the Euclidean distance between the two points.
        """
        assert isinstance(x1, Point)
        assert isinstance(x2, Point)
        check_same_types(x1, x2)
        return np.linalg.norm(x1.array - x2.array)

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        assert isinstance(x, float)
        self._x = x

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        assert isinstance(y, float)
        self._y = y

    @property
    def frame(self) -> str:
        return self._frame

    @property
    def norm(self) -> float:
        """
        Calculates the Euclidean norm of the vector represented by this point.

        Returns:
            float: The Euclidean norm of the vector.
        """
        return np.linalg.norm(self.array)

    @property
    @abstractmethod
    def array(self) -> np.ndarray:
        """
        Returns an array containing the coordinates of this point.

        Returns:
            np.ndarray: the coordinates of this point.
        """
        pass

    @classmethod
    @abstractmethod
    def by_array(cls, other, frame: str) -> "Point":
        """
        Creates a Point object from an array.

        Args:
          other : an array-like object representing the coordinates of the point.
          frame (str): the frame of reference for the point.

        Returns:
          Point: a Point object.
        """
        pass

    @abstractmethod
    def copy(self) -> "Point":
        """
        Creates a deep copy of this point.

        Returns:
            Point: a deep copy of this point.
        """
        pass

    @abstractmethod
    def copyInverse(self) -> "Point":
        """
        Creates a deep copy of this point and negate coordinates.

        Returns:
            Point: a deep copy of this point with negated coordinates.
        """
        pass

    def distance(self, other) -> float:
        """
        Calculates the Euclidean distance between two points.

        Parameters:
            other (Point): the other point to calculate the distance to.

        Returns:
            float: the Euclidean distance between the two points.
        """
        assert isinstance(other, type(self))
        assert self.frame == other.frame
        return np.linalg.norm(self.array - other.array)

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other: float):
        pass

    @abstractmethod
    def __rmul__(self, other: float):
        pass

    @abstractmethod
    def __truediv__(self, other: Union[int, float]):
        pass

    def __iadd__(self, other):
        assert isinstance(other, type(self))
        assert self.frame == other.frame
        self._x += other._x
        self._y += other._y
        self._z += other._z
        return self

    def __isub__(self, other):
        assert isinstance(other, type(self))
        assert self.frame == other.frame
        self._x -= other._x
        self._y -= other._y
        self._z -= other._z
        return self

    def __imul__(self, other: float):
        assert np.isscalar(other)
        self._x *= other
        self._y *= other
        self._z *= other
        return self

    def __itruediv__(self, other: Union[int, float]):
        assert np.isscalar(other)
        if other == 0.0:
            raise ValueError("Cannot divide by zeros.")
        self._x /= other
        self._y /= other
        self._z /= other
        return self

    def __neg__(self):
        return self.copyInverse()

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return (
                np.allclose(self.array, other.array, _TRANSLATION_TOLERANCE)
                and self.frame == other.frame
            )
        return False

    def __hash__(self) -> int:
        return hash((self._dim, self._x, self._y, self._z, self.frame))


class Point2(Point):
    def __init__(self, x: float, y: float, frame: str) -> None:
        super().__init__(DIM.TWO, x, y, 0.0, frame)

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @classmethod
    def by_array(
        cls, other: Union[List[float], Tuple[float, float], np.ndarray], frame: str
    ) -> "Point2":
        return cls(other[0], other[1], frame)

    def copy(self) -> "Point2":
        return Point2(self.x, self.y, self.frame)

    def copyInverse(self) -> "Point2":
        return Point2(-self.x, -self.y, self.frame)

    def __add__(self, other: "Point2") -> "Point2":
        assert isinstance(other, Point2)
        assert self.frame == other.frame
        return Point2(self.x + other.x, self.y + other.y, self.frame)

    def __sub__(self, other: "Point2") -> "Point2":
        assert isinstance(other, Point2)
        assert self.frame == other.frame
        return Point2(self.x - other.x, self.y - other.y, self.frame)

    def __mul__(self, other: float) -> "Point2":
        assert np.isscalar(other)
        return Point2(self.x * other, self.y * other, self.frame)

    def __rmul__(self, other: float) -> "Point2":
        assert np.isscalar(other)
        return Point2(self.x * other, self.y * other, self.frame)

    def __truediv__(self, other: Union[int, float]) -> "Point2":
        assert np.isscalar(other)
        if other == 0.0:
            raise ValueError("Cannot divide by zeros.")
        return Point2(self.x / other, self.y / other, self.frame)

    def __str__(self) -> str:
        return f"Point2[x: {self.x} y: {self.y}, frame: {self.frame}]"


class Point3(Point):
    def __init__(self, x: float, y: float, z: float, frame: str) -> None:
        super().__init__(DIM.THREE, x, y, z, frame)

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, z: float) -> None:
        assert isinstance(z, float)
        self._z = z

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def by_array(
        cls,
        other: Union[List[float], Tuple[float, float, float], np.ndarray],
        frame: str,
    ) -> "Point3":
        return cls(other[0], other[1], other[2], frame)

    def copy(self) -> "Point3":
        return Point3(self.x, self.y, self.z, self.frame)

    def copyInverse(self) -> "Point3":
        return Point3(-self.x, -self.y, -self.z, self.frame)

    def __add__(self, other: "Point3") -> "Point3":
        assert isinstance(other, Point3)
        assert self.frame == other.frame
        return Point3(self.x + other.x, self.y + other.y, self.z + other.z, self.frame)

    def __sub__(self, other: "Point3") -> "Point3":
        assert isinstance(other, Point3)
        assert self.frame == other.frame
        return Point3(self.x - other.x, self.y - other.y, self.z - other.z, self.frame)

    def __mul__(self, other: float) -> "Point3":
        assert np.isscalar(other)
        return Point3(self.x * other, self.y * other, self.z * other, self.frame)

    def __rmul__(self, other: float) -> "Point3":
        return Point3(self.x * other, self.y * other, self.z * other, self.frame)

    def __truediv__(self, other: Union[int, float]) -> "Point3":
        assert np.isscalar(other)
        if other == 0.0:
            raise ValueError("Cannot divide by zeros.")
        return Point3(self.x / other, self.y / other, self.z / other, self.frame)

    def __str__(self) -> str:
        return f"Point3[x: {self.x} y: {self.y} z: {self.z}, frame: {self.frame}]"


class Rot(ABC):
    def __init__(self, local_frame: str, base_frame: str) -> None:
        """
        Abstract base class for 2D and 3D rotation classes.

        Args:
            local_frame (str): the local frame of the rotation.
            base_frame (str): the base frame of the rotation.
        """
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)

        self._SO = None
        self._local_frame = local_frame
        self._base_frame = base_frame

    @staticmethod
    def dist(x1: "Rot", x2: "Rot") -> float:
        """
        Calculates the chordal distance between two rotations.

        Args:
            x1 (Rot2): the first Rot object.
            x2 (Rot2): the second Rot object.

        Returns:
            float: the chordal distance between x1 and x2.
        """
        assert isinstance(x1, Rot)
        assert isinstance(x2, Rot)
        check_same_types(x1, x2)
        return np.linalg.norm((x1.copyInverse() * x2).log_map)

    @property
    def local_frame(self) -> str:
        return self._local_frame

    @property
    def base_frame(self) -> str:
        return self._base_frame

    @property
    def lie_group(self) -> Union[SO2, SO3]:
        """
        Returns the underlying SO(n) group representing the rotation. For 2D n=2, for 3D n=3.

        Returns:
            Union[SO2, SO3]: the SO(n) group.
        """
        return self._SO

    @property
    @abstractmethod
    def angles(self) -> Union[float, Tuple[float, float, float]]:
        """
        Returns the angular representation of the rotation in radians. For 2D rotations, theta is returned. For 3D rotations, (roll, pitch, yaw) is returned.

        Returns:
            Union[float, Tuple[float, float, float]]: angles representing the Rot object.
        """
        pass

    @property
    def matrix(self) -> np.ndarray:
        """
        Returns the matrix representation of the rotation.

        Returns:
            np.ndarray: a 2D array representing the rotation matrix.
        """
        return self.lie_group.as_matrix()

    @property
    def log_map(self) -> np.ndarray:
        """
        Return the log map of the rotation.

        Returns:
            np.ndarray: the log map of the rotation.
        """
        return self.lie_group.log()

    @classmethod
    @abstractmethod
    def by_array(cls, angles: np.ndarray, local_frame: str, base_frame: str):
        """
        Creates a Rot object from an array of angles with a specified local and base frame.

        Args:
            angles (np.array): an array of angles representing the rotation.
            local_frame (str): the local frame of the rotation.
            base_frame (str): the base frame of the rotation.

        Returns:
            Rot: the Rot object.
        """
        pass

    @classmethod
    @abstractmethod
    def by_matrix(cls, matrix: np.ndarray, local_frame: str, base_frame: str):
        """
        Creates a Rot object from a rotation matrix.

        Args:
            matrix (np.ndarray): the rotation matrix.
            local_frame (str): the local frame of the rotation.
            base_frame (str): the base frame of the rotation.

        Returns:
            Rot: the Rot object.
        """
        pass

    @classmethod
    @abstractmethod
    def by_exp_map(cls, vector: np.ndarray, local_frame: str, base_frame: str):
        """
        Creates a Rot object from a rotation vector with a specified local and base frame.

        Args:
            vector (np.array): the rotation vector.
            local_frame (str): the local frame of the rotation.
            base_frame (str): the base frame of the rotation.

        Returns:
            Rot: the Rot object.
        """
        pass

    @abstractmethod
    def copy(self):
        """
        Returns deep copy of this rotation.

        Returns:
            Rot: a deep copy of this rotation.
        """
        pass

    @abstractmethod
    def copyInverse(self):
        """
        Returns deep copy of the inverse of this rotation.

        Returns:
            Rot: a deep copy of the inverse of this rotation.
        """
        pass

    @abstractmethod
    def rotate_point(self, local_pt):
        """
        Rotates a point in the local frame to the base frame.

        Args:
            local_pt (Point): the given point in the local frame.

        Returns:
            Point: the rotated point.
        """
        pass

    @abstractmethod
    def unrotate_point(self, base_frame_pt):
        """
        Rotates a point in the base frame to the local frame.

        Args:
            base_frame_pt (Point): a point in the base frame.

        Returns:
            Point: the point in the local frame.
        """
        pass

    @abstractmethod
    def bearing_to_local_frame_point(
        self, local_pt
    ) -> Union[float, Tuple[float, float]]:
        """
        Returns the bearing of the local frame point in radians.

        Args:
            local_frame_pt (Point): the point in the local frame.

        Returns:
            Union[float, Tuple[float, float]]: the bearing of the local frame point in radians.
        """
        pass

    @abstractmethod
    def bearing_to_base_frame_point(
        self, base_frame_pt
    ) -> Union[float, Tuple[float, float]]:
        """
        Gets the bearing to a given point expressed in the same base frame as this rotation.

        Args:
            base_frame_pt (Point): the given point.

        Returns:
            Union[float, Tuple[float, float]]: the bearing in radians.
        """
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            same_frames = (
                self.local_frame == other.local_frame
                and self.base_frame == other.base_frame
            )
            angles_close = np.allclose(
                np.array(self.angles),
                np.array(other.angles),
                _ROTATION_TOLERANCE,
            )
            return same_frames and angles_close
        return False

    def __hash__(self) -> int:
        return hash((self.lie_group, self.local_frame, self.base_frame))


class Rot2(Rot):
    def __init__(self, theta: float, local_frame: str, base_frame: str) -> None:
        super().__init__(local_frame, base_frame)
        """
        Creates a 2D rotation in radians with a specified base and local frame.

        Args:
            theta (float): the rotation in radians.
            local_frame (str): the local frame of the rotation.
            base_frame (str): the base frame of the rotation.
        """
        assert isinstance(theta, float)
        self._SO = SO2.from_angle(none_to_zero(theta))

    @property
    def angles(self) -> float:
        return self.lie_group.to_angle()

    @property
    def theta(self) -> float:
        return wrap_angle_to_pipi(self.angles)

    @theta.setter
    def theta(self, theta: float):
        assert np.isscalar(theta)
        self._SO = SO2.from_angle(theta)

    @classmethod
    def by_array(cls, angles: np.ndarray, local_frame: str, base_frame: str) -> "Rot2":
        assert isinstance(angles, np.ndarray)
        assert len(angles) == 1
        theta = none_to_zero(angles[0])
        return cls(theta, local_frame, base_frame)

    @classmethod
    def by_matrix(cls, matrix: np.ndarray, local_frame: str, base_frame: str) -> "Rot2":
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)
        theta = SO2.from_matrix(matrix).to_angle()
        return cls(theta, local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "Rot2":
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 1
        return cls(vector[0], local_frame, base_frame)

    def copy(self) -> "Rot2":
        return Rot2(self.theta, self.local_frame, self.base_frame)

    def copyInverse(self) -> "Rot2":
        return Rot2(-self.theta, self.base_frame, self.local_frame)

    def rotate_point(self, local_pt: Point2) -> Point2:
        assert isinstance(local_pt, Point2)
        assert self.local_frame == local_pt.frame
        return self * local_pt

    def unrotate_point(self, base_frame_pt: Point2) -> Point2:
        assert isinstance(base_frame_pt, Point2)
        assert self.base_frame == base_frame_pt.frame
        return self.copyInverse() * base_frame_pt

    def bearing_to_local_frame_point(self, local_pt: Point2) -> float:
        assert isinstance(local_pt, Point2)
        assert self.local_frame == local_pt.frame
        return math.atan2(local_pt.y, local_pt.x)

    def bearing_to_base_frame_point(self, base_frame_pt: Point2) -> float:
        assert isinstance(base_frame_pt, Point2)
        assert self.base_frame == base_frame_pt.frame
        local_pt = self.unrotate_point(base_frame_pt)
        return math.atan2(local_pt.y, local_pt.x)

    @overload
    def __mul__(self, other: "Rot2") -> "Rot2":
        pass

    @overload
    def __mul__(self, other: Point2) -> Point2:
        pass

    def __mul__(self, other: Union["Rot2", Point2]) -> Union["Rot2", Point2]:
        assert isinstance(other, (Rot2, Point2))
        if isinstance(other, Rot2):
            assert self.local_frame == other.base_frame
            theta = self.lie_group.dot(other.lie_group).to_angle()
            return Rot2(theta, self.local_frame, other.base_frame)
        elif isinstance(other, Point2):
            assert self.local_frame == other.frame
            point_rotated = self.lie_group.dot(other.array)
            x, y = point_rotated[0], point_rotated[1]
            return Point2(x, y, self.base_frame)
        else:
            raise ValueError("Not a valid Point2 or Rot2 type to multiply.")

    def __str__(self) -> str:
        return f"Rot2[theta: {self.theta:.3f}, local_frame: {self.local_frame}, base_frame: {self.base_frame}]"


class Rot3(Rot):
    def __init__(
        self, roll: float, pitch: float, yaw: float, local_frame: str, base_frame: str
    ) -> None:
        super().__init__(local_frame, base_frame)

        """
        Creates a 3D rotation in radians with a specified base and local frame.

        Args:
            roll (float): the roll rotation in radians.
            pitch (float): the pitch rotation in radians.
            yaw (float): the yaw rotation in radians.
            local_frame (str): the local frame of the rotation.
            base_frame (str): the base frame of the rotation.
        """
        assert isinstance(roll, float)
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)
        self._SO = SO3.from_rpy(roll, pitch, yaw)

    @property
    def roll(self) -> float:
        return self.lie_group.to_rpy()[0]

    @property
    def pitch(self) -> float:
        return self.lie_group.to_rpy()[1]

    @property
    def yaw(self) -> float:
        return self.lie_group.to_rpy()[2]

    @property
    def angles(self) -> Tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)

    @classmethod
    def by_array(cls, angles: np.ndarray, local_frame: str, base_frame: str) -> "Rot3":
        assert isinstance(angles, np.ndarray)
        assert len(angles) == 3
        roll = none_to_zero(angles[0])
        pitch = none_to_zero(angles[1])
        yaw = none_to_zero(angles[2])
        return cls(roll, pitch, yaw, local_frame, base_frame)

    @classmethod
    def by_matrix(cls, matrix: np.ndarray, local_frame: str, base_frame: str) -> "Rot3":
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        roll, pitch, yaw = SO3.from_matrix(matrix).to_rpy()
        return cls(roll, pitch, yaw, local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "Rot3":
        assert isinstance(vector, np.array)
        assert len(vector) == 3
        roll, pitch, yaw = SO3.exp(vector).to_rpy()
        return cls(roll, pitch, yaw, local_frame, base_frame)

    def copy(self) -> "Rot3":
        return Rot3(self.roll, self.pitch, self.yaw, self.local_frame, self.base_frame)

    def copyInverse(self) -> "Rot3":
        roll, pitch, yaw = self.lie_group.inv().to_rpy()
        return Rot3(roll, pitch, yaw, self.base_frame, self.local_frame)

    def rotate_point(self, local_pt: Point3) -> Point3:
        assert isinstance(local_pt, Point3)
        assert self.local_frame == local_pt.frame
        return self * local_pt

    def unrotate_point(self, base_frame_pt: Point3) -> Point3:
        assert isinstance(base_frame_pt, Point3)
        assert self.base_frame == base_frame_pt.frame
        return self.copyInverse() * base_frame_pt

    def bearing_to_local_frame_point(self, local_pt: Point3) -> Tuple[float, float]:
        assert isinstance(local_pt, Point3)
        assert self.local_frame == local_pt.frame
        return math.atan2(local_pt.y, local_pt.x), math.atan2(local_pt.z, local_pt.x)

    def bearing_to_base_frame_point(self, base_frame_pt: Point3) -> Tuple[float, float]:
        assert isinstance(base_frame_pt, Point3)
        assert self.base_frame == base_frame_pt.frame
        local_pt = self.unrotate_point(base_frame_pt)
        return math.atan2(local_pt.y, local_pt.x), math.atan2(local_pt.z, local_pt.x)

    @overload
    def __mul__(self, other: "Rot3") -> "Rot3":
        pass

    @overload
    def __mul__(self, other: Point3) -> Point3:
        pass

    def __mul__(self, other: Union["Rot3", Point3]) -> Union["Rot3", Point3]:
        assert isinstance(other, (Rot3, Point3))
        if isinstance(other, Rot3):
            assert self.local_frame == other.base_frame
            roll, pitch, yaw = self.lie_group.dot(other.lie_group).to_rpy()
            return Rot3(roll, pitch, yaw, self.local_frame, other.base_frame)
        elif isinstance(other, Point3):
            assert self.local_frame == other.frame
            point_rotated = self.lie_group.dot(other.array)
            x, y, z = point_rotated[0], point_rotated[1], point_rotated[2]
            return Point3(x, y, z, self.base_frame)
        else:
            raise ValueError("Not a valid Point3 or Rot3 type to multiply.")

    def __str__(self) -> str:
        return f"Rot3[roll: {self.roll:.3f}, pitch: {self.pitch:.3f}, yaw: {self.yaw:.3f}, local_frame: {self.local_frame}, base_frame: {self.base_frame}]"


class SEPose(ABC):
    def __init__(self, local_frame: str, base_frame: str) -> None:
        """
        Abstract base class for SE(n) pose classes. For 2D n=2, for 3D n=3.

        Args:
            local_frame (str): the local frame of the pose.
            base_frame (str): the base frame of the pose.
        """
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)
        assert not local_frame == base_frame

        self._SE = None
        self._local_frame = local_frame
        self._base_frame = base_frame

    @staticmethod
    def dist(x1: "SEPose", x2: "SEPose") -> float:
        """
        Calculates the chordal distance between two poses. The chordal distance is calculated as the Frobenius norm of the difference between x1's and x2's homogenous transformation matrices.

        Args:
            x1 (SEPose): The first pose object.
            x2 (SEPose): The second pose object.

        Raises:
            AssertionError: If the shapes of x1 and x2 are not equal.
            AssertionError: If the number of rows in x1 and the number of columns in x2 are not equal.
            AssertionError: If the number of rows in x1 is not 2 or 3.

        Returns:
            float: The chordal distance between x1 and x2.
        """
        assert isinstance(x1, SEPose)
        assert isinstance(x2, SEPose)
        check_same_types(x1, x2)
        T1 = x1.matrix()
        T2 = x2.matrix()
        return np.linalg.norm(T1 - T2, ord="fro")

    @property
    def se_group(self):
        return self._SE

    @property
    def local_frame(self) -> str:
        return self._local_frame

    @property
    def base_frame(self) -> str:
        return self._base_frame

    @property
    @abstractmethod
    def rot(self):
        """
        Returns a Rot object representing the rotation of this pose.

        Returns:
            Rot: The Rot object.
        """
        pass

    @property
    @abstractmethod
    def point(self):
        """
        Returns a Point object representing the translation of this pose.

        Returns:
            Point: The Point object.
        """
        pass

    @property
    def matrix(self) -> np.ndarray:
        """
        Returns the matrix representation of the pose object.

        Returns:
            np.ndarray: The matrix representation of the pose object.
        """
        return self.se_group.as_matrix()

    @property
    @abstractmethod
    def array(self) -> np.ndarray:
        """
        Returns the array representation of the pose object.

        Returns:
            np.ndarray: The array representation of the pose object.
        """
        pass

    @property
    def log_map(self) -> np.ndarray:
        """
        Computes the logarithmic map of the current pose object.

        Returns:
            np.ndarray: The logarithmic map of the current pose object.
        """
        return self.se_group.log()

    @classmethod
    @abstractmethod
    def by_point_and_rotation(cls, point, rot, local_frame: str, base_frame: str):
        """
        Creates a pose object from Point and Rot objects.

        Args:
            point (Point): The Point object.
            rot (Rot): The Rot object.
            local_frame (str): The local frame.
            base_frame (str): The base frame.

        Returns:
            SEPose: The Pose object.
        """
        pass

    @classmethod
    @abstractmethod
    def by_matrix(cls, matrix: np.ndarray, local_frame: str, base_frame: str):
        """
        Creates a pose object from a homogenous transformation matrix.

        Args:
            matrix (np.ndarray): the homogenous transformation matrix.
            local_frame (str): the local frame of the matrix.
            base_frame (str): the base frame of the matrix.

        Returns:
            SEPose: The Pose object.
        """
        pass

    @classmethod
    @abstractmethod
    def by_exp_map(cls, vector: np.ndarray, local_frame: str, base_frame: str):
        """
        Creates a pose object from an exponential map vector.

        Args:
            vector (np.array): the exponential map vector.
            local_frame (str): the local frame of the exponential map vector.
            base_frame (str): the base frame of the exponential map vector.

        Returns:
            SEPose: The Pose object.
        """
        pass

    @abstractmethod
    def copy(self):
        """
        Returns deep copy of this pose.

        Returns:
            SEPose: A deep copy of this pose.
        """
        pass

    @abstractmethod
    def copyInverse(self):
        """
        Returns deep copy of the inverse of this pose.

        Returns:
            SEPose: A deep copy of the inverse of this pose.
        """
        pass

    def range_and_bearing_to_point(
        self, point: Point
    ) -> Union[Tuple[float, float], Tuple[float, Tuple[float, float]]]:
        """
        Returns the range and bearing from this pose to the point. For the 2D case, bearing is measured from the (x,y) coordinate of the point to give a single bearing. For the 3D case, bearing is measured from the (x,y) and (x,z) coordinates of the point to give two bearings.

        Args:
            point (Point): the point to measure to.

        Returns:
            Union[Tuple[float, float], Tuple[float, Tuple[float, float]]]: (range, bearing).
        """
        check_compatible_types(self, point)
        diff = point - self.point
        dist = diff.norm
        bearing = self.rot.bearing_to_base_frame_point(diff)
        return dist, bearing

    def transform_to(self, other: "SEPose") -> "SEPose":
        """
        Returns the coordinate frame of the other pose in the coordinate frame of this pose.

        Args:
            other (SEPose): the other pose which we want the coordinate frame with respect to.

        Returns:
            SEPose: the coordinate frame of this pose with respect to the given pose.
        """
        check_same_types(self, other)
        assert self.base_frame == other.base_frame
        assert not self.local_frame == other.local_frame
        return self.copyInverse() * other

    def transform_local_point_to_base(self, local_point: Point) -> Point:
        """
        Returns a point expressed in local frame of self in the base frame of self.

        Args:
            local_point (Point): the point expressed in the local frame of self.

        Returns:
            Point: a point expressed in the base frame of self.
        """
        check_compatible_types(self, local_point)
        return self * local_point

    def transform_base_point_to_local(self, base_point: Point) -> Point:
        """
        Returns a point expressed in base frame of self in the local frame of self.

        Args:
            base_point (Point): the point expressed in the base frame of self.

        Returns:
            Point: a point expressed in the local frame of self.
        """
        check_compatible_types(self, base_point)
        return self.copyInverse() * base_point

    def distance_to_pose(self, other: "SEPose") -> float:
        """
        Returns the distance between this pose and another pose.

        Args:
            other (SEPose): the other pose.

        Returns:
            float: the distance between this pose and the other pose.
        """
        check_same_types(self, other)
        cur_position = self.point
        other_position = other.point
        assert cur_position.frame == other_position.frame

        dist = cur_position.distance(other_position)
        assert isinstance(dist, float)
        return dist

    def __mul__(self, other):
        check_compatible_types(self, other)
        if isinstance(other, SEPose):
            assert self.local_frame == other.base_frame
            T = self.se_group.dot(other.se_group).as_matrix()
            return self.__class__.by_matrix(
                matrix=T, local_frame=other.local_frame, base_frame=self.base_frame
            )
        if isinstance(other, Point):
            assert self.local_frame == other.frame
            return self.rot * other + self.point

    def __imul__(self, other):
        check_same_types(self, other)
        self._SE = self.se_group.dot(other.se_group)
        self._local_frame = other.local_frame

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SEPose):
            rotation_close = np.allclose(
                np.array(self.rot.angles),
                np.array(other.rot.angles),
                _ROTATION_TOLERANCE,
            )
            translation_close = np.allclose(
                self.point.array, other.point.array, _TRANSLATION_TOLERANCE
            )
            return (
                rotation_close
                and translation_close
                and self._local_frame == other.local_frame
                and self._base_frame == other.base_frame
            )
        return False

    def __hash__(self):
        return hash((self.se_group, self._local_frame, self._base_frame))


class SE2Pose(SEPose):
    def __init__(
        self, x: float, y: float, theta: float, local_frame: str, base_frame: str
    ) -> None:
        super().__init__(local_frame, base_frame)
        """
        An SE(2) pose.

        Args:
            x (float): the x-coordinate of the pose in the base frame.
            y (float): the y-coordinate of the pose in the base frame.
            theta (float): the angle of the pose in radians in the base frame.
            local_frame (str): the local frame of the pose.
            base_frame (str): the base frame of the pose.
        """
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(theta, float)

        # create homogeneous transformation matrix
        R = SO2.from_angle(theta)
        t = np.array([x, y])
        T = np.eye(3)
        T[:2, :2] = R.as_matrix()
        T[:2, 2] = t

        # class attributes
        self._SE = SE2.from_matrix(T)
        self._local_frame = local_frame
        self._base_frame = base_frame

    @property
    def x(self) -> float:
        return self.se_group.trans[0]

    @property
    def y(self) -> float:
        return self.se_group.trans[1]

    @property
    def theta(self) -> float:
        return wrap_angle_to_pipi(self.se_group.rot.to_angle())

    @property
    def rot(self) -> Rot2:
        return Rot2(self.theta, self.local_frame, self.base_frame)

    @property
    def point(self) -> Point2:
        return Point2(self.x, self.y, self.base_frame)

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    @classmethod
    def by_point_and_rotation(
        cls, point: Point2, rot: Rot2, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        assert isinstance(point, Point2)
        assert isinstance(rot, Rot2)
        return cls(point.x, point.y, rot.theta, local_frame, base_frame)

    @classmethod
    def by_matrix(
        cls, matrix: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        point = Point2.by_array(matrix[:2, 2], local_frame)
        rot = Rot2.by_matrix(matrix[:2, :2], local_frame, base_frame)
        return SE2Pose.by_point_and_rotation(point, rot, local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        T = SE2.exp(vector)
        x = T.trans[0]
        y = T.trans[1]
        theta = T.rot.to_angle()
        return cls(x, y, theta, local_frame, base_frame)

    def copy(self) -> "SE2Pose":
        return SE2Pose(
            x=self.x,
            y=self.y,
            theta=self.theta,
            local_frame=self._local_frame,
            base_frame=self._base_frame,
        )

    def copyInverse(self) -> "SE2Pose":
        T = self.se_group.inv()
        return SE2Pose(
            x=T.trans[0],
            y=T.trans[1],
            theta=T.rot.to_angle(),
            local_frame=self._base_frame,
            base_frame=self._local_frame,
        )

    def __str__(self) -> str:
        return f"Pose2[x: {self.x:.3f}, y: {self.y:.3f}, theta: {self.theta:.3f}, local_frame: {self.local_frame}, base_frame: {self.base_frame}]"


class SE3Pose(SEPose):
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        yaw: float,
        pitch: float,
        local_frame: str,
        base_frame: str,
    ) -> None:
        super().__init__(local_frame, base_frame)
        """
        An SE(3) pose.

        Args:
            x (float): the x coordinate.
            y (float): the y coordinate.
            z (float): the z coordinate.
            roll (float): the roll rotation in radians.
            pitch (float): the pitch rotation in radians.
            yaw (float): the yaw rotation in radians.
            local_frame (str): the local frame of the rotation.
            base_frame (str): the base frame of the rotation.
        """
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
        assert isinstance(roll, float)
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)

        # create homogeneous transformation matrix
        R = SO3.from_rpy(roll, pitch, yaw)
        t = np.array([x, y, z])
        T = np.eye(4)
        T[:3, :3] = R.as_matrix()
        T[:3, 3] = t

        # class attributes
        self._SE = SE3.from_matrix(T)
        self._local_frame = local_frame
        self._base_frame = base_frame

    @property
    def x(self) -> float:
        return self.se_group.trans[0]

    @property
    def y(self) -> float:
        return self.se_group.trans[1]

    @property
    def z(self) -> float:
        return self.se_group.trans[2]

    @property
    def roll(self) -> float:
        return self.se_group.rot.to_rpy()[0]

    @property
    def pitch(self) -> float:
        return self.se_group.rot.to_rpy()[1]

    @property
    def yaw(self) -> float:
        return self.se_group.rot.to_rpy()[2]

    @property
    def rot(self) -> Rot3:
        return Rot3(self.roll, self.pitch, self.yaw, self.local_frame, self.base_frame)

    @property
    def point(self) -> Point3:
        return Point3(self.x, self.y, self.z, self.base_frame)

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])

    @classmethod
    def by_point_and_rotation(
        cls, point: Point3, rot: Rot3, local_frame: str, base_frame: str
    ) -> "SE3Pose":
        assert isinstance(point, Point3)
        assert isinstance(rot, Rot3)
        return cls(
            point.x,
            point.y,
            point.z,
            rot.roll,
            rot.pitch,
            rot.yaw,
            local_frame,
            base_frame,
        )

    @classmethod
    def by_matrix(
        cls, matrix: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE3Pose":
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (4, 4)
        point = Point3.by_array(matrix[:3, 3], local_frame)
        rot = Rot3.by_matrix(matrix[:3, :3], local_frame, base_frame)
        return SE3Pose.by_point_and_rotation(point, rot, local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE3Pose":
        assert isinstance(vector, np.array)
        assert len(vector) == 6
        T = SE3.exp(vector)
        x = T.trans[0]
        y = T.trans[1]
        z = T.trans[2]
        roll, pitch, yaw = T.rot.to_rpy()
        return cls(x, y, z, roll, pitch, yaw, local_frame, base_frame)

    def copy(self) -> "SE3Pose":
        return SE3Pose(
            x=self.x,
            y=self.y,
            z=self.z,
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
            local_frame=self._local_frame,
            base_frame=self._base_frame,
        )

    def copyInverse(self) -> "SE3Pose":
        T = self.se_group.inv()
        return SE3Pose(
            x=T.trans[0],
            y=T.trans[1],
            z=T.trans[2],
            roll=T.rot.to_rpy()[0],
            pitch=T.rot.to_rpy()[1],
            yaw=T.rot.to_rpy()[2],
            local_frame=self._base_frame,
            base_frame=self._local_frame,
        )

    def __str__(self) -> str:
        return f"Pose3[x: {self.x:.3f}, y: {self.y:.3f}, z: {self.z:.3f}, roll: {self.roll:.3f}, pitch: {self.pitch:.3f}, yaw: {self.yaw:.3f}, local_frame: {self.local_frame}, base_frame: {self.base_frame}]"
