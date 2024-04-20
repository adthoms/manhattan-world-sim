import numpy as np
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Union, Optional, overload
from liegroups.numpy import SO2, SO3
from liegroups.numpy import SE2, SE3


_TRANSLATION_TOLERANCE = 1e-6  # m
_ROTATION_TOLERANCE = 1e-9  # rad
_RAD_TO_DEG_FACTOR = 180.0 / np.pi
_DEG_TO_RAD_FACTOR = np.pi / 180.0
_TWO_PI = 2 * np.pi


class DIM(Enum):
    TWO = 2
    THREE = 3


def none_to_zero(x: Optional[float]) -> float:
    """
    Converts a None value to 0.0 if x is None. Otherwise, returns x.

    Args:
        x (Optional[float]): The input value.

    Returns:
        float: The converted value.
    """
    return 0.0 if x is None else x


def theta_to_pipi(theta: float) -> float:
    """
    Wraps an angle in radians to the range [-pi, pi].

    Parameters:
        theta (float): The angle in radians.

    Returns:
        float: The angle in the range [-pi, pi].
    """
    return (theta + np.pi) % _TWO_PI - np.pi


class Point(ABC):
    def __init__(self, dim: DIM, x: float, y: float, z: float, frame: str) -> None:
        """
        Abstract base class for 2D and 3D point classes.

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
            x1 (Point): The coordinates of the first point.
            x2 (Point): The coordinates of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        assert isinstance(x1, Point)
        assert isinstance(x2, Point)
        return np.linalg.norm(x1.array() - x2.array())

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
        return np.linalg.norm([self._x, self._y, self._z])

    @property
    @abstractmethod
    def array(self) -> np.ndarray:
        """
        Returns an array containing the coordinates of this point.

        Returns:
            np.ndarray: The coordinates of this point.
        """
        pass

    @classmethod
    @abstractmethod
    def by_array(cls, other, frame: str) -> "Point":
        """
        Creates a Point object from an array.

        Args:
          other : An array-like object representing the coordinates of the point.
          frame (str): The frame of reference for the point.

        Returns:
          Point: A Point object.
        """
        pass

    @abstractmethod
    def copy(self) -> "Point":
        """
        Creates a deep copy of this point.

        Returns:
            Point: A deep copy of this point.
        """
        pass

    @abstractmethod
    def copyInverse(self) -> "Point":
        """
        Creates a deep copy of this point and negate coordinates.

        Returns:
            Point: A deep copy of this point with negated coordinates.
        """
        pass

    def distance(self, other) -> float:
        """
        Calculates the Euclidean distance between two points.

        Parameters:
            other (Point): The other point to calculate the distance to.

        Returns:
            float: The Euclidean distance between the two points.
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
                np.allclose(self.array(), other.array(), _TRANSLATION_TOLERANCE)
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

        self._local_frame = local_frame
        self._base_frame = base_frame

    @staticmethod
    def dist(x1: "Rot", x2: "Rot") -> float:
        """
        Calculates the chordal distance between two rotations.

        Args:
            x1 (Rot2): The first Rot object.
            x2 (Rot2): The second Rot object.

        Returns:
            float: The chordal distance between x1 and x2.
        """
        return np.linalg.norm((x1.copyInverse() * x2).log_map())

    @property
    def local_frame(self) -> str:
        return self._local_frame

    @property
    def base_frame(self) -> str:
        return self._base_frame

    @property
    @abstractmethod
    def degrees(self) -> Union[float, Tuple[float, float, float]]:
        """
        Returns the angular representation of the rotation in degrees. For 2D rotations, theta is returned. For 3D rotations, (roll, pitch, yaw) is returned.

        Returns:
            float: angle(s) in degrees.
        """
        pass

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """
        Returns the matrix representation of the rotation.

        Returns:
          np.ndarray: A 2D array representing the rotation matrix.
        """
        pass

    @property
    @abstractmethod
    def log_map(self) -> np.ndarray:
        """
        Return the log map of the rotation.

        Returns:
            np.ndarray: the log map of the rotation.
        """
        pass

    @classmethod
    @abstractmethod
    def by_degrees(cls, degrees, local_frame: str, base_frame: str):
        """
        Creates a Rot object from degrees with a specified local and base frame. For 2D rotations, theta must be in degrees. For 3D rotations, (roll, pitch, yaw) must be in degrees.

        Args:
            degrees (float): the angle(s) in degrees respresenting the rotation.
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
            vector (np.ndarray): the rotation vector.
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

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


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

        # enforce _theta in [-pi, pi] as a state
        self._theta = theta_to_pipi(none_to_zero(theta))

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, theta: float):
        assert np.isscalar(theta)
        self._theta = theta

    @property
    def cos(self) -> float:
        return math.cos(self.theta)

    @property
    def sin(self) -> float:
        return math.sin(self.theta)

    @property
    def degrees(self) -> float:
        return self.theta * _RAD_TO_DEG_FACTOR

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.cos, -self.sin], [self.sin, self.cos]])

    @property
    def log_map(self) -> np.ndarray:
        return np.array([self.theta])

    @classmethod
    def by_degrees(cls, degrees: float, local_frame: str, base_frame: str) -> "Rot2":
        theta = none_to_zero(degrees) * _DEG_TO_RAD_FACTOR
        return cls(theta, local_frame, base_frame)

    @classmethod
    def by_matrix(cls, matrix: np.ndarray, local_frame: str, base_frame: str) -> "Rot2":
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)
        assert matrix.dtype == np.float64
        return cls(np.arctan2(matrix[1, 0], matrix[0, 0]), local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "Rot2":
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (1, 1)
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
        if isinstance(other, Rot2):
            assert self.local_frame == other.base_frame
            return Rot2(self.theta + other.theta, self.local_frame, other.base_frame)
        elif isinstance(other, Point2):
            assert self.local_frame == other.frame
            x = self.cos * other.x - self.sin * other.y
            y = self.sin * other.x + self.cos * other.y
            return Point2(x, y, self.base_frame)
        else:
            raise ValueError("Not a Point2 or Rot2 type to multiply.")

    def __str__(self) -> str:
        return f"Rot2[theta: {self.theta:.3f}, local_frame: {self.local_frame}, base_frame: {self.base_frame}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rot2):
            same_frames = (
                self.local_frame == other.local_frame
                and self.base_frame == other.base_frame
            )
            angle_similar = abs(self.theta - other.theta) < 1e-8
            return same_frames and angle_similar
        return False

    def __hash__(self) -> int:
        return hash((self.theta, self.local_frame, self.base_frame))


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
        self._SO3 = SO3.from_rpy(roll, pitch, yaw)

    @property
    def roll(self) -> float:
        return self._SO3.to_rpy()[0]

    @property
    def pitch(self) -> float:
        return self._SO3.to_rpy()[1]

    @property
    def yaw(self) -> float:
        return self._SO3.to_rpy()[2]

    @property
    def degrees(self) -> Tuple[float, float, float]:
        return self._SO3.to_rpy() * _RAD_TO_DEG_FACTOR

    @property
    def matrix(self) -> np.ndarray:
        return self._SO3.as_matrix()

    @property
    def log_map(self) -> np.ndarray:
        return self._SO3.log()

    @classmethod
    def by_degrees(
        cls, degrees: Tuple[float, float, float], local_frame: str, base_frame: str
    ) -> "Rot3":
        roll = none_to_zero(degrees[0]) * _DEG_TO_RAD_FACTOR
        pitch = none_to_zero(degrees[1]) * _DEG_TO_RAD_FACTOR
        yaw = none_to_zero(degrees[2]) * _DEG_TO_RAD_FACTOR
        return cls(roll, pitch, yaw, local_frame, base_frame)

    @classmethod
    def by_matrix(cls, matrix: np.ndarray, local_frame: str, base_frame: str) -> "Rot3":
        assert isinstance(matrix, np.ndarray)
        roll, pitch, yaw = SO3.from_matrix(matrix).to_rpy()
        return cls(roll, pitch, yaw, local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "Rot3":
        assert isinstance(vector, np.ndarray)
        roll, pitch, yaw = SO3.exp(vector).to_rpy()
        return cls(roll, pitch, yaw, local_frame, base_frame)

    def copy(self) -> "Rot3":
        return Rot3(self.roll, self.pitch, self.yaw, self.local_frame, self.base_frame)

    def copyInverse(self) -> "Rot3":
        roll, pitch, yaw = self._SO3.inv().to_rpy()
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
        if isinstance(other, Rot3):
            assert self.local_frame == other.base_frame
            roll, pitch, yaw = self._SO3.dot(other._SO3).to_rpy()
            return Rot3(roll, pitch, yaw, self.local_frame, other.base_frame)
        elif isinstance(other, Point3):
            assert self.local_frame == other.frame
            rotated_point = self._SO3.as_matrix() * other.array()
            x, y, z = rotated_point[0], rotated_point[1], rotated_point[2]
            return Point3(x, y, z, self.base_frame)
        else:
            raise ValueError("Not a Point3 or Rot3 type to multiply.")

    def __str__(self) -> str:
        return f"Rot3[roll: {self.roll:.3f}, pitch: {self.pitch:.3f}, yaw: {self.yaw:.3f}, local_frame: {self.local_frame}, base_frame: {self.base_frame}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rot3):
            same_frames = (
                self.local_frame == other.local_frame
                and self.base_frame == other.base_frame
            )
            angle_similar = np.allclose(
                np.array(self._SO3.to_rpy()),
                np.array(other._SO3.to_rpy()),
                _ROTATION_TOLERANCE,
            )
            return same_frames and angle_similar
        return False

    def __hash__(self) -> int:
        return hash(
            (self.roll, self.pitch, self.yaw, self.local_frame, self.base_frame)
        )


class SEPose(ABC):
    def __init__(self, local_frame: str, base_frame: str) -> None:
        """
        Abstract base class for 2D and 3D pose classes.

        Args:
            local_frame (str): the local frame of the pose.
            base_frame (str): the base frame of the pose.
        """
        assert isinstance(local_frame, str)
        assert isinstance(base_frame, str)
        assert not local_frame == base_frame

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
        T1 = x1.matrix()
        T2 = x2.matrix()
        assert T1.shape == T2.shape
        assert T1.shape[0] == T2.shape[1]
        assert T1.shape[0] == DIM.TWO or T1.shape[0] == DIM.THREE
        return np.linalg.norm(T1 - T2, ord="fro")

    @property
    def local_frame(self) -> str:
        return self._local_frame

    @property
    def base_frame(self) -> str:
        return self._base_frame

    @property
    @abstractmethod
    def rotation(self):
        pass

    @property
    @abstractmethod
    def translation(self):
        pass

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """
        Returns the matrix representation of the pose object.

        Returns:
            np.ndarray: The matrix representation of the pose object.
        """
        pass

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
    @abstractmethod
    def log_map(self) -> np.ndarray:
        """
        Computes the logarithmic map of the current pose object.

        Returns:
            np.ndarray: The logarithmic map of the current pose object.
        """
        pass

    @property
    def grad_x_logmap(self) -> np.ndarray:
        """
        Computes the gradient of the logarithmic map with respect to the pose object.

        Returns:
            np.ndarray: The gradient of the logarithmic map with respect to the pose object.
        """
        pass

    @classmethod
    @abstractmethod
    def by_point_and_rotation(cls, point, rotation, local_frame: str, base_frame: str):
        """
        Creates a pose object from point and rotation objects.

        Args:
            point (Point): The point object.
            rotation (Rot): The rotation object.
            local_frame (str): The local frame.
            base_frame (str): The base frame.

        Returns:
            SEPose: the pose object corresponding to the input point and rotation objects.
        """
        pass

    @classmethod
    @abstractmethod
    def by_matrix(cls, matrix: np.ndarray, local_frame: str, base_frame: str):
        """
        Creates a pose object from a homogenous transformation matrix.

        Args:
            matrix: the homogenous transformation matrix.
            local_frame: the local frame of the matrix.
            base_frame: the base frame of the matrix.

        Returns:
            SEPose: the SEPose object corresponding to the input matrix.
        """
        pass

    @classmethod
    @abstractmethod
    def by_exp_map(cls, vector: np.ndarray, local_frame: str, base_frame: str):
        """
        Creates a pose object from an exponential map vector.

        Args:
            vector: the exponential map vector.
            local_frame: the local frame of the exponential map vector.
            base_frame: the base frame of the exponential map vector.

        Returns:
            SEPose: the SEPose object corresponding to the input vector.
        """
        pass

    @abstractmethod
    def copy(self):
        """
        Returns deep copy of this pose.

        Returns:
            SEPose: a deep copy of this pose.
        """
        pass

    @abstractmethod
    def copyInverse(self):
        """
        Returns deep copy of the inverse of this pose.

        Returns:
            SEPose: a deep copy of the inverse of this pose.
        """
        pass

    @abstractmethod
    def range_and_bearing_to_point(
        self, point
    ) -> Union[Tuple[float, float], Tuple[float, Tuple[float, float]]]:
        """
        Returns the range and bearing from this pose to the point.

        Args:
            point (Point): the point to measure to.

        Returns:
            Union[Tuple[float, float], Tuple[float, Tuple[float, float]]]: (range, bearing).
        """
        pass

    @abstractmethod
    def transform_to(self, other):
        """
        Returns the coordinate frame of the other pose in the coordinate frame of this pose.

        Args:
            other (SEPose): the other pose which we want the coordinate frame with respect to.

        Returns:
            SEPose: the coordinate frame of this pose with respect to the given pose.
        """
        pass

    @abstractmethod
    def transform_local_point_to_base(self, local_point):
        """
        Returns a point expressed in local frame of self in the base frame of self.

        Args:
            local_point (Point): the point expressed in the local frame of self.

        Returns:
            Point: a point expressed in the base frame of self.
        """
        pass

    @abstractmethod
    def transform_base_point_to_local(self, base_point):
        """
        Returns a point expressed in base frame of self in the local frame of self.

        Args:
            base_point (Point): the point expressed in the base frame of self.

        Returns:
            Point: a point expressed in the local frame of self.
        """
        pass

    @abstractmethod
    def distance_to_pose(self, other) -> float:
        """
        Returns the distance between this pose and another pose.

        Args:
            other (SEPose): the other pose.

        Returns:
            float: the distance between this pose and the other pose.
        """
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __imul__(self, other):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


class SE2Pose(SEPose):
    def __init__(
        self, x: float, y: float, theta: float, local_frame: str, base_frame: str
    ) -> None:
        super().__init__(local_frame, base_frame)
        """
        A pose in SE(2).

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

        self._point = Point2(x=x, y=y, frame=base_frame)
        self._rot = Rot2(theta=theta, local_frame=local_frame, base_frame=base_frame)
        self._local_frame = local_frame
        self._base_frame = base_frame

    @property
    def theta(self):
        return self._rot.theta

    @property
    def x(self):
        return self._point.x

    @property
    def y(self):
        return self._point.y

    @property
    def rotation(self):
        return self._rot

    @property
    def translation(self):
        return self._point

    @property
    def matrix(self):
        r_c = self._rot.cos
        r_s = self._rot.sin
        x = self._point.x
        y = self._point.y
        return np.array([[r_c, -r_s, x], [r_s, r_c, y], [0, 0, 1]])

    @property
    def array(self):
        return np.array([self.x, self.y, self.theta])

    @property
    def log_map(self) -> np.ndarray:
        r = self._rot
        t = self._point
        w = r.theta
        if abs(w) < 1e-10:
            return np.array([t.x, t.y, w])
        else:
            c_1 = r.cos - 1.0
            s = r.sin
            det = c_1 * c_1 + s * s
            rot_pi_2 = Rot2(np.pi / 2.0, self.local_frame, self.base_frame)
            p = rot_pi_2 * (r.unrotate_point(t) - t)
            v = (w / det) * p
            return np.array([v.x, v.y, w])

    @property
    def grad_x_logmap(self) -> np.ndarray:
        if abs(self.theta) < 1e-10:
            return np.identity(3)
        else:
            logmap_x, logmap_y, logmap_th = self.log_map()
            th_2 = logmap_th / 2.0
            diag1 = th_2 * np.sin(logmap_th) / (1.0 - np.cos(logmap_th))
            return np.array(
                [
                    [
                        diag1,
                        th_2,
                        (
                            logmap_x / logmap_th
                            + th_2 * (self.x / (np.cos(logmap_th) - 1))
                        ),
                    ],
                    [
                        -th_2,
                        diag1,
                        (
                            logmap_y / logmap_th
                            + th_2 * (self.y / (np.cos(logmap_th) - 1))
                        ),
                    ],
                    [0.0, 0.0, 1.0],
                ]
            )

    @classmethod
    def by_point_and_rotation(
        cls, point: Point2, rotation: Rot2, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        assert isinstance(point, Point2)
        assert isinstance(rotation, Rot2)
        return cls(point.x, point.y, rotation.theta, local_frame, base_frame)

    @classmethod
    def by_matrix(
        cls, matrix: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        assert isinstance(matrix, np.ndarray)
        point = Point2.by_array(matrix[:2, 2], local_frame)
        rotation = Rot2.by_matrix(matrix[:2, :2], local_frame, base_frame)
        return SE2Pose.by_point_and_rotation(point, rotation, local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE2Pose":
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        w = vector[2]
        if abs(w) < 1e-10:
            return SE2Pose(vector[0], vector[1], w, local_frame, base_frame)
        else:
            cos_theta = np.cos(w)
            sin_theta = np.sin(w)

            # get rotation
            R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            assert R.shape == (2, 2)
            rt = Rot2.by_matrix(R, local_frame, base_frame)

            # get translation
            V = np.array([[sin_theta, cos_theta - 1], [1 - cos_theta, sin_theta]]) / w
            u = vector[0:2]
            t = V @ u
            assert len(u) == 2
            pt = Point2.by_array(t, base_frame)

            return SE2Pose.by_point_and_rotation(pt, rt, local_frame, base_frame)

    def copy(self) -> "SE2Pose":
        return SE2Pose(
            x=self.x,
            y=self.y,
            theta=self.theta,
            local_frame=self._local_frame,
            base_frame=self._base_frame,
        )

    def copyInverse(self) -> "SE2Pose":
        inv_t = -(self._rot.unrotate_point(self._point))
        return SE2Pose.by_point_and_rotation(
            point=inv_t,
            rotation=self._rot.copyInverse(),
            local_frame=self._base_frame,
            base_frame=self._local_frame,
        )

    def range_and_bearing_to_point(self, point: Point2) -> Tuple[float, float]:
        assert isinstance(point, Point2)
        diff = point - self._point
        dist = diff.norm
        bearing = self._rot.bearing_to_base_frame_point(diff)
        return dist, bearing

    def transform_to(self, other):
        assert isinstance(other, SE2Pose)
        assert self.base_frame == other.base_frame
        assert not self.local_frame == other.local_frame
        return self.copyInverse() * other

    def transform_local_point_to_base(self, local_point: Point2) -> Point2:
        assert isinstance(local_point, Point2)
        return self * local_point

    def transform_base_point_to_local(self, base_point: Point2) -> Point2:
        assert isinstance(base_point, Point2)
        return self.copyInverse() * base_point

    def distance_to_pose(self, other: "SE2Pose") -> float:
        assert isinstance(other, SE2Pose)

        cur_position = self.translation
        other_position = other.translation
        assert cur_position.frame == other_position.frame

        dist = cur_position.distance(other_position)
        assert isinstance(dist, float)
        return dist

    def __mul__(self, other):
        assert isinstance(other, (SE2Pose, Point2))
        if isinstance(other, SE2Pose):
            assert self.local_frame == other.base_frame
            r = self._rot * other.rotation
            t = self._point + self._rot * other.translation
            return SE2Pose.by_point_and_rotation(
                point=t,
                rotation=r,
                local_frame=other.local_frame,
                base_frame=self.base_frame,
            )
        if isinstance(other, Point2):
            assert self.local_frame == other.frame
            return self._rot * other + self._point

    def __imul__(self, other):
        if isinstance(other, SE2Pose):
            pos = self * other
            self._point = self._point.x(x=pos.x)
            self._point = self._point.y(y=pos.y)
            self._rot = self._rot.theta(pos.theta)
            self._local_frame = other.local_frame
            return self
        raise ValueError("Not a Pose2 type to multiply.")

    def __str__(self) -> str:
        return f"Pose2[x: {self.x:.3f}, y: {self.y:.3f}, theta: {self.theta:.3f}, local_frame: {self.local_frame}, base_frame: {self.base_frame}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SE2Pose):
            return (
                abs(self._rot.theta - other.theta) < _ROTATION_TOLERANCE
                and abs(self._point.x - other.x) < _TRANSLATION_TOLERANCE
                and abs(self._point.y - other.y) < _TRANSLATION_TOLERANCE
                and self._local_frame == other.local_frame
                and self._base_frame == other.base_frame
            )
        return False

    def __hash__(self):
        return hash(
            (
                self._point.x,
                self._point.y,
                self._rot.theta,
                self._local_frame,
                self._base_frame,
            )
        )


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
        A pose in SE(3).

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
        T[:3, :3] = R.matrix()
        T[:3, 3] = t

        # class attributes
        self._SE3 = SE3.from_matrix(T)
        self._local_frame = local_frame
        self._base_frame = base_frame

    @property
    def x(self) -> float:
        return self._SE3.trans[0]

    @property
    def y(self) -> float:
        return self._SE3.trans[1]

    @property
    def z(self) -> float:
        return self._SE3.trans[2]

    @property
    def roll(self) -> float:
        return self._SE3.rot.to_rpy()[0]

    @property
    def pitch(self) -> float:
        return self._SE3.rot.to_rpy()[1]

    @property
    def yaw(self) -> float:
        return self._SE3.rot.to_rpy()[2]

    @property
    def rotation(self) -> Rot3:
        return Rot3(self.roll, self.pitch, self.yaw, self.local_frame, self.base_frame)

    @property
    def translation(self) -> Point3:
        return Point3(self.x, self.y, self.z, self.base_frame)

    @property
    def matrix(self) -> np.ndarray:
        return self._SE3.as_matrix()

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])

    @property
    def log_map(self) -> np.ndarray:
        return self._SE3.log()

    @property
    def grad_x_logmap(self) -> np.ndarray:
        raise NotImplementedError("Gradient of logmap not implemented for SE3Pose.")

    @classmethod
    def by_point_and_rotation(
        cls, point: Point3, rotation: Rot3, local_frame: str, base_frame: str
    ) -> "SE3Pose":
        assert isinstance(point, Point3)
        assert isinstance(rotation, Rot3)
        return cls(
            point.x,
            point.y,
            point.z,
            rotation.roll,
            rotation.pitch,
            rotation.yaw,
            local_frame,
            base_frame,
        )

    @classmethod
    def by_matrix(
        cls, matrix: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE3Pose":
        assert isinstance(matrix, np.ndarray)
        point = Point3.by_array(matrix[:3, 3], local_frame)
        rotation = Rot3.by_matrix(matrix[:3, :3], local_frame, base_frame)
        return SE3Pose.by_point_and_rotation(point, rotation, local_frame, base_frame)

    @classmethod
    def by_exp_map(
        cls, vector: np.ndarray, local_frame: str, base_frame: str
    ) -> "SE3Pose":
        assert isinstance(vector, np.ndarray)
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
        T = self._SE3.inv()
        return SE3Pose(
            x=T.trans[0],
            y=T.trans[1],
            z=T.trans[2],
            roll=T.rot.to_rpy()[0],
            pitch=T.rot.to_rpy()[1],
            yaw=T.rot.to_rpy()[2],
            local_frame=self._local_frame,
            base_frame=self._base_frame,
        )

    def range_and_bearing_to_point(
        self, pt: Point3
    ) -> Tuple[float, Tuple[float, float]]:
        assert isinstance(pt, Point3)
        diff = pt - self.translation
        dist = diff.norm
        bearing = self.rotation.bearing_to_base_frame_point(diff)
        return dist, bearing

    def transform_to(self, other):
        assert isinstance(other, SE3Pose)
        assert self.base_frame == other.base_frame
        assert not self.local_frame == other.local_frame
        return self.copyInverse() * other

    def transform_local_point_to_base(self, local_point: Point3) -> Point3:
        assert isinstance(local_point, Point3)
        return self * local_point

    def transform_base_point_to_local(self, base_point: Point3) -> Point3:
        assert isinstance(base_point, Point3)
        return self.copyInverse() * base_point

    def distance_to_pose(self, other: "SE3Pose") -> float:
        assert isinstance(other, SE3Pose)

        cur_position = self.translation
        other_position = other.translation
        assert cur_position.frame == other_position.frame

        dist = cur_position.distance(other_position)
        assert isinstance(dist, float)
        return dist

    def __mul__(self, other):
        assert isinstance(other, (SE3Pose, Point3))
        if isinstance(other, SE3Pose):
            assert self.local_frame == other.base_frame
            T = self.matrix @ other.matrix
            return SE3Pose.by_matrix(
                matrix=T, local_frame=other.local_frame, base_frame=self.base_frame
            )
        if isinstance(other, Point3):
            assert self.local_frame == other.frame
            return self.rotation * other + self.translation

    def __imul__(self, other):
        if isinstance(other, SE3Pose):
            pose = self * other
            self._SE3 = SE3.from_matrix(pose.matrix())
            self._local_frame = other.local_frame
            return self
        raise ValueError("Not a Pose3 type to multiply.")

    def __str__(self) -> str:
        return f"Pose3[x: {self.x:.3f}, y: {self.y:.3f}, z: {self.z:.3f}, roll: {self.roll:.3f}, pitch: {self.pitch:.3f}, yaw: {self.yaw:.3f}, local_frame: {self.local_frame}, base_frame: {self.base_frame}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SE3Pose):
            rotation_similar = np.allclose(
                np.array(self._SE3.rot.to_rpy()),
                np.array(other._SE3.rot.to_rpy()),
                _ROTATION_TOLERANCE,
            )
            translation_similar = np.allclose(
                self._SE3.trans, other._SE3.trans, _TRANSLATION_TOLERANCE
            )
            return (
                rotation_similar
                and translation_similar
                and self._local_frame == other.local_frame
                and self._base_frame == other.base_frame
            )
        return False

    def __hash__(self):
        return hash((self._SE3, self._local_frame, self._base_frame))
