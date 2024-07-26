from copy import deepcopy
from abc import ABC, abstractmethod

import itertools
import math
import numpy as np
from typing import Tuple, List, Union, Optional
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes

from manhattan.geometry.Elements import Point, Point3, SE2Pose, Point2, SE3Pose, SEPose, DIM
from manhattan.agent.agent import Robot
from manhattan.utils.sample_utils import choice


def _find_nearest(
    array: Union[np.ndarray, List[float]], value: float
) -> Tuple[int, float, float]:
    """Finds the nearest value in the array to the given value. Returns the
    index, difference, and value of the nearest value in the array.

    Args:
        array (Union[np.ndarray, List[float]]): the array to check the nearest
            value of
        value (float): the value to check for the nearest value in the array

    Returns:
        Tuple[int, float, float]: index of the nearest value, difference between
            values, and value of the nearest value
    """
    assert len(array) > 0

    array = np.asarray(array)
    distances = np.abs(array - value)
    idx = int(distances.argmin())
    arr_val = float(array[idx])
    delta = value - arr_val
    return idx, delta, arr_val

VERTEX_TYPES = Union[Tuple[int, int], Tuple[int, int, int]]
VERTEX_LIST_TYPES = Union[List[Tuple[int, int]], List[Tuple[int, int, int]]]
COORDINATE_TYPES = Union[Tuple[float, float], Tuple[float, float, float]]
COORDINATE_LIST_TYPES = Union[List[Tuple[float, float]], List[Tuple[float, float, float]]]
AREA_TYPES = Union[List[Tuple[int, int]], List[Tuple[int, int, int]]]
BOUNDS_TYPES = Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]
POSE_TYPES = Union[SE2Pose, SE3Pose]
POINT_TYPES = Union[Point2, Point3]

class ManhattanWorld:
    """
    This class creates a simulated environment of Manhattan world with beacons.
    """

    def __init__(
        self,
        dim: DIM = DIM.TWO,
        grid_vertices_shape: tuple = None,
        z_steps_to_intersection: int = 1,
        y_steps_to_intersection: int = 1,
        x_steps_to_intersection: int = 1,
        cell_scale: float = 1.0,
        robot_area: Optional[AREA_TYPES] = None,
        check_collision: bool = True,
        tol: float = 1e-5,
    ):
        """Constructor for Manhattan waterworld environment. Note that the
        beacons are only allowed in areas that is infeasible to the robot. As
        of now the robot feasible area is only rectangular

        Args:
            dim (int, optional): dimension of the world. Defaults to 2.
            grid_vertices_shape (tuple): a tuple defining the shape of
                grid vertices; note that the vertices follow ij indexing.
                Defaults to (9, 9).
            cell_scale (int, optional): width and length of a cell. Defaults to 1.
            robot_area (List[Tuple], optional): [(left, bottom), (right, top)]
                bottom left and top right vertices of a rectangular area; all
                the rest area will be infeasible. Defaults to None.
            check_collision (bool, optional): [description]. Defaults to True.
            tol (float, optional): [description]. Defaults to 1e-5.
        """
        # Assert dimension validity
        assert dim in [DIM.TWO, DIM.THREE]
        if dim == DIM.TWO:
            assert len(grid_vertices_shape) == 2
            self._num_x_pts, self._num_y_pts = grid_vertices_shape
            self._num_z_pts = 0
        else: 
            assert len(grid_vertices_shape) == 3
            self._num_x_pts, self._num_y_pts, self._num_z_pts = grid_vertices_shape
        
        self.dim = dim

        # have to add one to get the number of rows and columns
        self._num_x_pts += 1
        self._num_y_pts += 1
        self._num_z_pts += 1

        # controls when grid lines intersect
        self._z_steps_to_intersection = z_steps_to_intersection
        self._y_steps_to_intersection = y_steps_to_intersection
        self._x_steps_to_intersection = x_steps_to_intersection

        self._scale = cell_scale

        self._check_collision = check_collision

        self._tol = tol

        if robot_area is not None:
            assert self.check_vertex_list_valid(robot_area)

        # create grid
        self._grid = np.zeros(grid_vertices_shape, dtype=np.float32)

        self._x_coords = np.arange(self._num_x_pts) * self._scale
        self._y_coords = np.arange(self._num_y_pts) * self._scale
        self._z_coords = np.arange(self._num_z_pts) * self._scale
        if (self.dim == DIM.TWO):
            self._xv, self._yv = np.meshgrid(self._x_coords, self._y_coords, indexing="ij")
        else:
            self._xv, self._yv, self._zv = np.meshgrid(self._x_coords, self._y_coords, self._z_coords, indexing="ij")

        if robot_area is not None:
            if (self.dim == DIM.TWO):
                # ensure a rectangular feasible area for robot
                bl, tr = robot_area

                # set bounds on feasible area as variables
                self._min_x_idx_feasible = bl[0]
                self._max_x_idx_feasible = tr[0]
                self._min_y_idx_feasible = bl[1]
                self._max_y_idx_feasible = tr[1]

                self._min_x_coord_feasible = bl[0] * self._scale
                self._max_x_coord_feasible = tr[0] * self._scale
                self._min_y_coord_feasible = bl[1] * self._scale
                self._max_y_coord_feasible = tr[1] * self._scale

                # also save a mask for the feasible area
                self._robot_feasibility = np.zeros(
                    (self._num_x_pts, self._num_y_pts), dtype=bool
                )
                self._robot_feasibility[bl[0] : tr[0] + 1, bl[1] : tr[1] + 1] = True
            else:
                # ensure a rectangular prismic volume for robot
                # blf: bottom left front, trb: top right back
                blf, trb = robot_area

                # set bounds on feasible area as variables
                self._min_x_idx_feasible = blf[0]
                self._max_x_idx_feasible = trb[0]
                self._min_y_idx_feasible = blf[1]
                self._max_y_idx_feasible = trb[1]
                self._min_z_idx_feasible = blf[2]
                self._max_z_idx_feasible = trb[2]

                self._min_x_coord_feasible = blf[0] * self._scale
                self._max_x_coord_feasible = trb[0] * self._scale
                self._min_y_coord_feasible = blf[1] * self._scale
                self._max_y_coord_feasible = trb[1] * self._scale
                self._min_z_coord_feasible = blf[2] * self._scale
                self._max_z_coord_feasible = trb[2] * self._scale

                # also save a mask for the feasible area
                self._robot_feasibility = np.zeros(
                    (self._num_x_pts, self._num_y_pts, self._num_z_pts), dtype=bool
                )
                self._robot_feasibility[blf[0] : trb[0] + 1, blf[1] : trb[1] + 1, blf[2] : trb[2] + 1] = True
        else:
            if (self.dim == DIM.TWO):
                # if no area specified, all area is now feasible

                # set bounds on feasible area as variables
                self._min_x_idx_feasible = 0
                self._max_x_idx_feasible = self._num_x_pts - 1
                self._min_y_idx_feasible = 0
                self._max_y_idx_feasible = self._num_y_pts - 1

                self._min_x_coord_feasible = np.min(self._x_coords)
                self._max_x_coord_feasible = np.max(self._x_coords)
                self._min_y_coord_feasible = np.min(self._y_coords)
                self._max_y_coord_feasible = np.max(self._y_coords)

                # also save a mask for the feasible area
                self._robot_feasibility = np.ones(
                    (self._num_x_pts, self._num_y_pts), dtype=bool
                )
            else:
                # if no area specified, all area is now feasible

                # set bounds on feasible area as variables
                self._min_x_idx_feasible = 0
                self._max_x_idx_feasible = self._num_x_pts - 1
                self._min_y_idx_feasible = 0
                self._max_y_idx_feasible = self._num_y_pts - 1
                self._min_z_idx_feasible = 0
                self._max_z_idx_feasible = self._num_z_pts - 1

                self._min_x_coord_feasible = np.min(self._x_coords)
                self._max_x_coord_feasible = np.max(self._x_coords)
                self._min_y_coord_feasible = np.min(self._y_coords)
                self._max_y_coord_feasible = np.max(self._y_coords)
                self._min_z_coord_feasible = np.min(self._z_coords)
                self._max_z_coord_feasible = np.max(self._z_coords)

                # also save a mask for the feasible area
                self._robot_feasibility = np.ones(
                    (self._num_x_pts, self._num_y_pts, self._num_z_pts), dtype=bool
                )

        # make sure nothing weird happened in recording these feasible values
        assert self._x_coords[self._min_x_idx_feasible] == self._min_x_coord_feasible
        assert self._x_coords[self._max_x_idx_feasible] == self._max_x_coord_feasible
        assert self._y_coords[self._min_y_idx_feasible] == self._min_y_coord_feasible
        assert self._y_coords[self._max_y_idx_feasible] == self._max_y_coord_feasible

        if (self.dim == DIM.THREE):
            assert self._z_coords[self._min_z_idx_feasible] == self._min_z_coord_feasible
            assert self._z_coords[self._max_z_idx_feasible] == self._max_z_coord_feasible

    def __str__(self):
        line = "ManhattanWorld Environment\n"
        line += "Shape: " + self.shape.__repr__() + "\n"
        if (self.dim == DIM.THREE):
            line += f"Height Corner Number: {self.z_steps_to_intersection}\n"
        line += f"Row Corner Number: {self.y_steps_to_intersection}\n"
        line += f"Column Corner Number: {self.x_steps_to_intersection}\n"
        line += f"Cell Scale: {self.cell_scale}\n"
        line += f"Robot Feasible Area: {self.robot_area}\n"
        line += (
            "Beacon feasible vertices: " + self._beacon_feasibility.__repr__() + "\n"
        )
        return line

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def bounds(self) -> BOUNDS_TYPES:
        if (self.dim == DIM.TWO):
            return (0.0, 0.0, self._x_coords[-1], self._y_coords[-1])
        return (0.0, 0.0, 0.0, self._x_coords[-1], self._y_coords[-1], self._z_coords[-1])

    def set_robot_area_feasibility(self, area: AREA_TYPES):
        """Sets the feasibility status for the robots as a rectangular area. Anything
        outside of this area will be the inverse of the status.

        Args:
            area (List[Tuple[int, int]]): the feasibility area for robots, denoted by the
                bottom left and top right vertices.
        """
        if (self.dim == DIM.TWO):
            assert self.check_vertex_list_valid(area)
            assert len(area) == 2

            mask = np.zeros((self._num_x_pts, self._num_y_pts), dtype=bool)
            bl, tr = area

            # set bounds on feasible area as variables
            self._min_x_idx_feasible = bl[0]
            self._max_x_idx_feasible = tr[0]
            self._min_y_idx_feasible = bl[1]
            self._max_y_idx_feasible = tr[1]

            self._min_x_coord_feasible = bl[0] * self._scale
            self._max_x_coord_feasible = tr[0] * self._scale
            self._min_y_coord_feasible = bl[1] * self._scale
            self._max_y_coord_feasible = tr[1] * self._scale

            # also save a mask for the feasible area
            mask[bl[0] : tr[0] + 1, bl[1] : tr[1] + 1] = True
            self._robot_feasibility[mask] = True
            self._robot_feasibility[np.invert(mask)] = False

            # make sure nothing weird happened in recording these feasible values
            assert (
                abs(self._min_x_idx_feasible * self._scale - self._min_x_coord_feasible)
                < self._tol
            )
            assert (
                abs(self._max_x_idx_feasible * self._scale - self._max_x_coord_feasible)
                < self._tol
            )
            assert (
                abs(self._min_y_idx_feasible * self._scale - self._min_y_coord_feasible)
                < self._tol
            )
            assert (
                abs(self._max_y_idx_feasible * self._scale - self._max_y_coord_feasible)
                < self._tol
            )
        else:
            assert self.check_vertex_list_valid(area)
            assert len(area) == 2

            mask = np.zeros((self._num_x_pts, self._num_y_pts, self._num_z_pts), dtype=bool)
            blf, trb = area

            # set bounds on feasible area as variables
            self._min_x_idx_feasible = blf[0]
            self._max_x_idx_feasible = trb[0]
            self._min_y_idx_feasible = blf[1]
            self._max_y_idx_feasible = trb[1]
            self._min_z_idx_feasible = blf[2]
            self._max_z_idx_feasible = trb[2]

            self._min_x_coord_feasible = blf[0] * self._scale
            self._max_x_coord_feasible = trb[0] * self._scale
            self._min_y_coord_feasible = blf[1] * self._scale
            self._max_y_coord_feasible = trb[1] * self._scale
            self._min_z_coord_feasible = blf[2] * self._scale
            self._max_z_coord_feasible = trb[2] * self._scale

            # also save a mask for the feasible area
            mask[blf[0] : trb[0] + 1, blf[1] : trb[1] + 1, blf[2] : trb[2] + 1] = True
            self._robot_feasibility[mask] = True
            self._robot_feasibility[np.invert(mask)] = False

            # make sure nothing weird happened in recording these feasible values
            assert (
                abs(self._min_x_idx_feasible * self._scale - self._min_x_coord_feasible)
                < self._tol
            )
            assert (
                abs(self._max_x_idx_feasible * self._scale - self._max_x_coord_feasible)
                < self._tol
            )
            assert (
                abs(self._min_y_idx_feasible * self._scale - self._min_y_coord_feasible)
                < self._tol
            )
            assert (
                abs(self._max_y_idx_feasible * self._scale - self._max_y_coord_feasible)
                < self._tol
            )
            assert (
                abs(self._min_z_idx_feasible * self._scale - self._min_z_coord_feasible)
                < self._tol
            )
            assert (
                abs(self._max_z_idx_feasible * self._scale - self._max_z_coord_feasible)
                < self._tol
            )

    def get_neighboring_vertices(self, vert: VERTEX_TYPES) -> VERTEX_TYPES:
        """gets all neighboring vertices to the vertex at index (i, j). Only
        returns valid indices (not out of bounds)

        Args:
            vert (tuple): a vertex index (i, j)

        Returns:
            List[tuple]: list of all neighboring vertices
        """

        assert self.check_vertex_valid(vert)

        if (self.dim == DIM.TWO):
            i, j = vert
            candidate_vertices = []

            # connectivity is based on whether we are at a corner or not
            if i % self._x_steps_to_intersection == 0:
                candidate_vertices.append((i, j - 1))
                candidate_vertices.append((i, j + 1))
            if j % self._y_steps_to_intersection == 0:
                candidate_vertices.append((i - 1, j))
                candidate_vertices.append((i + 1, j))

            # prune all vertices that are out of bounds
            vertices_in_bound = [
                v for v in candidate_vertices if self.vertex_is_in_bounds(v)
            ]
            return vertices_in_bound
        else:
            i, j, k = vert
            candidate_vertices = []

            # connectivity is based on whether we are at a corner or not
            # i is on the y-axis, j is on the x-axis, k is on the z-axis
            if i % self._x_steps_to_intersection == 0:
                candidate_vertices.append((i, j - 1, k))
                candidate_vertices.append((i, j + 1, k))
            if j % self._y_steps_to_intersection == 0:
                candidate_vertices.append((i - 1, j, k))
                candidate_vertices.append((i + 1, j, k))
            if k % self._z_steps_to_intersection == 0:
                candidate_vertices.append((i, j, k - 1))
                candidate_vertices.append((i, j, k + 1))
            
            # prune all vertices that are out of bounds
            vertices_in_bound = [
                v for v in candidate_vertices if self.vertex_is_in_bounds(v)
            ]
            return vertices_in_bound

    def get_neighboring_robot_vertices(
        self, vert: VERTEX_TYPES
    ) -> Union[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """get all neighboring vertices to the vertex at index (i, j) that are
        feasible for the robot. Only returns valid indices (not out of bounds)

        Args:
            vert (tuple): a vertex index (i, j)

        Returns:
            List[Tuple[int, int]]: the list of neighboring vertices that are
                feasible for the robot
        """

        assert self.check_vertex_valid(vert)
        neighbor_verts = self.get_neighboring_vertices(vert)
        assert self.check_vertex_list_valid(neighbor_verts)

        if (self.dim == DIM.TWO):
            assert 2 <= len(neighbor_verts) <= 4

            feasible_neighbor_verts = [
                v for v in neighbor_verts if self.vertex_is_robot_feasible(v)
            ]

            assert len(feasible_neighbor_verts) <= 4
            return feasible_neighbor_verts
        else:
            assert 3 <= len(neighbor_verts) <= 6

            feasible_neighbor_verts = [
                v for v in neighbor_verts if self.vertex_is_robot_feasible(v)
            ]

            assert len(feasible_neighbor_verts) <= 6
            return feasible_neighbor_verts

    def get_neighboring_robot_vertices_not_behind_robot(
        self, robot: Robot
    ) -> Union[List[Tuple[Point2, float]], List[Tuple[Point3, Tuple[float, float, float]]]]:
        """get all neighboring vertices to the vertex the robot is at which are
        not behind the given robot

        Args:
            robot (Robot): the robot

        Returns:
            List[Tuple[Point2, float]]: the list of neighboring vertices that are
                not behind the robot
        """
        # get robot position
        robot_loc = robot.position
        robot_pose = robot.pose

        if (self.dim == DIM.TWO):
            # get robot vertex
            robot_vert = self.point2vertex(robot_loc)
            assert self.check_vertex_valid(robot_vert)

            # get neighboring vertices in the robot feasible space
            neighboring_feasible_vertices = self.get_neighboring_robot_vertices(robot_vert)
            assert self.check_vertex_list_valid(neighboring_feasible_vertices)

            # convert vertices to points
            neighboring_feasible_pts = [
                self.vertex2point(v) for v in neighboring_feasible_vertices
            ]
            assert len(neighboring_feasible_pts) <= 4

            not_behind_pts = []
            for pt in neighboring_feasible_pts:
                distance, bearing = robot_pose.range_and_bearing_to_point(pt)
                if np.abs(bearing) < (np.pi / 2) + self._tol:
                    not_behind_pts.append((pt, bearing))

            return not_behind_pts
        else:
            # get robot vertex
            robot_vert = self.point2vertex(robot_loc)
            assert self.check_vertex_valid(robot_vert)

            # get neighboring vertices in the robot feasible space
            neighboring_feasible_vertices = self.get_neighboring_robot_vertices(robot_vert)
            assert self.check_vertex_list_valid(neighboring_feasible_vertices)

            # convert vertices to points
            neighboring_feasible_pts = [
                self.vertex2point(v) for v in neighboring_feasible_vertices
            ]

            assert len(neighboring_feasible_pts) <= 6
            
            not_behind_pts = []
            for pt in neighboring_feasible_pts:
                # bearing only encodes pitch and yaw, must calculate roll
                diff_pt = pt - robot_pose.point
                assert isinstance(diff_pt, Point3)
                assert robot_pose.base_frame == diff_pt.frame
                local_diff_pt = robot_pose.rot.unrotate_point(diff_pt)
                roll = math.atan2(local_diff_pt.z, local_diff_pt.y)

                distance, bearing = robot_pose.range_and_bearing_to_point(pt)
                if (np.abs(bearing[0]) < (np.pi / 2) + self._tol) and (np.abs(bearing[1]) < (np.pi / 2) + self._tol):
                    not_behind_pts.append((pt, (roll, bearing[0], bearing[1])))
            return not_behind_pts

    def get_vertex_behind_robot(self, robot: Robot) -> Union[List[Tuple[Point2, float]], List[Tuple[Point3, Tuple[float, float]]]]:
        """get the vertex that is behind the robot

        Args:
            robot (Robot): the robot

        Returns:
            Tuple[Point2, float]: the vertex behind the robot
        """
        # get robot position
        robot_loc = robot.position
        robot_pose = robot.pose

        # get robot vertex
        robot_vert = self.point2vertex(robot_loc)
        assert self.check_vertex_valid(robot_vert)

        # get neighboring vertices in the robot feasible space
        neighboring_feasible_vertices = self.get_neighboring_robot_vertices(robot_vert)
        assert self.check_vertex_list_valid(neighboring_feasible_vertices)

        # convert vertices to points
        neighboring_feasible_pts = [
            self.vertex2point(v) for v in neighboring_feasible_vertices
        ]

        for pt in neighboring_feasible_pts:
            distance, bearing = robot_pose.range_and_bearing_to_point(pt)

            if (self.dim == DIM.TWO):
                # 2D: Any bearing between 90 and 270 degrees is behind the robot
                if np.abs(bearing) > (np.pi / 2) + self._tol:
                    return (pt, bearing)
            else:
                # 3D: any bearing where abs(atan2(y, x)) > 90 degrees and abs(atan2(z, x)) > 90 degrees is behind the robot
                if (np.abs(bearing[0]) > (np.pi / 2) + self._tol) and (np.abs(bearing[1]) > (np.pi / 2) + self._tol):
                    return (pt, bearing)

    def get_random_robot_pose(self, local_frame: str) -> POSE_TYPES:
        """Returns a random, feasible robot pose located on a corner in the
        grid.

        Note: this will not sample any points on the edge of the grid

        Returns:
            SE2Pose: a random, feasible robot pose
        """
        # TODO: Extend to 3D domain

        if (self.dim == DIM.TWO):
            feasible_x_vals = (self._min_x_coord_feasible < self._x_coords) & (
                self._x_coords < self._max_x_coord_feasible
            )
            feasible_y_vals = (self._min_y_coord_feasible < self._y_coords) & (
                self._y_coords < self._max_y_coord_feasible
            )

            cornered_x_vals = np.zeros(feasible_x_vals.shape).astype(bool)
            cornered_y_vals = np.zeros(feasible_y_vals.shape).astype(bool)
            for i in range(len(cornered_x_vals)):
                if i % self._x_steps_to_intersection == 0:
                    cornered_x_vals[i] = True
            for j in range(len(cornered_y_vals)):
                if j % self._y_steps_to_intersection == 0:
                    cornered_y_vals[j] = True

            sampleable_x_vals = cornered_x_vals & feasible_x_vals
            sampleable_y_vals = cornered_y_vals & feasible_y_vals

            x_sample = np.random.choice(self._x_coords[sampleable_x_vals])
            y_sample = np.random.choice(self._y_coords[sampleable_y_vals])

            # pick a rotation from 0 to 3/2 pi
            rotation_sample = np.random.choice(np.linspace(0, (3 / 2) * np.pi, num=4))

            return SE2Pose(
                x_sample,
                y_sample,
                rotation_sample,
                local_frame=local_frame,
                base_frame="world",
            )
        else:
            feasible_x_vals = (self._min_x_coord_feasible < self._x_coords) & (
                self._x_coords < self._max_x_coord_feasible
            )
            feasible_y_vals = (self._min_y_coord_feasible < self._y_coords) & (
                self._y_coords < self._max_y_coord_feasible
            )
            feasible_z_vals = (self._min_z_coord_feasible < self._z_coords) & (
                self._z_coords < self._max_z_coord_feasible
            )

            cornered_x_vals = np.zeros(feasible_x_vals.shape).astype(bool)
            cornered_y_vals = np.zeros(feasible_y_vals.shape).astype(bool)
            cornered_z_vals = np.zeros(feasible_z_vals.shape).astype(bool)
            for i in range(len(cornered_x_vals)):
                if i % self._x_steps_to_intersection == 0:
                    cornered_x_vals[i] = True
            for j in range(len(cornered_y_vals)):
                if j % self._y_steps_to_intersection == 0:
                    cornered_y_vals[j] = True
            for k in range(len(cornered_z_vals)):
                if k % self._z_steps_to_intersection == 0:
                    cornered_z_vals[k] = True

            sampleable_x_vals = cornered_x_vals & feasible_x_vals
            sampleable_y_vals = cornered_y_vals & feasible_y_vals
            sampleable_z_vals = cornered_z_vals & feasible_z_vals

            x_sample = np.random.choice(self._x_coords[sampleable_x_vals])
            y_sample = np.random.choice(self._y_coords[sampleable_y_vals])
            z_sample = np.random.choice(self._z_coords[sampleable_z_vals])

            # pick a rotation from 0 to 3/2 pi
            rotation_sample = np.random.choice(np.linspace(0, (3 / 2) * np.pi, num=4))

            return SE3Pose(
                x_sample,
                y_sample,
                z_sample,
                rotation_sample,
                local_frame=local_frame,
                base_frame="world",
            )

    def get_random_beacon_point(self, frame: str) -> Union[Optional[Point2], Optional[Point3]]:
        """Returns a random beacon point on the grid.

        Args:
            frame (str): the frame of the beacon

        Returns:
            Optional[Point2]: a random valid beacon point, None if no position
                is feasible
        """

        # * this is also somewhat naive but it works... could revisit this later

        if (self.dim == DIM.TWO):
            # get random beacon position by generating all possible coordinates and
            # then just pruning those that are not feasible for the beacon
            x_idxs = np.arange(self._num_x_pts)
            y_idxs = np.arange(self._num_y_pts)

            # combination of all possible verts on the grid
            possible_verts = itertools.product(x_idxs, y_idxs)  # cartesian product

            # prune out the infeasible vertices
            feasible_verts = [
                vert for vert in possible_verts if self.vertex_is_beacon_feasible(vert)
            ]

            assert len(feasible_verts) > 0, "No feasible beacon positions"

            # randomly sample one of the vertices
            vert_sample = choice(feasible_verts)
            assert len(vert_sample) == 2

            i, j = vert_sample
            position = Point2(self._xv[i, j], self._yv[i, j], frame=frame)
            return position
        else:
            x_idxs = np.arange(self._num_x_pts)
            y_idxs = np.arange(self._num_y_pts)
            z_idxs = np.arange(self._num_z_pts)

            # combination of all possible verts on the grid
            possible_verts = itertools.product(x_idxs, y_idxs, z_idxs)

            # print(self._xv)
            # print(self._yv)
            # print(self._zv)

            print(self._robot_feasibility)

            # prune out the infeasible vertices
            feasible_verts = [
                vert for vert in possible_verts if self.vertex_is_beacon_feasible(vert)
            ]

            assert len(feasible_verts) > 0, "No feasible beacon positions"

            # randomly sample one of the vertices
            vert_sample = choice(feasible_verts)
            assert len(vert_sample) == 3

            i, j, k = vert_sample
            position = Point3(self._xv[i, j, k], self._yv[i, j, k], self._zv[i, j, k], frame=frame)
            return position

    ###### Coordinate and vertex conversion methods ######

    def coordinate2vertex(self, x: float, y: float, z: float = 0) -> VERTEX_TYPES:
        """Takes a coordinate and returns the corresponding vertex. Requires the
        coordinate correspond to a valid vertex.

        Args:
            x (float): x-coordinate
            y (float): y-coordinate

        Raises:
            ValueError: the coordinate does not correspond to a valid vertex

        Returns:
            Tuple[int, int]: the corresponding vertex indices
        """
        if (self.dim == DIM.TWO):
            i, dx, x_close = _find_nearest(self._x_coords, x)
            j, dy, y_close = _find_nearest(self._y_coords, y)
            if abs(dx) < self._tol and abs(dy) < self._tol:
                return (i, j)
            else:
                raise ValueError(
                    f"The input ({str(x)}, {str(y)}) is off grid vertices."
                )
        else:
            i, dx, x_close = _find_nearest(self._x_coords, x)
            j, dy, y_close = _find_nearest(self._y_coords, y)
            k, dz, z_close = _find_nearest(self._z_coords, z)
            if abs(dx) < self._tol and abs(dy) < self._tol and abs(dz) < self._tol:
                return (i, j, k)
            else:
                raise ValueError(
                    f"The input ({str(x)}, {str(y)}, {str(z)}) is off grid vertices."
                )

    def coordinates2vertices(
        self, coords: COORDINATE_LIST_TYPES
    ) -> VERTEX_LIST_TYPES:
        """Takes in a list of coordinates and returns a list of the respective
        corresponding vertices

        Args:
            coords (List[Tuple[int, int]]): list of coordinates

        Returns:
            List[Tuple[int, int]]: list of vertices
        """

        assert len(coords) >= 1
        if (self.dim == DIM.TWO):
            assert all(len(c) == 2 for c in coords)
        else:
            assert all(len(c) == 3 for c in coords)

        nearest_vertices = [self.coordinate2vertex(*c) for c in coords]
        assert self.check_vertex_list_valid(nearest_vertices)
        return nearest_vertices

    def vertex2coordinate(self, vert: VERTEX_TYPES) -> COORDINATE_TYPES:
        """Takes a vertex and returns the corresponding coordinates

        Args:
            vert (Tuple[int, int]): (i, j) vertex

        Returns:
            Tuple[float, float]: (x, y) coordinates
        """
        # TODO: Extend to 3D domain

        assert self.check_vertex_valid(vert)

        if (self.dim == DIM.TWO):
            # print("xv: " + str(self._xv))
            # print("yv: " + str(self._yv))
            i, j = vert
            return (self._xv[i, j], self._yv[i, j])
        else:
            # print("xv: " + str(self._xv))
            # print("yv: " + str(self._yv))
            # print("zv: " + str(self._zv))
            i, j, k = vert
            return (self._xv[i, j, k], self._yv[i, j, k], self._zv[i, j, k])

    def vertices2coordinates(
        self, vertices: VERTEX_LIST_TYPES
    ) -> COORDINATE_LIST_TYPES:
        """Takes a list of vertices and returns a list of the corresponding coordinates

        Args:
            vertices (List[Tuple[int, int]]): list of (i, j) vertices

        Returns:
            List[Tuple[float, float]]: list of (x, y) coordinates
        """
        assert self.check_vertex_list_valid(vertices)
        return [self.vertex2coordinate(v) for v in vertices]

    def vertex2point(self, vert: VERTEX_TYPES) -> POINT_TYPES:
        """Takes a vertex and returns the corresponding point in the world frame

        Args:
            vert (Tuple[int, int]): (i, j) vertex

        Returns:
            Point2: point in the world frame
        """

        assert self.check_vertex_valid(vert)

        if (self.dim == DIM.TWO):
            x, y = self.vertex2coordinate(vert)
            return Point2(float(x), float(y), frame="world")
        else:
            x, y, z = self.vertex2coordinate(vert)
            return Point3(float(x), float(y), float(z), frame="world")

    def point2vertex(self, point: POINT_TYPES) -> VERTEX_TYPES:
        """Takes a point in the world frame and returns the corresponding
        vertex

        Args:
            point (Point2): point in the world frame

        Returns:
            Tuple[int, int]: (i, j) vertex
        """

        assert point.frame == "world"

        # z is set to 0 for Point2
        x, y, z = point.x, point.y, point.z
        return self.coordinate2vertex(x, y, z)

    ####### Check vertex validity #########

    def pose_is_robot_feasible(self, pose: POSE_TYPES) -> bool:
        """Takes in a pose and returns whether the robot is feasible at that
        pose. Checks that rotation is a multiple of pi/2, that the
        position is on a robot feasible point in the grid

        Args:
            pose (SE2Pose): the pose to check

        Returns:
            bool: True if the robot is feasible at that pose, False otherwise
        """

        if (self.dim == DIM.TWO):
            rotation_is_good = abs(pose.theta % (np.pi / 2.0)) < self._tol
            if not rotation_is_good:
                print(f"Rotation is {pose.theta} and not a multiple of pi/2")
                return False

            vert = self.coordinate2vertex(pose.x, pose.y)
            if not self.vertex_is_robot_feasible(vert):
                print(f"Coordinate {pose.x}, {pose.y} from vertex {vert} is not feasible")
                return False
        else:
            roll, pitch, yaw = pose.rot.angles
            assert len(pose.rot.angles) == 3

            roll_is_good = abs(roll % (np.pi / 2.0)) < self._tol
            pitch_is_good = abs(pitch % (np.pi / 2.0)) < self._tol
            yaw_is_good = abs(yaw % (np.pi / 2.0)) < self._tol

            if not (roll_is_good and pitch_is_good and yaw_is_good):
                print(f"Rotation is {pose.rot.angles}; not a multiple of pi/2")
                return False

            vert = self.coordinate2vertex(pose.x, pose.y, pose.z)
            if not self.vertex_is_robot_feasible(vert):
                print(f"Coordinate {pose.x}, {pose.y}, {pose.z} from vertex {vert} is not feasible")
                return False
            
        return True

    def position_is_beacon_feasible(self, position: POINT_TYPES) -> bool:
        """Takes in a position and returns whether the position is feasible
        for a beacon.

        Args:
            position (Point2): the position to check

        Returns:
            bool: True if the position is feasible for a beacon, False otherwise
        """

        if (self.dim == DIM.TWO):
            vert = self.coordinate2vertex(position.x, position.y)
            return self.vertex_is_beacon_feasible(vert)
        else:
            vert = self.coordinate2vertex(position.x, position.y, position.z)
            return self.vertex_is_beacon_feasible(vert)

    def vertex_is_beacon_feasible(self, vert: VERTEX_TYPES) -> bool:
        """Returns whether the vertex is feasible for beacons.

        Args:
            vert (Tuple[int, int]): vertex to be checked

        Returns:
            bool: True if vertex is feasible for beacons, False otherwise
        """
        assert self.check_vertex_valid(vert)

        # if not a robot travelable location then it is good for a beacon
        return not self.vertex_is_robot_feasible(vert)

    def vertex_is_robot_feasible(self, vert: VERTEX_TYPES) -> bool:
        """Returns whether the vertex is feasible for robot. This checks whether
        the index of the vertex would be on one of the allowed lines the robot
        can travel on and then returns whether this is within the defined
        'feasible region'

        Args:
            vert (Tuple[int, int]): vertex to be checked

        Returns:
            bool: True if the vertex is feasible for robot, False otherwise
        """

        assert self.check_vertex_valid(vert)

        if (self.dim == DIM.TWO):
            i, j = vert

            # vertex can only be feasible if on one of the lines defined by the
            # row/column spacing
            if (
                i % self._x_steps_to_intersection == 0
                or j % self._y_steps_to_intersection == 0
            ):
                return self._robot_feasibility[i, j]
            else:
                return False
        else:
            i, j, k = vert

            # vertex can only be feasible if on one of the lines defined by the
            # row/column/height spacing
            if (
                i % self._x_steps_to_intersection == 0
                or j % self._y_steps_to_intersection == 0
                or k % self._z_steps_to_intersection == 0
            ):
                return self._robot_feasibility[i, j, k]
            else:
                return False

    def vertex_is_in_bounds(self, vert: VERTEX_TYPES) -> bool:
        if (self.dim == DIM.TWO):
            assert len(vert) == 2

            x_in_bounds = 0 <= vert[0] < self._num_x_pts
            y_in_bounds = 0 <= vert[1] < self._num_y_pts
            return x_in_bounds and y_in_bounds
        else:
            assert len(vert) == 3

            x_in_bounds = 0 <= vert[0] < self._num_x_pts
            y_in_bounds = 0 <= vert[1] < self._num_y_pts
            z_in_bounds = 0 <= vert[2] < self._num_z_pts
            return x_in_bounds and y_in_bounds and z_in_bounds

    def check_vertex_valid(self, vert: VERTEX_TYPES):
        """Checks that the indices of the vertex are within the bounds of the grid

        Args:
            vert (tuple): (i, j) indices of the vertex

        Returns:
            bool: True if the vertex is valid, False otherwise
        """

        if self.dim == DIM.TWO:
            assert len(vert) == 2, f"vert: {vert}, len: {len(vert)}"
            assert 0 <= vert[0] < self._num_x_pts
            assert 0 <= vert[1] < self._num_y_pts
        else:
            assert len(vert) == 3, f"vert: {vert}, len: {len(vert)}"
            assert 0 <= vert[0] < self._num_x_pts
            assert 0 <= vert[1] < self._num_y_pts
            assert 0 <= vert[2] < self._num_z_pts
        return True

    def check_vertex_list_valid(self, vertices: AREA_TYPES):
        """Checks that the indices of the vertex list are within the bounds of the grid

        Args:
            vertices (List[tuple]): list of vertices
        """
        assert all(self.check_vertex_valid(v) for v in vertices)
        return True

    ####### visualization #############

    def plot_environment(self, ax: Axes):
        # TODO: Extend to 3D domain

        if (self.dim == DIM.TWO):
            assert self._robot_feasibility.shape == (self._num_x_pts, self._num_y_pts)

            # get rows and cols that the robot is allowed to travel on
            x_pts = np.arange(self._num_x_pts)
            valid_x = x_pts[x_pts % self._x_steps_to_intersection == 0]
            valid_x = self._scale * valid_x

            y_pts = np.arange(self._num_y_pts)
            valid_y = y_pts[y_pts % self._y_steps_to_intersection == 0]
            valid_y = self._scale * valid_y

            # the bounds of the valid x and y values
            max_x = np.max(valid_x)
            min_x = np.min(valid_x)
            max_y = np.max(valid_y)
            min_y = np.min(valid_y)

            # plot the travelable rows and columns
            ax.vlines(valid_x, min_y, max_y)
            ax.hlines(valid_y, min_x, max_x)

            for i in range(self._num_x_pts):
                for j in range(self._num_y_pts):

                    # the robot should not be traveling on these locations
                    if (
                        i % self._x_steps_to_intersection != 0
                        and j % self._y_steps_to_intersection != 0
                    ):
                        continue

                    if self._robot_feasibility[i, j]:
                        ax.plot(self._xv[i, j], self._yv[i, j], "ro", markersize=3)
                    else:
                        ax.plot(self._xv[i, j], self._yv[i, j], "go", markersize=3)
        else:
            assert self._robot_feasibility.shape == (
                self._num_x_pts,
                self._num_y_pts,
                self._num_z_pts,
            )

            # get rows and cols that the robot is allowed to travel on
            x_pts = np.arange(self._num_x_pts)
            valid_x = x_pts[x_pts % self._x_steps_to_intersection == 0]
            valid_x = self._scale * valid_x

            y_pts = np.arange(self._num_y_pts)
            valid_y = y_pts[y_pts % self._y_steps_to_intersection == 0]
            valid_y = self._scale * valid_y

            z_pts = np.arange(self._num_z_pts)
            valid_z = z_pts[z_pts % self._z_steps_to_intersection == 0]
            valid_z = self._scale * valid_z

            # the bounds of the valid x and y values
            max_x = np.max(valid_x)
            min_x = np.min(valid_x)
            max_y = np.max(valid_y)
            min_y = np.min(valid_y)
            max_z = np.max(valid_z)
            min_z = np.min(valid_z)

            # plot the travelable rows and columns
            # ax.vlines(valid_x, min_y, max_y)
            # ax.hlines(valid_y, min_x, max_x)
            # ax.
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_zlim(min_z, max_z)

            for i in range(self._num_x_pts):
                for j in range(self._num_y_pts):
                    for k in range(self._num_z_pts):

                        # the robot should not be traveling on these locations
                        if (
                            i % self._x_steps_to_intersection != 0
                            and j % self._y_steps_to_intersection != 0
                            and k % self._z_steps_to_intersection != 0
                        ):
                            continue

                        if self._robot_feasibility[i, j, k]:
                            ax.plot(
                                self._xv[i, j, k],
                                self._yv[i, j, k],
                                self._zv[i, j, k],
                                "ro",
                                markersize=3,
                            )
                        else:
                            ax.plot(
                                self._xv[i, j, k],
                                self._yv[i, j, k],
                                self._zv[i, j, k],
                                "go",
                                markersize=3,
                            )
