import sys
sys.path.append('../')

import math
import unittest

from manhattan.geometry.Elements import Point, Point2, Point3, Rot, Rot2, Rot3, SEPose, SE2Pose, SE3Pose
from manhattan.environment.environment import ManhattanWorld
FRAME_1 = "odom"
FRAME_2 = "world"
FRAME_3 = "tool"

theta_1 = math.pi / 2
test_rot_1 = Rot2(theta_1, FRAME_1, FRAME_2)

theta_2 = math.pi / 3
test_rot_2 = Rot2(theta_2, FRAME_3, FRAME_2)

x_1 = 42.5123
y_1 = 23.4530
test_point_1 = Point2(x_1, y_1, FRAME_2)

x_2 = 12.0923
y_2 = 9.576
test_point_2 = Point2(x_2, y_2, FRAME_2)

test_pose_1 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, FRAME_1, FRAME_2)
test_pose_2 = SE2Pose.by_point_and_rotation(test_point_2, test_rot_2, FRAME_3, FRAME_2)

x_3 = 12.1324
y_3 = 85.4509
z_3 = 76.3453
test_point_3 = Point3(x_3, y_3, z_3, FRAME_2)

x_4 = 45.2312
y_4 = 71.2356
z_4 = 8.8643
test_point_4 = Point3(x_4, y_4, z_4, FRAME_2)

roll_3 = math.pi
pitch_3 = math.pi / 3
yaw_3 = math.pi / 2
test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, FRAME_1, FRAME_2)

roll_4 = math.pi / 6
pitch_4 = math.pi / 3
yaw_4 = math.pi / 2
test_rot_4 = Rot3(roll_4, pitch_4, yaw_4, FRAME_3, FRAME_2)

test_pose_3 = SE3Pose.by_point_and_rotation(test_point_3, test_rot_3, FRAME_1, FRAME_2)
test_pose_4 = SE3Pose.by_point_and_rotation(test_point_4, test_rot_4, FRAME_3, FRAME_2)

manhat_2d = ManhattanWorld()
manhat_3d = ManhattanWorld(dim=3, grid_vertices_shape=(9, 9, 9))
manhat_scaled_2d = ManhattanWorld(cell_scale = 2.0)
manhat_scaled_3d = ManhattanWorld(dim=3, grid_vertices_shape=(9, 9, 9), cell_scale = 2.0)

class TestValidity(unittest.TestCase):
    def test_check_vertex_valid(self):
        # Ensure only vertices of same dimension are accepted
        with self.assertRaises(AssertionError):
            manhat_2d.check_vertex_valid((1, 1, 1))
        with self.assertRaises(AssertionError):
            manhat_3d.check_vertex_valid((1, 1))

        # Ensure vertex indices are within the grid
        with self.assertRaises(AssertionError):
            manhat_2d.check_vertex_valid((10, 10))
        with self.assertRaises(AssertionError):
            manhat_3d.check_vertex_valid((10, 10, 10))
        
        # Ensure vertex indices are positive
        with self.assertRaises(AssertionError):
            manhat_2d.check_vertex_valid((-1, -1))
        with self.assertRaises(AssertionError):
            manhat_3d.check_vertex_valid((-1, -1, -1))
        
        self.assertTrue(manhat_2d.check_vertex_valid((1, 1)))
        self.assertTrue(manhat_3d.check_vertex_valid((1, 1, 1)))
        

    def test_check_vertex_list_valid(self):
        # Ensure only vertices of same dimension are accepted
        with self.assertRaises(AssertionError):
            manhat_2d.check_vertex_list_valid([(1, 1, 1)])
        with self.assertRaises(AssertionError):
            manhat_3d.check_vertex_list_valid([(1, 1)])

        # Ensure vertex indices are within the grid
        with self.assertRaises(AssertionError):
            manhat_2d.check_vertex_list_valid([(10, 10)])
        with self.assertRaises(AssertionError):
            manhat_3d.check_vertex_list_valid([(10, 10, 10)])
        
        # Ensure vertex indices are positive
        with self.assertRaises(AssertionError):
            manhat_2d.check_vertex_list_valid([(-1, -1)])
        with self.assertRaises(AssertionError):
            manhat_3d.check_vertex_list_valid([(-1, -1, -1)])

        # Ensure assertion error when at least one vertex is out of bounds
        with self.assertRaises(AssertionError):
            manhat_2d.check_vertex_list_valid([(1, 1), (2, 2), (10, 10)])
        with self.assertRaises(AssertionError):
            manhat_3d.check_vertex_list_valid([(1, 1, 1), (2, 2, 2), (10, 10, 10)])
        
        self.assertTrue(manhat_2d.check_vertex_list_valid([(1, 1), (2, 2), (3, 3)]))
        self.assertTrue(manhat_3d.check_vertex_list_valid([(1, 1, 1), (2, 2, 2), (3, 3, 3)]))
    
class TestConversion(unittest.TestCase):
    def test_coordinate2vertex(self):
        # Ensure coordinates beyond tolerance are not mapped to a vertex
        with self.assertRaises(ValueError):
            manhat_2d.coordinate2vertex(1.1, 1.2)
        with self.assertRaises(ValueError):
            manhat_3d.coordinate2vertex(1.2, 1.3, 1.4)

        self.assertTrue(manhat_2d.coordinate2vertex(1.0, 1.0) == (1, 1))
        self.assertTrue(manhat_2d.coordinate2vertex(1.000001, 1.000001) == (1, 1))
        self.assertTrue(manhat_3d.coordinate2vertex(1, 1, 1) == (1, 1, 1))
        self.assertTrue(manhat_3d.coordinate2vertex(1.000001, 1.000001, 1.000001) == (1, 1, 1))

    def test_coordinates2vertices(self):
        # Ensure coordinates beyond tolerance are not mapped to a vertex
        with self.assertRaises(ValueError):
            manhat_2d.coordinates2vertices([(1.1, 1.2), (1.3, 1.4)])
        with self.assertRaises(ValueError):
            manhat_3d.coordinates2vertices([(1.2, 1.3, 1.4), (1.5, 1.6, 1.7)])

        self.assertTrue(manhat_2d.coordinates2vertices([(1.0, 1.0), (1.000001, 1.000001)]) == [(1, 1), (1, 1)])
        self.assertTrue(manhat_3d.coordinates2vertices([(1, 1, 1), (1.000001, 1.000001, 1.000001)]) == [(1, 1, 1), (1, 1, 1)])
    
    def test_vertex2coordinate(self):
        # Ensure vertex is valid
        with self.assertRaises(AssertionError):
            manhat_2d.vertex2coordinate((10, 10))
        with self.assertRaises(AssertionError):
            manhat_3d.vertex2coordinate((10, 10, 10))

        self.assertTrue(manhat_2d.vertex2coordinate((1, 1)) == (1.0, 1.0))
        self.assertTrue(manhat_3d.vertex2coordinate((1, 1, 1)) == (1.0, 1.0, 1.0))
        self.assertTrue(manhat_scaled_2d.vertex2coordinate((1, 1)) == (2.0, 2.0))
        self.assertTrue(manhat_scaled_3d.vertex2coordinate((1, 1, 1)) == (2.0, 2.0, 2.0))

    def test_vertices2coordinates(self):
        # Ensure vertex is valid
        with self.assertRaises(AssertionError):
            manhat_2d.vertices2coordinates([(10, 10), (10, 10)])
        with self.assertRaises(AssertionError):
            manhat_3d.vertices2coordinates([(10, 10, 10), (10, 10, 10)])
        
        self.assertTrue(manhat_2d.vertices2coordinates([(1, 1), (2, 2)]) == [(1.0, 1.0), (2.0, 2.0)])
        self.assertTrue(manhat_3d.vertices2coordinates([(1, 1, 1), (2, 2, 2)]) == [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)])
        self.assertTrue(manhat_scaled_2d.vertices2coordinates([(1, 1), (2, 2)]) == [(2.0, 2.0), (4.0, 4.0)])
        self.assertTrue(manhat_scaled_3d.vertices2coordinates([(1, 1, 1), (2, 2, 2)]) == [(2.0, 2.0, 2.0), (4.0, 4.0, 4.0)])

    def test_vertex2point(self):
        # Ensure vertex is valid
        with self.assertRaises(AssertionError):
            manhat_2d.vertex2point((10, 10))
        with self.assertRaises(AssertionError):
            manhat_3d.vertex2point((10, 10, 10))

        self.assertTrue(manhat_2d.vertex2point((1, 1)) == Point2(1.0, 1.0, FRAME_2))
        self.assertTrue(manhat_3d.vertex2point((1, 1, 1)) == Point3(1.0, 1.0, 1.0, FRAME_2))
        self.assertTrue(manhat_scaled_2d.vertex2point((1, 1)) == Point2(2.0, 2.0, FRAME_2))
        self.assertTrue(manhat_scaled_3d.vertex2point((1, 1, 1)) == Point3(2.0, 2.0, 2.0, FRAME_2))

    def test_point2vertex(self):
        # Ensure point is valid
        with self.assertRaises(AssertionError):
            manhat_2d.point2vertex(Point2(10, 10, FRAME_2))
        with self.assertRaises(AssertionError):
            manhat_3d.point2vertex(Point3(10, 10, 10, FRAME_2))

        # Ensure frame is "world"
        with self.assertRaises(AssertionError):
            manhat_2d.point2vertex(Point2(1.0, 1.0, FRAME_1))
        with self.assertRaises(AssertionError):
            manhat_3d.point2vertex(Point3(1.0, 1.0, 1.0, FRAME_1))

        self.assertTrue(manhat_2d.point2vertex(Point2(1.0, 1.0, FRAME_2)) == (1, 1))
        self.assertTrue(manhat_3d.point2vertex(Point3(1.0, 1.0, 1.0, FRAME_2)) == (1, 1, 1))
        self.assertTrue(manhat_scaled_2d.point2vertex(Point2(2.0, 2.0, FRAME_2)) == (1, 1))
        self.assertTrue(manhat_scaled_3d.point2vertex(Point3(2.0, 2.0, 2.0, FRAME_2)) == (1, 1, 1))

if __name__ == "__main__":
    unittest.main()