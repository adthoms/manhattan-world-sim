import sys
sys.path.append('../')

import unittest
import math
import numpy as np
from manhattan.geometry.Elements import Point, Point2, Point3, Rot, Rot2, Rot3, SEPose, SE2Pose, SE3Pose

EPSILON = 1e-4
ODOM = "odom"

class TestGeometry(unittest.TestCase):
    # Unit tests per class

    # Point class:
    # Euclidean Norm (norm funct)
    def test_norm(self) -> None:
        x_1 = -3.5412
        y_1 = 0.8188
        test_point_1 = Point2(x_1, y_1, ODOM)
        self.assertTrue(test_point_1.norm - math.sqrt((x_1 ** 2) + (y_1 ** 2)) < EPSILON)

        x_2 = -0.4150
        y_2 = -0.5616
        z_2 = 0.1436
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)
        self.assertTrue(test_point_2.norm - math.sqrt((x_2 ** 2) + (y_2 ** 2) + (z_2 ** 2)) < EPSILON)

    # Distance between two points (static dist funct)
    def test_dist_static(self) -> None:
        x_1 = -0.8138
        y_1 = 2.2845
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 0.4049
        y_2 = 0.7406
        test_point_2 = Point2(x_2, y_2, ODOM)

        x_3 = -0.4150
        y_3 = -0.5616
        z_3 = 0.1436
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)

        x_4 = -0.5606
        y_4 = 0.5571
        z_4 = 1.0
        test_point_4 = Point3(x_4, y_4, z_4, ODOM)

        with self.assertRaises(AssertionError):
            Point.dist(test_point_1, test_point_3)
        with self.assertRaises(AssertionError):
            Point.dist(test_point_2, test_point_4)
        
        self.assertTrue(Point.dist(test_point_1, test_point_1) < EPSILON)
        self.assertTrue(Point.dist(test_point_3, test_point_3) < EPSILON)

        self.assertTrue(Point.dist(test_point_1, test_point_2) - np.linalg.norm([x_1 - x_2, y_1 - y_2]) < EPSILON)
        self.assertTrue(Point.dist(test_point_3, test_point_4) - np.linalg.norm([x_3 - x_4, y_3 - y_4, z_3 - z_4]) < EPSILON)

    # Distance between two points (dist funct)
    def test_dist(self) -> None:
        x_1 = -0.8138
        y_1 = 2.2845
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 0.4049
        y_2 = 0.7406
        test_point_2 = Point2(x_2, y_2, ODOM)

        x_3 = -0.4150
        y_3 = -0.5616
        z_3 = 0.1436
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)

        x_4 = -0.5606
        y_4 = 0.5571
        z_4 = 1.0
        test_point_4 = Point3(x_4, y_4, z_4, ODOM)

        with self.assertRaises(AssertionError):
            test_point_1.distance(test_point_3)
        with self.assertRaises(AssertionError):
            test_point_2.distance(test_point_4)
        
        self.assertTrue(test_point_1.distance(test_point_1) < EPSILON)
        self.assertTrue(test_point_3.distance(test_point_3) < EPSILON)

        self.assertTrue(test_point_1.distance(test_point_2) - np.linalg.norm([x_1 - x_2, y_1 - y_2]) < EPSILON)
        self.assertTrue(test_point_3.distance(test_point_4) - np.linalg.norm([x_3 - x_4, y_3 - y_4, z_3 - z_4]) < EPSILON)

        # Test if non-static and static distance functions compute the same value
        self.assertTrue(test_point_1.distance(test_point_2) - Point.dist(test_point_1, test_point_2) < EPSILON)
        self.assertTrue(test_point_3.distance(test_point_4) - Point.dist(test_point_3, test_point_4) < EPSILON)

    # Array of points (array funct)
    def test_array(self) -> None:
        x_1 = -0.8138
        y_1 = 2.2845
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = -0.4150
        y_2 = -0.5616
        z_2 = 0.1436
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        self.assertTrue(np.array_equal(test_point_1.array, np.array([x_1, y_1])))
        self.assertTrue(np.array_equal(test_point_2.array, np.array([x_2, y_2, z_2])))


    # Create point from array (by_array funct, abstract)
    # Deep copy (make sure it is independent from original point) (copy funct, abstract)
    # Deep copy inverse (copy funct, abstract)
    # Add, sub, mul, div, etc. (abstract)

    # Point2 class:
    # Unit test the abstract functions of Point

    # Point3 class:
    # Unit test the abstract functions of Point

    # Rot class:
    # Chordal distance of two rotations (static dist function)
    # Angular representation (angles funct, abstract)
    # Create rotation from array (by_array funct, abstract)
    # Create rotation from matrix (by_matrix funct, abstract)
    # Create rotation from vector (by_exp_map funct, abstract)
    # Deep copy (abstract)
    # Deep copy inverse (abstract)
    # Rotate, unrotate, bearing to local, bearing to base (abstract)

    # Rot2 class:
    # Unit test the abstract functions of Rot

    # Rot3 class:
    # Unit test the abstract functions of Rot


    # SEPose class

if __name__ == "__main__":
    unittest.main()