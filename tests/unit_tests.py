import sys
sys.path.append('../')

import unittest
import math
import numpy as np
from manhattan.geometry.Elements import Point, Point2, Point3, Rot, Rot2, Rot3, SEPose, SE2Pose, SE3Pose

_TRANSLATION_TOLERANCE = 1e-4
_ROTATION_TOLERANCE = 1e-4
ODOM = "odom"
WORLD = "world"

class TestGeometry(unittest.TestCase):
    # Unit tests per class

    # Point class:
    # Euclidean Norm (norm funct)
    def test_point_norm(self) -> None:
        x_1 = -3.5412
        y_1 = 0.8188
        test_point_1 = Point2(x_1, y_1, ODOM)
        self.assertTrue(test_point_1.norm - math.sqrt((x_1 ** 2) + (y_1 ** 2)) < _TRANSLATION_TOLERANCE)

        x_2 = -0.4150
        y_2 = -0.5616
        z_2 = 0.1436
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)
        self.assertTrue(test_point_2.norm - math.sqrt((x_2 ** 2) + (y_2 ** 2) + (z_2 ** 2)) < _TRANSLATION_TOLERANCE)

    # Distance between two points (static dist funct)
    def test_point_dist_static(self) -> None:
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
        
        self.assertTrue(Point.dist(test_point_1, test_point_1) < _TRANSLATION_TOLERANCE)
        self.assertTrue(Point.dist(test_point_3, test_point_3) < _TRANSLATION_TOLERANCE)

        self.assertTrue(Point.dist(test_point_1, test_point_2) - np.linalg.norm([x_1 - x_2, y_1 - y_2]) < _TRANSLATION_TOLERANCE)
        self.assertTrue(Point.dist(test_point_3, test_point_4) - np.linalg.norm([x_3 - x_4, y_3 - y_4, z_3 - z_4]) < _TRANSLATION_TOLERANCE)

    # Distance between two points (dist funct)
    def test_point_dist(self) -> None:
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
        
        self.assertTrue(test_point_1.distance(test_point_1) < _TRANSLATION_TOLERANCE)
        self.assertTrue(test_point_3.distance(test_point_3) < _TRANSLATION_TOLERANCE)

        self.assertTrue(test_point_1.distance(test_point_2) - np.linalg.norm([x_1 - x_2, y_1 - y_2]) < _TRANSLATION_TOLERANCE)
        self.assertTrue(test_point_3.distance(test_point_4) - np.linalg.norm([x_3 - x_4, y_3 - y_4, z_3 - z_4]) < _TRANSLATION_TOLERANCE)

        # Test if non-static and static distance functions compute the same value
        self.assertTrue(test_point_1.distance(test_point_2) - Point.dist(test_point_1, test_point_2) < _TRANSLATION_TOLERANCE)
        self.assertTrue(test_point_3.distance(test_point_4) - Point.dist(test_point_3, test_point_4) < _TRANSLATION_TOLERANCE)

    # Array of points (array funct)
    def test_point_array(self) -> None:
        x_1 = -0.8138
        y_1 = 2.2845
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = -0.4150
        y_2 = -0.5616
        z_2 = 0.1436
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        self.assertTrue(np.array_equal(test_point_1.array, np.array([x_1, y_1])))
        self.assertTrue(np.array_equal(test_point_2.array, np.array([x_2, y_2, z_2])))

    # Equality of Points
    def test_point_eq(self) -> None:
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)
        test_point_2 = Point2(x_1, y_1, WORLD)

        x_3 = 6.5409
        y_3 = 23.9728
        z_3 = 54.1029
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)
        test_point_4 = Point3(x_3, y_3, z_3, WORLD)

        # Ensure that differing dimensions cannot be tested for equality
        self.assertFalse(test_point_1 == test_point_3)
        self.assertFalse(test_point_1 == test_point_4)

        # Ensure that differing frames cannot be tested for equality
        self.assertFalse(test_point_1 == test_point_2)
        self.assertFalse(test_point_3 == test_point_4)

        self.assertTrue(test_point_1 == test_point_1.copy())
        self.assertTrue(test_point_2 == test_point_2.copy())
        self.assertTrue(test_point_3 == test_point_3.copy())
        self.assertTrue(test_point_4 == test_point_4.copy())

    # Create point from array (by_array funct, abstract)
    def test_point_by_array(self) -> None:
        x_1 = 16.5438
        y_1 = 5.2302
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 24.0923
        y_2 = 12.5469
        z_2 = 9.4521
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        self.assertTrue(test_point_1 == Point2.by_array(np.array([x_1, y_1]), ODOM)) # Must call from Point2, not Point
        self.assertTrue(test_point_2 == Point3.by_array(np.array([x_2, y_2, z_2]), ODOM)) # Must call from Point3, not Point

    # Deep copy (make sure it is independent from original point) (copy funct, abstract)
    def test_point_copy(self) -> None:
        x_1 = 13.1329
        y_1 = 2.0197
        test_point_1 = Point2(x_1, y_1, ODOM)
        test_point_2 = test_point_1.copy()

        x_3 = 5.4756
        y_3 = 15.6423
        z_3 = 7.9432
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)
        test_point_4 = test_point_3.copy()

        self.assertTrue(test_point_1 == test_point_2)
        self.assertTrue(test_point_3 == test_point_4)

        test_point_2.x = 9.8741
        test_point_2.y = 0.7835
        test_point_4.x = 5.9784
        test_point_4.y = 2.3940
        test_point_4.z = 7.1235

        self.assertFalse(test_point_1 == test_point_2)
        self.assertFalse(test_point_3 == test_point_4)

    # Deep copy inverse (copy funct, abstract)
    def test_point_copy_inv(self) -> None:
        x_1 = 13.1329
        y_1 = 2.0197
        test_point_1 = Point2(x_1, y_1, ODOM)
        test_point_2 = test_point_1.copyInverse()

        x_3 = 5.4756
        y_3 = 15.6423
        z_3 = 7.9432
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)
        test_point_4 = test_point_3.copyInverse()

        self.assertFalse(test_point_1 == test_point_2)
        self.assertFalse(test_point_3 == test_point_4)

        test_point_2.x = -test_point_2.x
        test_point_2.y = -test_point_2.y
        test_point_4.x = -test_point_4.x
        test_point_4.y = -test_point_4.y
        test_point_4.z = -test_point_4.z

        self.assertTrue(test_point_1 == test_point_2)
        self.assertTrue(test_point_3 == test_point_4)

    # Add, sub, mul, div, etc. (abstract)
    def test_point_add(self) -> None:
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.3254
        y_2 = 78.2376
        test_point_2 = Point2(x_2, y_2, ODOM)

        x_3 = 6.5409
        y_3 = 23.9728
        z_3 = 54.1029
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)

        x_4 = 12.1324
        y_4 = 85.4509
        z_4 = 76.3453
        test_point_4 = Point3(x_4, y_4, z_4, ODOM)

        test_point_5 = Point2(x_1, y_1, WORLD)
        test_point_6 = Point3(x_3, y_3, z_3, WORLD)

        # Ensure that differing dimensions cannot be added
        with self.assertRaises(AssertionError):
            test_point_1 + test_point_3
        with self.assertRaises(AssertionError):
            test_point_2 + test_point_4

        # Ensure that differing frames cannot be added
        with self.assertRaises(AssertionError):
            test_point_1 + test_point_5
        with self.assertRaises(AssertionError):
            test_point_3 + test_point_6

        self.assertTrue((test_point_1 + test_point_2) == Point2(x_1 + x_2, y_1 + y_2, ODOM))
        self.assertTrue((test_point_3 + test_point_4) == Point3(x_3 + x_4, y_3 + y_4, z_3 + z_4, ODOM))

    def test_point_iadd(self) -> None:
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.3254
        y_2 = 78.2376
        test_point_2 = Point2(x_2, y_2, ODOM)

        x_3 = 6.5409
        y_3 = 23.9728
        z_3 = 54.1029
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)

        x_4 = 12.1324
        y_4 = 85.4509
        z_4 = 76.3453
        test_point_4 = Point3(x_4, y_4, z_4, ODOM)

        test_point_5 = Point2(x_1, y_1, WORLD)
        test_point_6 = Point3(x_3, y_3, z_3, WORLD)

        # Ensure that differing dimensions cannot be added
        with self.assertRaises(AssertionError):
            test_point_1 += test_point_3
        with self.assertRaises(AssertionError):
            test_point_2 += test_point_4

        # Ensure that differing frames cannot be added
        with self.assertRaises(AssertionError):
            test_point_1 += test_point_5
        with self.assertRaises(AssertionError):
            test_point_3 += test_point_6
        
        test_point_1 += test_point_2
        test_point_3 += test_point_4

        self.assertTrue(test_point_1 == Point2(x_1 + x_2, y_1 + y_2, ODOM))
        self.assertTrue(test_point_3 == Point3(x_3 + x_4, y_3 + y_4, z_3 + z_4, ODOM))

    def test_point_sub(self) -> None:
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.3254
        y_2 = 78.2376
        test_point_2 = Point2(x_2, y_2, ODOM)

        x_3 = 6.5409
        y_3 = 23.9728
        z_3 = 54.1029
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)

        x_4 = 12.1324
        y_4 = 85.4509
        z_4 = 76.3453
        test_point_4 = Point3(x_4, y_4, z_4, ODOM)

        test_point_5 = Point2(x_1, y_1, WORLD)
        test_point_6 = Point3(x_3, y_3, z_3, WORLD)

        # Ensure that differing dimensions cannot be subtracted
        with self.assertRaises(AssertionError):
            test_point_1 - test_point_3
        with self.assertRaises(AssertionError):
            test_point_2 - test_point_4

        # Ensure that differing frames cannot be subtracted
        with self.assertRaises(AssertionError):
            test_point_1 - test_point_5
        with self.assertRaises(AssertionError):
            test_point_3 - test_point_6

        self.assertTrue((test_point_1 - test_point_2) == Point2(x_1 - x_2, y_1 - y_2, ODOM))
        self.assertTrue((test_point_3 - test_point_4) == Point3(x_3 - x_4, y_3 - y_4, z_3 - z_4, ODOM))

    def test_point_isub(self) -> None:
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.3254
        y_2 = 78.2376
        test_point_2 = Point2(x_2, y_2, ODOM)

        x_3 = 6.5409
        y_3 = 23.9728
        z_3 = 54.1029
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)

        x_4 = 12.1324
        y_4 = 85.4509
        z_4 = 76.3453
        test_point_4 = Point3(x_4, y_4, z_4, ODOM)

        test_point_5 = Point2(x_1, y_1, WORLD)
        test_point_6 = Point3(x_3, y_3, z_3, WORLD)

        # Ensure that differing dimensions cannot be subtracted
        with self.assertRaises(AssertionError):
            test_point_1 -= test_point_3
        with self.assertRaises(AssertionError):
            test_point_2 -= test_point_4

        # Ensure that differing frames cannot be subtracted
        with self.assertRaises(AssertionError):
            test_point_1 -= test_point_5
        with self.assertRaises(AssertionError):
            test_point_3 -= test_point_6
        
        test_point_1 -= test_point_2
        test_point_3 -= test_point_4

        self.assertTrue(test_point_1 == Point2(x_1 - x_2, y_1 - y_2, ODOM))
        self.assertTrue(test_point_3 == Point3(x_3 - x_4, y_3 - y_4, z_3 - z_4, ODOM))

    def test_point_mul(self) -> None:
        scalar_1 = 4.2305
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        scalar_2 = 2.2301
        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        scalar_zero = 0.0

        self.assertTrue(np.allclose((test_point_1 * scalar_1).array, np.array([x_1 * scalar_1, y_1 * scalar_1]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose((test_point_2 * scalar_2).array, np.array([x_2 * scalar_2, y_2 * scalar_2, z_2 * scalar_2]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose((test_point_1 * scalar_zero).array, np.array([x_1 * scalar_zero, y_1 * scalar_zero]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose((test_point_2 * scalar_zero).array, np.array([x_2 * scalar_zero, y_2 * scalar_zero, z_2 * scalar_zero]), _TRANSLATION_TOLERANCE))

    def test_point_rmul(self) -> None:
        scalar_1 = 4.2305
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        scalar_2 = 2.2301
        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        scalar_zero = 0.0

        self.assertTrue(np.allclose((scalar_1 * test_point_1).array, np.array([x_1 * scalar_1, y_1 * scalar_1]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose((scalar_2 * test_point_2).array, np.array([x_2 * scalar_2, y_2 * scalar_2, z_2 * scalar_2]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose((scalar_zero * test_point_1).array, np.array([x_1 * scalar_zero, y_1 * scalar_zero]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose((scalar_zero * test_point_2).array, np.array([x_2 * scalar_zero, y_2 * scalar_zero, z_2 * scalar_zero]), _TRANSLATION_TOLERANCE))

    def test_point_imul(self) -> None:
        scalar_1 = 4.2305
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        scalar_2 = 2.2301
        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        test_point_1 *= scalar_1
        test_point_2 *= scalar_2

        # Should mul and imul have int arguments too?

        self.assertTrue(np.allclose(test_point_1.array, np.array([x_1 * scalar_1, y_1 * scalar_1]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose(test_point_2.array, np.array([x_2 * scalar_2, y_2 * scalar_2, z_2 * scalar_2]), _TRANSLATION_TOLERANCE))

    # test truediv
    def test_point_truediv(self) -> None:
        scalar_1 = 4.2305
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        scalar_2 = 2.2301
        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        scalar_3 = 0.0

        with self.assertRaises(ValueError):
            test_point_1 / scalar_3
        with self.assertRaises(ValueError):
            test_point_2 / scalar_3
        
        self.assertTrue(np.allclose((test_point_1 / scalar_1).array, np.array([x_1 / scalar_1, y_1 / scalar_1]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose((test_point_2 / scalar_2).array, np.array([x_2 / scalar_2, y_2 / scalar_2, z_2 / scalar_2]), _TRANSLATION_TOLERANCE))

    def test_point_itruediv(self) -> None:
        scalar_1 = 4.2305
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)
        test_point_2 = test_point_1.copy()

        scalar_3 = 2.2301
        x_3 = 12.1324
        y_3 = 85.4509
        z_3 = 76.3453
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)
        test_point_4 = test_point_3.copy()

        scalar_4 = 0.0
        scalar_5 = 0 # Test if 0 integer raises an error
        scalar_6 = 2

        with self.assertRaises(ValueError):
            test_point_1 /= scalar_4
        with self.assertRaises(ValueError):
            test_point_2 /= scalar_4
        with self.assertRaises(ValueError):
            test_point_1 /= scalar_5
        with self.assertRaises(ValueError):
            test_point_2 /= scalar_5

        test_point_1 /= scalar_1
        test_point_3 /= scalar_3
        test_point_2 /= scalar_6
        test_point_4 /= scalar_6
        
        self.assertTrue(np.allclose(test_point_1.array, np.array([x_1 / scalar_1, y_1 / scalar_1]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose(test_point_3.array, np.array([x_3 / scalar_3, y_3 / scalar_3, z_3 / scalar_3]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose(test_point_2.array, np.array([x_1 / scalar_6, y_1 / scalar_6]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose(test_point_4.array, np.array([x_3 / scalar_6, y_3 / scalar_6, z_3 / scalar_6]), _TRANSLATION_TOLERANCE))

    def test_point_neg(self) -> None:
        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)
        
        self.assertTrue(np.allclose((-test_point_1).array, np.array([-x_1, -y_1]), _TRANSLATION_TOLERANCE))
        self.assertTrue(np.allclose((-test_point_2).array, np.array([-x_2, -y_2, -z_2]), _TRANSLATION_TOLERANCE))

    # Rot class:
    # Chordal distance of two rotations (static dist function)
    def test_rot_dist_static(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        theta_2 = math.pi / 3
        test_rot_2 = Rot2(theta_2, ODOM, WORLD)

        roll_3 = math.pi
        pitch_3 = math.pi / 2
        yaw_3 = math.pi / 4
        test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, ODOM, WORLD)

        roll_4 = math.pi / 6
        pitch_4 = math.pi / 3
        yaw_4 = math.pi / 2
        test_rot_4 = Rot3(roll_4, pitch_4, yaw_4, ODOM, WORLD)

        x_1 = 12.1324
        y_1 = 85.4509
        z_1 = 76.3453
        test_point_1 = Point3(x_1, y_1, z_1, ODOM)

        # Ensure that rotations of differing dimensions cannot be measured
        with self.assertRaises(AssertionError):
            Rot.dist(test_rot_1, test_rot_3)
        
        # Ensure that non-rotations cannot be measured
        with self.assertRaises(AssertionError):
            Rot.dist(test_rot_1, test_point_1)
        
        # Test if the local frame of the first rotation and the base frame of the second rotation is checked
        with self.assertRaises(AssertionError):
            Rot.dist(test_rot_1, test_rot_1.copyInverse())
        with self.assertRaises(AssertionError):
            Rot.dist(test_rot_3, test_rot_3.copyInverse())

        self.assertTrue(Rot.dist(test_rot_1, test_rot_1) < _ROTATION_TOLERANCE)
        self.assertTrue(Rot.dist(test_rot_1, test_rot_2) - np.linalg.norm((test_rot_1.copyInverse() * test_rot_2).log_map) < _ROTATION_TOLERANCE)
        self.assertTrue(Rot.dist(test_rot_3, test_rot_3) < _ROTATION_TOLERANCE)
        self.assertTrue(Rot.dist(test_rot_3, test_rot_4) - np.linalg.norm((test_rot_3.copyInverse() * test_rot_4).log_map) < _ROTATION_TOLERANCE)
        
    # Angular representation (angles funct, abstract)
    def test_rot_angles(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        # According to the implementation of SO3.pi, when pitch = pi/2, yaw = 0
        # This is because pitch = pi/2 or -pi/2 is the "singularity" of the
        # RPY angles representation for SO(3); there are infinitely many sets of
        # RPY angles for a given rotation matrix at those angles

        # Should I create a test case for pitch = pi/2?
        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        test_rot_2 = Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD)

        self.assertTrue(isinstance(test_rot_1.angles, float))
        self.assertTrue(isinstance(test_rot_2.angles, tuple))
        
        self.assertTrue(test_rot_1.angles == theta_1)
        self.assertTrue(test_rot_2.angles == (roll_2, pitch_2, yaw_2))

    # Create rotation from array (by_array funct, abstract)
    def test_rot_by_array(self) -> None:
        theta_1 = math.pi / 2
        so2_arr = np.array([theta_1])

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        so3_arr = np.array([roll_2, pitch_2, yaw_2])
        so3_tup = (roll_2, pitch_2, yaw_2)

        with self.assertRaises(AssertionError):
            Rot2.by_array(so3_arr, ODOM, WORLD)
        with self.assertRaises(AssertionError):
            Rot3.by_array(so2_arr, ODOM, WORLD)
        
        self.assertTrue(Rot2.by_array(so2_arr, ODOM, WORLD).angles - theta_1 < _ROTATION_TOLERANCE)
        self.assertTrue(Rot3.by_array(so3_arr, ODOM, WORLD).angles == so3_tup)

    # Create rotation from matrix (by_matrix funct, abstract)
    def test_rot_by_matrix(self) -> None:
        theta_1 = math.pi / 2
        so2_mat = np.array([[math.cos(theta_1), -1 * math.sin(theta_1)],
                            [math.sin(theta_1), math.cos(theta_1)]])

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        so3_arr = (roll_2, pitch_2, yaw_2)

        yaw_2_mat = np.array([[math.cos(yaw_2), -1 * math.sin(yaw_2), 0],
                              [math.sin(yaw_2), math.cos(yaw_2), 0],
                              [0, 0, 1]])
        pitch_2_mat = np.array([[math.cos(pitch_2), 0, math.sin(pitch_2)],
                                [0, 1, 0],
                                [-1 * math.sin(pitch_2), 0, math.cos(pitch_2)]])
        roll_2_mat = np.array([[1, 0, 0],
                               [0, math.cos(roll_2), -1 * math.sin(roll_2)],
                               [0, math.sin(roll_2), math.cos(roll_2)]])
        
        so3_mat = np.matmul(np.matmul(yaw_2_mat, pitch_2_mat), roll_2_mat)

        with self.assertRaises(AssertionError):
            Rot2.by_array(so3_mat, ODOM, WORLD)
        with self.assertRaises(AssertionError):
            Rot3.by_array(so2_mat, ODOM, WORLD)

        self.assertTrue(Rot2.by_matrix(so2_mat, ODOM, WORLD).angles - theta_1 < _ROTATION_TOLERANCE)
        self.assertTrue(Rot3.by_matrix(so3_mat, ODOM, WORLD).angles == so3_arr)

    # Create rotation from vector (by_exp_map funct, abstract)
    def test_rot_by_exp_map(self) -> None:
        theta_1 = math.pi / 2
        so2_arr = np.array([theta_1])

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        so3_arr = np.array([roll_2, pitch_2, yaw_2])
        so3_tup = (roll_2, pitch_2, yaw_2)

        with self.assertRaises(AssertionError):
            Rot2.by_exp_map(so3_arr, ODOM, WORLD)
        with self.assertRaises(AssertionError):
            Rot3.by_exp_map(so2_arr, ODOM, WORLD)
        
        self.assertTrue(Rot2.by_exp_map(so2_arr, ODOM, WORLD).angles - theta_1 < _ROTATION_TOLERANCE)

        # How to convert roll, pitch, and yaw to axis-angle vector representation?
        # print(Rot3.by_exp_map(so3_arr, ODOM, WORLD).angles)
        # print(so3_tup)
        # self.assertTrue(Rot3.by_exp_map(so3_arr, ODOM, WORLD).angles == so3_tup)
    
    # Deep copy (abstract)
    def test_rot_copy(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)
        test_rot_2 = test_rot_1.copy()

        roll_3 = math.pi
        pitch_3 = math.pi / 3
        yaw_3 = math.pi / 4
        test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, ODOM, WORLD)
        test_rot_4 = test_rot_3.copy()

        self.assertTrue(test_rot_1 == test_rot_2)
        self.assertTrue(test_rot_3 == test_rot_4)

        test_rot_2.theta = math.pi / 3

        test_rot_4.roll = math.pi / 6
        test_rot_4.pitch = math.pi
        test_rot_4.yaw = math.pi / 3

        self.assertFalse(test_rot_1 == test_rot_2)
        self.assertFalse(test_rot_3 == test_rot_4)

    # Deep copy inverse (abstract)
    def test_rot_copy_inv(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)
        test_rot_2 = test_rot_1.copyInverse()

        roll_3 = math.pi
        pitch_3 = math.pi / 3
        yaw_3 = math.pi / 4
        test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, ODOM, WORLD)
        test_rot_4 = test_rot_3.copyInverse()

        yaw_2_mat = np.array([[math.cos(yaw_3), -1 * math.sin(yaw_3), 0],
                              [math.sin(yaw_3), math.cos(yaw_3), 0],
                              [0, 0, 1]])
        pitch_2_mat = np.array([[math.cos(pitch_3), 0, math.sin(pitch_3)],
                                [0, 1, 0],
                                [-1 * math.sin(pitch_3), 0, math.cos(pitch_3)]])
        roll_2_mat = np.array([[1, 0, 0],
                               [0, math.cos(roll_3), -1 * math.sin(roll_3)],
                               [0, math.sin(roll_3), math.cos(roll_3)]])
        
        # v = Ryaw * Rpitch * Rroll * v_0
        # v_0 = Rroll^-1 * Rpitch^-1 * Ryaw^-1
        so3_mat = np.matmul(np.matmul(yaw_2_mat, pitch_2_mat), roll_2_mat)
        so3_mat_inv = np.matmul(np.matmul(np.linalg.inv(roll_2_mat), np.linalg.inv(pitch_2_mat)), np.linalg.inv(yaw_2_mat))

        self.assertFalse(test_rot_1 == test_rot_2)
        self.assertFalse(test_rot_3 == test_rot_4)

        self.assertTrue(Rot2(-theta_1, WORLD, ODOM) == test_rot_2)

        self.assertTrue(Rot3.by_matrix(so3_mat_inv, WORLD, ODOM) == test_rot_4)
        self.assertTrue(Rot3.by_matrix(so3_mat.T, WORLD, ODOM) == test_rot_4) # Inverse of rotation matrix == Transpose of rotation matrix, since SO3 is orthonormal

    # Rotate, unrotate, bearing to local, bearing to base (abstract)
    def test_rot_rotate_point(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        so2_mat = np.array([[math.cos(theta_1), -1 * math.sin(theta_1)],
                            [math.sin(theta_1), math.cos(theta_1)]])

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        test_rot_2 = Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD)

        yaw_2_mat = np.array([[math.cos(yaw_2), -1 * math.sin(yaw_2), 0],
                              [math.sin(yaw_2), math.cos(yaw_2), 0],
                              [0, 0, 1]])
        pitch_2_mat = np.array([[math.cos(pitch_2), 0, math.sin(pitch_2)],
                                [0, 1, 0],
                                [-1 * math.sin(pitch_2), 0, math.cos(pitch_2)]])
        roll_2_mat = np.array([[1, 0, 0],
                               [0, math.cos(roll_2), -1 * math.sin(roll_2)],
                               [0, math.sin(roll_2), math.cos(roll_2)]])
        so3_mat = np.matmul(np.matmul(yaw_2_mat, pitch_2_mat), roll_2_mat)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)
        test_point_2 = Point2(x_1, y_1, WORLD)

        x_3 = 12.1324
        y_3 = 85.4509
        z_3 = 76.3453
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)
        test_point_4 = Point3(x_3, y_3, z_3, WORLD)

        with self.assertRaises(AssertionError):
            test_rot_1.rotate_point(test_point_3)
        with self.assertRaises(AssertionError):
            test_rot_2.rotate_point(test_point_1)
        with self.assertRaises(AssertionError):
            test_rot_1.rotate_point(test_point_2)
        with self.assertRaises(AssertionError):
            test_rot_2.rotate_point(test_point_4)

        x_1_dot, y_1_dot = np.dot(so2_mat, test_point_1.array)[0], np.dot(so2_mat, test_point_1.array)[1]
        x_3_dot, y_3_dot, z_3_dot = np.dot(so3_mat, test_point_3.array)[0], np.dot(so3_mat, test_point_3.array)[1], np.dot(so3_mat, test_point_3.array)[2]
        
        self.assertTrue(test_rot_1.rotate_point(test_point_1) == Point2(x_1_dot, y_1_dot, WORLD))
        self.assertTrue(test_rot_2.rotate_point(test_point_3) == Point3(x_3_dot, y_3_dot, z_3_dot, WORLD))

    def test_rot_unrotate_point(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        so2_mat = np.array([[math.cos(theta_1), -1 * math.sin(theta_1)],
                            [math.sin(theta_1), math.cos(theta_1)]])
        so2_mat_inv = np.linalg.inv(so2_mat)

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        test_rot_2 = Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD)

        yaw_2_mat = np.array([[math.cos(yaw_2), -1 * math.sin(yaw_2), 0],
                              [math.sin(yaw_2), math.cos(yaw_2), 0],
                              [0, 0, 1]])
        pitch_2_mat = np.array([[math.cos(pitch_2), 0, math.sin(pitch_2)],
                                [0, 1, 0],
                                [-1 * math.sin(pitch_2), 0, math.cos(pitch_2)]])
        roll_2_mat = np.array([[1, 0, 0],
                               [0, math.cos(roll_2), -1 * math.sin(roll_2)],
                               [0, math.sin(roll_2), math.cos(roll_2)]])
        so3_mat_inv = np.matmul(np.matmul(np.linalg.inv(roll_2_mat), np.linalg.inv(pitch_2_mat)), np.linalg.inv(yaw_2_mat))

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, WORLD)
        test_point_2 = Point2(x_1, y_1, ODOM)

        x_3 = 12.1324
        y_3 = 85.4509
        z_3 = 76.3453
        test_point_3 = Point3(x_3, y_3, z_3, WORLD)
        test_point_4 = Point3(x_3, y_3, z_3, ODOM)

        with self.assertRaises(AssertionError):
            test_rot_1.unrotate_point(test_point_3)
        with self.assertRaises(AssertionError):
            test_rot_2.unrotate_point(test_point_1)
        with self.assertRaises(AssertionError):
            test_rot_1.unrotate_point(test_point_2)
        with self.assertRaises(AssertionError):
            test_rot_2.unrotate_point(test_point_4)

        x_1_dot, y_1_dot = np.dot(so2_mat_inv, test_point_1.array)[0], np.dot(so2_mat_inv, test_point_1.array)[1]
        x_3_dot, y_3_dot, z_3_dot = np.dot(so3_mat_inv, test_point_3.array)[0], np.dot(so3_mat_inv, test_point_3.array)[1], np.dot(so3_mat_inv, test_point_3.array)[2]

        self.assertTrue(test_rot_1.unrotate_point(test_point_1) == Point2(x_1_dot, y_1_dot, ODOM))
        self.assertTrue(test_rot_2.unrotate_point(test_point_3) == Point3(x_3_dot, y_3_dot, z_3_dot, ODOM))

    def test_rot_bearing_to_local(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        test_rot_2 = Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)
        test_point_2 = Point2(x_1, y_1, WORLD)

        x_3 = 12.1324
        y_3 = 85.4509
        z_3 = 76.3453
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)
        test_point_4 = Point3(x_3, y_3, z_3, WORLD)

        # Ensure differing dimensions cannot measure bearing
        with self.assertRaises(AssertionError):
            test_rot_1.bearing_to_local_frame_point(test_point_3)
        with self.assertRaises(AssertionError):
            test_rot_2.bearing_to_local_frame_point(test_point_1)

        # Ensure differing local frames cannot measure bearing
        with self.assertRaises(AssertionError):
            test_rot_1.bearing_to_local_frame_point(test_point_2)
        with self.assertRaises(AssertionError):
            test_rot_2.bearing_to_local_frame_point(test_point_4)

        self.assertTrue(test_rot_1.bearing_to_local_frame_point(test_point_1) - math.atan2(y_1, x_1) < _ROTATION_TOLERANCE)
        self.assertTrue(test_rot_2.bearing_to_local_frame_point(test_point_3) == (math.atan2(y_3, x_3), math.atan2(z_3, x_3)))


    def test_rot_bearing_to_base(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        test_rot_2 = Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, WORLD)
        test_point_2 = Point2(x_1, y_1, ODOM)

        x_3 = 12.1324
        y_3 = 85.4509
        z_3 = 76.3453
        test_point_3 = Point3(x_3, y_3, z_3, WORLD)
        test_point_4 = Point3(x_3, y_3, z_3, ODOM)

        # Ensure differing dimensions cannot measure bearing
        with self.assertRaises(AssertionError):
            test_rot_1.bearing_to_base_frame_point(test_point_3)
        with self.assertRaises(AssertionError):
            test_rot_2.bearing_to_base_frame_point(test_point_1)

        # Ensure differing base frames cannot measure bearing
        with self.assertRaises(AssertionError):
            test_rot_1.bearing_to_base_frame_point(test_point_2)
        with self.assertRaises(AssertionError):
            test_rot_2.bearing_to_base_frame_point(test_point_4)

        local_point_1 = test_rot_1.unrotate_point(test_point_1)
        local_point_2 = test_rot_2.unrotate_point(test_point_3)
        self.assertTrue(test_rot_1.bearing_to_base_frame_point(test_point_1) - math.atan2(local_point_1.y, local_point_1.x) < _ROTATION_TOLERANCE)
        self.assertTrue(test_rot_2.bearing_to_base_frame_point(test_point_3) == (math.atan2(local_point_2.y, local_point_2.x), math.atan2(local_point_2.z, local_point_2.x)))

    # SEPose class
    def test_pose_dist_static(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        theta_2 = math.pi / 3
        test_rot_2 = Rot2(theta_2, ODOM, WORLD)

        roll_3 = math.pi
        pitch_3 = math.pi / 3
        yaw_3 = math.pi / 4
        test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, ODOM, WORLD)

        roll_4 = math.pi / 6
        pitch_4 = math.pi / 3
        yaw_4 = math.pi / 2
        test_rot_4 = Rot3(roll_4, pitch_4, yaw_4, ODOM, WORLD)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        test_pose_1 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, ODOM, WORLD)
        test_pose_2 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_2, ODOM, WORLD)
        test_pose_3 = SE3Pose.by_point_and_rotation(test_point_2, test_rot_3, ODOM, WORLD)
        test_pose_4 = SE3Pose.by_point_and_rotation(test_point_2, test_rot_4, ODOM, WORLD)

        # Ensure poses of differing dimensions cannot be measured
        with self.assertRaises(AssertionError):
            SEPose.dist(test_pose_1, test_pose_3)
        
        # Ensure non-poses cannot be measured
        with self.assertRaises(AssertionError):
            SEPose.dist(test_pose_1, test_point_1)
        with self.assertRaises(AssertionError):
            SEPose.dist(test_pose_1, test_rot_1)
        
        self.assertTrue(SEPose.dist(test_pose_1, test_pose_2) == np.linalg.norm(test_pose_1.matrix - test_pose_2.matrix, ord="fro"))
        self.assertTrue(SEPose.dist(test_pose_3, test_pose_4) == np.linalg.norm(test_pose_3.matrix - test_pose_4.matrix, ord="fro"))
        
    def test_pose_by_point_and_rotation(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)
        test_rot_2 = Rot2(theta_1, WORLD, ODOM)

        roll_3 = math.pi
        pitch_3 = math.pi / 3
        yaw_3 = math.pi / 4
        test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, ODOM, WORLD)
        test_rot_4 = Rot3(roll_3, pitch_3, yaw_3, WORLD, ODOM)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        # Ensure points and rotations of differing dimensions cannot be used to construct a pose
        with self.assertRaises(AssertionError):
            SE2Pose.by_point_and_rotation(test_point_2, test_rot_1, ODOM, WORLD)
        with self.assertRaises(AssertionError):
            SE2Pose.by_point_and_rotation(test_point_1, test_rot_3, ODOM, WORLD)
        with self.assertRaises(AssertionError):
            SE3Pose.by_point_and_rotation(test_point_1, test_rot_3, ODOM, WORLD)
        with self.assertRaises(AssertionError):
            SE3Pose.by_point_and_rotation(test_point_2, test_rot_1, ODOM, WORLD)

        # Ensure rotations of differing frames than desired cannot be used to construct a pose
        with self.assertRaises(AssertionError):
            SE2Pose.by_point_and_rotation(test_point_1, test_rot_2, ODOM, WORLD)
        with self.assertRaises(AssertionError):
            SE3Pose.by_point_and_rotation(test_point_2, test_rot_4, ODOM, WORLD)

        self.assertTrue(SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, ODOM, WORLD).point == Point2(x_1, y_1, WORLD))
        self.assertTrue(SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, ODOM, WORLD).rot == test_rot_1)
        self.assertTrue(SE3Pose.by_point_and_rotation(test_point_2, test_rot_3, ODOM, WORLD).point == Point3(x_2, y_2, z_2, WORLD))
        self.assertTrue(SE3Pose.by_point_and_rotation(test_point_2, test_rot_3, ODOM, WORLD).rot == test_rot_3)

    def test_pose_by_matrix(self) -> None:
        theta_1 = math.pi / 2
        so2_mat = np.array([[math.cos(theta_1), -1 * math.sin(theta_1)],
                            [math.sin(theta_1), math.cos(theta_1)]])

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4

        yaw_2_mat = np.array([[math.cos(yaw_2), -1 * math.sin(yaw_2), 0],
                              [math.sin(yaw_2), math.cos(yaw_2), 0],
                              [0, 0, 1]])
        pitch_2_mat = np.array([[math.cos(pitch_2), 0, math.sin(pitch_2)],
                                [0, 1, 0],
                                [-1 * math.sin(pitch_2), 0, math.cos(pitch_2)]])
        roll_2_mat = np.array([[1, 0, 0],
                               [0, math.cos(roll_2), -1 * math.sin(roll_2)],
                               [0, math.sin(roll_2), math.cos(roll_2)]])
        
        so3_mat = np.matmul(np.matmul(yaw_2_mat, pitch_2_mat), roll_2_mat)

        x_1 = 42.5123
        y_1 = 23.4530
        trans_arr_1 = np.array([[x_1], [y_1]])

        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        trans_arr_2 = np.array([[x_2], [y_2], [z_2]])

        se2_mat = np.append(so2_mat, trans_arr_1, axis=1)
        se3_mat = np.append(so3_mat, trans_arr_2, axis=1)
        
        se2_mat = np.r_[se2_mat, [np.array([0, 0, 1])]]
        se3_mat = np.r_[se3_mat, [np.array([0, 0, 0, 1])]]

        # Ensure matrices of differing dimensions cannot be used to construct a pose
        with self.assertRaises(AssertionError):
            SE2Pose.by_matrix(se3_mat, ODOM, WORLD)
        with self.assertRaises(AssertionError):
            SE3Pose.by_matrix(se2_mat, ODOM, WORLD)

        self.assertTrue(SE2Pose.by_matrix(se2_mat, ODOM, WORLD).point == Point2(x_1, y_1, WORLD))
        self.assertTrue(SE2Pose.by_matrix(se2_mat, ODOM, WORLD).rot == Rot2(theta_1, ODOM, WORLD))
        self.assertTrue(SE3Pose.by_matrix(se3_mat, ODOM, WORLD).point == Point3(x_2, y_2, z_2, WORLD))
        self.assertTrue(SE3Pose.by_matrix(se3_mat, ODOM, WORLD).rot == Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD))

    # by_exp_map ?

    def test_pose_copy(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        test_rot_2 = Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        test_point_2 = Point3(x_2, y_2, z_2, ODOM)

        se2_pose_1 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, ODOM, WORLD)
        se2_pose_2 = se2_pose_1.copy()

        se3_pose_1 = SE3Pose.by_point_and_rotation(test_point_2, test_rot_2, ODOM, WORLD)
        se3_pose_2 = se3_pose_1.copy()

        self.assertTrue(se2_pose_1 == se2_pose_2)
        self.assertTrue(se3_pose_1 == se3_pose_2)

        # No setter methods to modify poses

    # Deep copy inverse (abstract)
    def test_pose_copy_inv(self) -> None:
        theta_1 = math.pi / 2
        so2_mat = np.array([[math.cos(theta_1), -1 * math.sin(theta_1)],
                            [math.sin(theta_1), math.cos(theta_1)]])

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4

        yaw_2_mat = np.array([[math.cos(yaw_2), -1 * math.sin(yaw_2), 0],
                              [math.sin(yaw_2), math.cos(yaw_2), 0],
                              [0, 0, 1]])
        pitch_2_mat = np.array([[math.cos(pitch_2), 0, math.sin(pitch_2)],
                                [0, 1, 0],
                                [-1 * math.sin(pitch_2), 0, math.cos(pitch_2)]])
        roll_2_mat = np.array([[1, 0, 0],
                               [0, math.cos(roll_2), -1 * math.sin(roll_2)],
                               [0, math.sin(roll_2), math.cos(roll_2)]])
        
        so3_mat = np.matmul(np.matmul(yaw_2_mat, pitch_2_mat), roll_2_mat)

        x_1 = 42.5123
        y_1 = 23.4530
        trans_arr_1 = np.array([[x_1], [y_1]])

        x_2 = 12.1324
        y_2 = 85.4509
        z_2 = 76.3453
        trans_arr_2 = np.array([[x_2], [y_2], [z_2]])

        se2_mat = np.append(so2_mat, trans_arr_1, axis=1)
        se3_mat = np.append(so3_mat, trans_arr_2, axis=1)
        
        se2_mat = np.r_[se2_mat, [np.array([0, 0, 1])]]
        se3_mat = np.r_[se3_mat, [np.array([0, 0, 0, 1])]]

        se2_pose_1 = SE2Pose.by_matrix(se2_mat, ODOM, WORLD)
        se2_pose_2 = se2_pose_1.copyInverse()

        se3_pose_1 = SE3Pose.by_matrix(se3_mat, ODOM, WORLD)
        se3_pose_2 = se3_pose_1.copyInverse()

        self.assertFalse(se2_pose_1 == se2_pose_2)
        self.assertFalse(se3_pose_1 == se3_pose_2)

        self.assertTrue(se2_pose_2 == SE2Pose.by_matrix(np.linalg.inv(se2_mat), WORLD, ODOM))
        self.assertTrue(se3_pose_2 == SE3Pose.by_matrix(np.linalg.inv(se3_mat), WORLD, ODOM))

    # Range and bearing to point
    # Transform to
    

    # Transform local point to base
    def test_pose_local_point_to_base(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        test_rot_2 = Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, WORLD)

        x_2 = 12.0923
        y_2 = 9.576
        test_point_2 = Point2(x_2, y_2, ODOM)

        x_3 = 0.1233
        y_3 = 5.4323
        test_point_3 = Point2(x_3, y_3, WORLD)

        x_4 = 12.1324
        y_4 = 85.4509
        z_4 = 76.3453
        test_point_4 = Point3(x_4, y_4, z_4, WORLD)

        x_5 = 45.2312
        y_5 = 71.2356
        z_5 = 8.8643
        test_point_5 = Point3(x_5, y_5, z_5, ODOM)

        x_6 = 23.1066
        y_6 = 81.0921
        z_6 = 54.0045
        test_point_6 = Point3(x_6, y_6, z_6, WORLD)

        test_pose_1 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, ODOM, WORLD)
        test_pose_2 = SE3Pose.by_point_and_rotation(test_point_4, test_rot_2, ODOM, WORLD)

        # Check that points of differing dimensions cannot be measured
        with self.assertRaises(AssertionError):
            test_pose_1.transform_local_point_to_base(test_point_4)
        with self.assertRaises(AssertionError):
            test_pose_2.transform_local_point_to_base(test_point_1)
        # Check that points in a frame that differs from the pose's local frame cannot be measured
        with self.assertRaises(AssertionError):
            test_pose_1.transform_local_point_to_base(test_point_3)
        with self.assertRaises(AssertionError):
            test_pose_2.transform_local_point_to_base(test_point_6)

        self.assertTrue(test_pose_1.transform_local_point_to_base(test_point_2) == test_rot_1 * test_point_2 + test_point_1)
        self.assertTrue(test_pose_2.transform_local_point_to_base(test_point_5) == test_rot_2 * test_point_5 + test_point_4)

    # Transform base point to local
    def test_pose_base_point_to_local(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        roll_2 = math.pi
        pitch_2 = math.pi / 3
        yaw_2 = math.pi / 4
        test_rot_2 = Rot3(roll_2, pitch_2, yaw_2, ODOM, WORLD)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, WORLD)

        x_2 = 12.0923
        y_2 = 9.576
        test_point_2 = Point2(x_2, y_2, WORLD)

        x_3 = 0.1233
        y_3 = 5.4323
        test_point_3 = Point2(x_3, y_3, ODOM)

        x_4 = 12.1324
        y_4 = 85.4509
        z_4 = 76.3453
        test_point_4 = Point3(x_4, y_4, z_4, WORLD)

        x_5 = 45.2312
        y_5 = 71.2356
        z_5 = 8.8643
        test_point_5 = Point3(x_5, y_5, z_5, WORLD)

        x_6 = 23.1066
        y_6 = 81.0921
        z_6 = 54.0045
        test_point_6 = Point3(x_6, y_6, z_6, ODOM)

        test_pose_1 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, ODOM, WORLD)
        test_pose_2 = SE3Pose.by_point_and_rotation(test_point_4, test_rot_2, ODOM, WORLD)

        # Check that points of differing dimensions cannot be measured
        with self.assertRaises(AssertionError):
            test_pose_1.transform_base_point_to_local(test_point_4)
        with self.assertRaises(AssertionError):
            test_pose_2.transform_base_point_to_local(test_point_1)
        # Check that points in a frame that differs from the pose's base frame cannot be measured
        with self.assertRaises(AssertionError):
            test_pose_1.transform_base_point_to_local(test_point_3)
        with self.assertRaises(AssertionError):
            test_pose_2.transform_base_point_to_local(test_point_6)

        self.assertTrue(test_pose_1.transform_base_point_to_local(test_point_2) == test_rot_1.copyInverse() * test_point_2 + test_rot_1.copyInverse() * test_point_1.copyInverse())
        self.assertTrue(test_pose_2.transform_base_point_to_local(test_point_5) == test_rot_2.copyInverse() * test_point_5 + test_rot_2.copyInverse() * test_point_4.copyInverse())
    
    # Distance to pose
    def test_pose_dist_to_pose(self) -> None:
        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, ODOM, WORLD)

        theta_2 = math.pi / 3
        test_rot_2 = Rot2(theta_2, ODOM, WORLD)

        roll_3 = math.pi
        pitch_3 = math.pi / 3
        yaw_3 = math.pi / 4
        test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, ODOM, WORLD)

        roll_4 = math.pi / 6
        pitch_4 = math.pi / 3
        yaw_4 = math.pi / 2
        test_rot_4 = Rot3(roll_4, pitch_4, yaw_4, ODOM, WORLD)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, ODOM)

        x_2 = 12.0923
        y_2 = 9.576
        test_point_2 = Point2(x_2, y_2, ODOM)

        x_3 = 12.1324
        y_3 = 85.4509
        z_3 = 76.3453
        test_point_3 = Point3(x_3, y_3, z_3, ODOM)

        x_4 = 45.2312
        y_4 = 71.2356
        z_4 = 8.8643
        test_point_4 = Point3(x_4, y_4, z_4, ODOM)

        test_pose_1 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, ODOM, WORLD)
        test_pose_2 = SE2Pose.by_point_and_rotation(test_point_2, test_rot_2, ODOM, WORLD)
        test_pose_3 = SE3Pose.by_point_and_rotation(test_point_3, test_rot_3, ODOM, WORLD)
        test_pose_4 = SE3Pose.by_point_and_rotation(test_point_4, test_rot_4, ODOM, WORLD)

        # Check that poses of differing dimensions cannot be measured
        with self.assertRaises(AssertionError):
            test_pose_1.distance_to_pose(test_pose_3)
        with self.assertRaises(AssertionError):
            test_pose_4.distance_to_pose(test_pose_2)

        self.assertTrue(test_pose_1.distance_to_pose(test_pose_2) == test_point_1.distance(test_point_2))
        self.assertTrue(test_pose_3.distance_to_pose(test_pose_4) == test_point_3.distance(test_point_4))





if __name__ == "__main__":
    unittest.main()