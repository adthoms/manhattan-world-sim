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

        self.assertTrue(test_point_1 ==test_point_2)
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

        self.assertFalse(test_point_1 ==test_point_2)
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
        self.assertTrue(np.allclose(test_point_3.array, np.array([x_3 / scalar_3, y_3/ scalar_3, z_3 / scalar_3]), _TRANSLATION_TOLERANCE))
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
    # def test_rot_dist_static(self) -> None:
    #   theta_1 = math.pi / 2
    #    test_rot_1 = Rot2(theta_1, ODOM, WORLD)

    #    theta_2 = math.pi / 3
    #    test_rot_2 = Rot2(theta_2, ODOM, WORLD)

    #    self.assertTrue(Rot.dist(test_rot_1, test_rot_1) < _ROTATION_TOLERANCE)
        # self.assertTrue(Rot.dist(test_rot_1, test_rot_2))
    #    print(Rot.dist(test_rot_1, test_rot_1.copyInverse()))
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