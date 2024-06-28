import sys
sys.path.append('../')

import math
import unittest
import numpy as np
from manhattan.noise_models.loop_closure_model import (
    LoopClosureModel,
    GaussianLoop2DClosureModel as GaussLC2D,
    GaussianLoop3DClosureModel as GaussLC3D
)
from manhattan.noise_models.odom_noise_model import (
    OdomNoiseModel,
    GaussianOdom2DNoiseModel as GaussOdom2D,
    GaussianOdom3DNoiseModel as GaussOdom3D
)
from manhattan.noise_models.range_noise_model import (
    RangeNoiseModel,
    ConstantGaussianRangeNoiseModel,
    VaryingMeanGaussianRangeNoiseModel
)
from manhattan.geometry.Elements import Point, Point2, Point3, Rot, Rot2, Rot3, SEPose, SE2Pose, SE3Pose

FRAME_1 = "odom"
FRAME_2 = "world"
FRAME_3 = "tool"

class TestLoopClosureModel(unittest.TestCase):
    def test_relative_pose_measurement(self) -> None:
        """
        Test the relative pose measurement function inherited from LoopClosureModel
        by GaussianLoop2DClosureModel and GaussianLoop3DClosureModel
        """

        # Test the 2D model
        lc2d = GaussLC2D()
        #print(lc2d.covariance)
        #print(lc2d.mean)

        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, FRAME_1, FRAME_2)

        theta_2 = math.pi / 3
        test_rot_2 = Rot2(theta_2, FRAME_1, FRAME_2)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, FRAME_2)

        x_2 = 12.0923
        y_2 = 9.576
        test_point_2 = Point2(x_2, y_2, FRAME_2)

        test_pose_1 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, FRAME_1, FRAME_2)
        test_pose_2 = SE2Pose.by_point_and_rotation(test_point_2, test_rot_2, FRAME_1, FRAME_2)

        noisylc2d = lc2d.get_relative_pose_measurement(test_pose_1, test_pose_2, "test", 0)

        # Test the 3D model
        lc3d = GaussLC3D()
        #print(lc3d.covariance)
        #print(lc3d.mean)

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
        yaw_3 = math.pi / 4
        test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, FRAME_1, FRAME_2)

        roll_4 = math.pi / 6
        pitch_4 = math.pi / 3
        yaw_4 = math.pi / 2
        test_rot_4 = Rot3(roll_4, pitch_4, yaw_4, FRAME_1, FRAME_2)

        test_pose_3 = SE3Pose.by_point_and_rotation(test_point_3, test_rot_3, FRAME_1, FRAME_2)
        test_pose_4 = SE3Pose.by_point_and_rotation(test_point_4, test_rot_4, FRAME_1, FRAME_2)

        noisylc3d = lc3d.get_relative_pose_measurement(test_pose_3, test_pose_4, "test", 0)

        print(noisylc2d)
        print(noisylc3d)

class TestOdomModel(unittest.TestCase):
    def test_get_odometry_measurement(self) -> None:
        """
        Test get_odometry_measurement inherited from OdomNoiseModel
        by GaussianOdom2DNoiseModel and GaussianOdom3DNoiseModel
        """

        # Test the 2D model
        odom2d = GaussOdom2D(covariance=np.eye(3) / 10)
        #print(lc2d.covariance)
        #print(lc2d.mean)

        theta_1 = math.pi / 2
        test_rot_1 = Rot2(theta_1, FRAME_1, FRAME_2)

        x_1 = 42.5123
        y_1 = 23.4530
        test_point_1 = Point2(x_1, y_1, FRAME_2)

        test_pose_1 = SE2Pose.by_point_and_rotation(test_point_1, test_rot_1, FRAME_1, FRAME_2)

        noisy_odom_2d_measurement = odom2d.get_odometry_measurement(test_pose_1)

        # Test the 3D model
        odom3d = GaussOdom3D(covariance=np.eye(6) / 10)

        x_3 = 12.1324
        y_3 = 85.4509
        z_3 = 76.3453
        test_point_3 = Point3(x_3, y_3, z_3, FRAME_2)

        roll_3 = math.pi
        pitch_3 = math.pi / 3
        yaw_3 = math.pi / 4
        test_rot_3 = Rot3(roll_3, pitch_3, yaw_3, FRAME_1, FRAME_2)

        test_pose_3 = SE3Pose.by_point_and_rotation(test_point_3, test_rot_3, FRAME_1, FRAME_2)

        noisy_odom_3d_measurement = odom3d.get_odometry_measurement(test_pose_3)
        print(noisy_odom_2d_measurement)
        print()
        print(noisy_odom_3d_measurement)

class TestRangeModel(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()