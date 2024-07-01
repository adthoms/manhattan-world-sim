import sys

from matplotlib import pyplot as plt
sys.path.append('../')

import math
import unittest
import numpy as np

from manhattan.agent.agent import (
    Agent,
    Robot,
    Robot2,
    Robot3,
    Beacon,
    Beacon2,
    Beacon3
)
from manhattan.noise_models.loop_closure_model import (
    GaussianLoopClosureModel2, 
    GaussianLoopClosureModel3
)
from manhattan.noise_models.odom_noise_model import (
    GaussianOdomNoiseModel2,
    GaussianOdomNoiseModel3
)
from manhattan.noise_models.range_noise_model import (
    ConstantGaussianRangeNoiseModel
)

from manhattan.geometry.Elements import Point, Point2, Point3, Rot, Rot2, Rot3, SEPose, SE2Pose, SE3Pose
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

const_gauss_range = ConstantGaussianRangeNoiseModel()

gauss_loop_closure_2d = GaussianLoopClosureModel2()
gauss_odom_2d = GaussianOdomNoiseModel2()
test_robot_2d = Robot2("test_robot_2d", test_pose_1, const_gauss_range, gauss_odom_2d, gauss_loop_closure_2d)

gauss_loop_closure_3d = GaussianLoopClosureModel3()
gauss_odom_3d = GaussianOdomNoiseModel3()
test_robot_3d = Robot3("test_robot_3d", test_pose_3, const_gauss_range, gauss_odom_3d, gauss_loop_closure_3d)


class TestRobot(unittest.TestCase):
    
    def test_get_loop_closure_measurement(self) -> None:
        test_loop_closure_2d = test_robot_2d.get_loop_closure_measurement(test_pose_2, "test", gt_measure=False)
        test_loop_closure_gt_2d = test_robot_2d.get_loop_closure_measurement(test_pose_2, "test", gt_measure=True)
        #print(test_loop_closure_2d)
        #print(test_loop_closure_gt_2d)
        
        test_loop_closure_3d = test_robot_3d.get_loop_closure_measurement(test_pose_4, "test", gt_measure=False)
        test_loop_closure_gt_3d = test_robot_3d.get_loop_closure_measurement(test_pose_4, "test", gt_measure=True)
        #print(test_loop_closure_3d)
        #print(test_loop_closure_gt_3d)

    # TODO: Fix test to move with poses of correct frames
    def test_move(self) -> None:
        theta_move = math.pi / 4
        test_rot_move_2d = Rot2(theta_move, FRAME_3, FRAME_1)

        x_move_2d = 56.1230
        y_move_2d = 25.0953
        test_point_move_2d = Point2(x_move_2d, y_move_2d, FRAME_1)

        test_pose_move_2d = SE2Pose.by_point_and_rotation(test_point_move_2d, test_rot_move_2d, FRAME_3, FRAME_1)

        test_move_2d = test_robot_2d.move(test_pose_move_2d, gt_measure=False)
        test_move_gt_2d = test_robot_2d.move(test_pose_move_2d, gt_measure=True)
        #print(test_move_2d)
        #print(test_move_gt_2d)

        # TODO: Create pose of correct frame to move robot in 3D

        test_move_3d = test_robot_3d.move(test_pose_4, gt_measure=False)
        test_move_gt_3d = test_robot_3d.move(test_pose_4, gt_measure=True)
        #print(test_move_3d)
        #print(test_move_gt_3d)

    def test_plot(self) -> None:
        #test_plot_2d = test_robot_2d.plot()
        #plt.savefig("test_robot_2d.png")
        test_plot_3d = test_robot_3d.plot()
        plt.savefig("test_robot_3d.png")


test_beacon_2d = Beacon2("test_beacon_2d", test_point_1, const_gauss_range)
test_beacon_3d = Beacon3("test_beacon_3d", test_point_3, const_gauss_range)


class TestBeacon(unittest.TestCase):
    def test_plot(self) -> None:
        #test_plot_2d = test_beacon_2d.plot()
        #plt.savefig("test_beacon_2d.png")
        test_plot_3d = test_beacon_3d.plot()
        plt.savefig("test_beacon_3d.png")

if __name__ == "__main__":
    unittest.main()