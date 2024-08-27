import sys
sys.path.append('../')

import math
import unittest
import numpy as np

from manhattan.agent.agent import Robot3
from manhattan.geometry.Elements import DIM, Point, Point2, Point3, Rot, Rot2, Rot3, SEPose, SE2Pose, SE3Pose
from manhattan.environment.environment import ManhattanWorld
from manhattan.utils.sample_utils import choice
from manhattan.noise_models.range_noise_model import (
    RangeNoiseModel,
    ConstantGaussianRangeNoiseModel as ConstGaussRangeSensor,
    VaryingMeanGaussianRangeNoiseModel as VaryGaussRangeSensor,
)
from manhattan.noise_models.odom_noise_model import (
    OdomNoiseModel,
    GaussianOdomNoiseModel2 as GaussOdomSensor2,
    GaussianOdomNoiseModel3 as GaussOdomSensor3
)
from manhattan.noise_models.loop_closure_model import (
    LoopClosureModel,
    GaussianLoopClosureModel2 as GaussLoopClosureSensor2,
    GaussianLoopClosureModel3 as GaussLoopClosureSensor3
)
from manhattan.utils.geo_utils import bearing_is_behind_robot
FRAME_1 = "odom"
FRAME_2 = "world"
FRAME_3 = "tool"

class TestSimInterface(unittest.TestCase):
    def test_add_robot(self):
        # Implemented in simulator.py, must create unit test
        pass

    def test_add_beacon(self):
        # Implemented in simulator.py, must create unit test
        pass

class TestMovement(unittest.TestCase):
    pass

class TestMeasurements(unittest.TestCase):
    def test_store_odometry_measurements(self):
        # Implemented in simulator.py, must create unit test
        pass

    def test_update_range_measurements(self):
        # Implemented in simulator.py, must create unit test
        pass

    def test_update_loop_closures(self):
        # Implemented in simulator.py, must create unit test
        pass
    
    # Range measurements have no sense of direction; should be same for both 2D and 3D
    
    def test_get_incorrect_robot_to_robot_range_association(self):
        # Implemented in simulator.py, must create unit test
        pass

    def test_get_incorrect_robot_to_beacon_range_association(self):
        # Implemented in simulator.py, must create unit test
        pass

    def test_add_robot_to_robot_range_measurement(self):
        # Implemented in simulator.py, must create unit test
        pass

    def test_add_robot_to_beacon_range_measurement(self):
        # Implemented in simulator.py, must create unit test
        pass

class TestVisualization(unittest.TestCase):
    def test_print_simulator_state(self):
        pass

    def test_print_robot_states(self):
        pass

    def test_print_beacon_states(self):
        pass

    def test_plot_grid(self):
        pass

    def test_plot_beacons(self):
        pass

    def test_plot_robot_states(self):
        pass

    def test_show_plot(self):
        pass

    def test_close_plot(self):
        pass

class TestSimulator(unittest.TestCase):
    def test_neighbors(self):
        man_env = ManhattanWorld(
            dim=DIM.THREE,
            grid_vertices_shape=(4, 4, 4),
            z_steps_to_intersection=2,
            y_steps_to_intersection=2,
            x_steps_to_intersection=2,
            cell_scale=1.0,
        )

        print(man_env.get_neighboring_vertices((2, 2, 2)))

    def test_bearing(self):
        start_pose = SE3Pose.by_point_and_rotation(Point3(0.0, 0.0, 0.0, "world"), Rot3(0.0, 0.0, 0.0, "", "world"), "", "world")
        test_pt = Point3(0.0, 0.0, 1.0, "world")
        yaw, pitch = start_pose.range_and_bearing_to_point(test_pt)[1]
        print((0.0, pitch, yaw))
    
    def test_pose_transform(self):
        start_pose = SE3Pose.by_point_and_rotation(Point3(0.0, 3.0, 0.0, "world"), Rot3(-math.pi / 2, math.pi / 2, 0.0, "A5", "world"), "A5", "world")
        print(start_pose.matrix)
        transform_pose = SE3Pose.by_point_and_rotation(Point3(0.0, 0.0, 1.0, "A5"), Rot3(0.0, math.pi / 2, 0.0, "A6", "A5"), "A6", "A5")
        print(transform_pose.matrix)
        new_pose = start_pose * transform_pose
        print(new_pose)
        print(new_pose.matrix)

    def test_sim_measurements(self):
        def check_rot_is_manhattan(pose: SE3Pose, tol: float = 1e-2):
            for angle in pose.rot.angles:
                self.assertTrue(math.isclose(abs(angle), 0.0, abs_tol=tol) or 
                                math.isclose(abs(angle), math.pi/2, abs_tol=tol) or
                                math.isclose(abs(angle), math.pi, abs_tol=tol))
                
        def check_correct_bearing_and_movement(prev_pose: SE3Pose, new_pose: SE3Pose, heading_tol: float = 1e-2):
            roll, pitch, yaw = new_pose.rot.angles

            diff_pt = new_pose.point - prev_pose.point
            diff_x, diff_y, diff_z = diff_pt.x, diff_pt.y, diff_pt.z

            if abs(pitch - (np.pi / 2.0)) < heading_tol: # 90 degrees on y-axis; heading "down"
                self.assertAlmostEqual(diff_x, 0.0)
                self.assertAlmostEqual(diff_y, 0.0)
                self.assertAlmostEqual(diff_z, -1.0)
            elif abs(pitch + (np.pi / 2.0)) < heading_tol: # 270 degrees on y-axis; heading "up"
                self.assertAlmostEqual(diff_x, 0.0)
                self.assertAlmostEqual(diff_y, 0.0)
                self.assertAlmostEqual(diff_z, 1.0)
            elif abs(yaw - (np.pi / 2.0)) < heading_tol: # 90 degrees on z-axis; heading north
                self.assertAlmostEqual(diff_x, 0.0)
                self.assertAlmostEqual(diff_y, 1.0)
                self.assertAlmostEqual(diff_z, 0.0)
            elif ( # 180 degrees on z-axis; heading west
                abs(yaw + np.pi) < heading_tol
                or abs(yaw - np.pi) < heading_tol
                or abs(pitch + np.pi) < heading_tol
                or abs(pitch - np.pi) < heading_tol
            ):
                self.assertAlmostEqual(diff_x, -1.0)
                self.assertAlmostEqual(diff_y, 0.0)
                self.assertAlmostEqual(diff_z, 0.0)
            elif abs(yaw + (np.pi / 2.0)) < heading_tol: # 270 degrees on z-axis; heading south
                self.assertAlmostEqual(diff_x, 0.0)
                self.assertAlmostEqual(diff_y, -1.0)
                self.assertAlmostEqual(diff_z, 0.0)
            elif (abs(yaw) < heading_tol) or (abs(pitch) < heading_tol): # 0 degrees on z-axis; heading east
                self.assertAlmostEqual(diff_x, 1.0)
                self.assertAlmostEqual(diff_y, 0.0)
                self.assertAlmostEqual(diff_z, 0.0)
            else:
                raise AssertionError(f"Unhandled heading: {self.heading}")
                
        start_pose = SE3Pose.by_point_and_rotation(Point3(0.0, 0.0, 0.0, "world"), Rot3(0.0, 0.0, 0.0, "", "world"), "", "world")
        range_model = ConstGaussRangeSensor(
            mean=0.0, stddev=0.1
        )
        odometry_model = GaussOdomSensor3(
            mean=np.zeros(6),
            covariance=np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        )
        loop_closure_model = GaussLoopClosureSensor3(
            mean=np.zeros(6),
            covariance=np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        )
        robot = Robot3("A", start_pose, range_model, odometry_model, loop_closure_model)

        seed_cnt = 256
        man_env = ManhattanWorld(
            dim=DIM.THREE,
            grid_vertices_shape=(30, 30, 30),
            z_steps_to_intersection=2,
            y_steps_to_intersection=2,
            x_steps_to_intersection=2,
            cell_scale=1.0,
        )
        num_iters = 2000
        tol = 1e-2

        for _ in range(num_iters):
            # Doesn't account for when robot has no more moves (must include vertices not behind robot)
            possible_moves = man_env.get_neighboring_robot_vertices_not_behind_robot(robot)
            for pt, rot in possible_moves:
                # print(str(pt) + "   ---->   " + str(rot))
                self.assertFalse(bearing_is_behind_robot(rot[1], rot[2], tol))
            next_trans = choice(possible_moves)

            move_pt = next_trans[0]
            roll, pitch, yaw = next_trans[1]
            move_frame_name = f"{robot.name}{robot.timestep+1}"
            move_pt_local = robot.pose.transform_base_point_to_local(move_pt)

            # print("Chosen move: " + str(move_pt))
            # print("Chosen move local: " + str(move_pt_local))

            move_transform = SE3Pose(
                move_pt_local.x,
                move_pt_local.y,
                move_pt_local.z,
                roll,
                pitch,
                yaw,
                local_frame=move_frame_name,
                base_frame=robot.pose.local_frame,
            )

            prev_pose = robot.pose

            # move the robot and store the measurement and new pose
            robot.move(
                move_transform, True
            )

            new_pose = robot.pose

            # check that the robot moved correctly
            check_rot_is_manhattan(robot.pose)

            # check that robot moved by exactly one vertex
            self.assertAlmostEqual(abs(move_pt_local.x) + abs(move_pt_local.y) + abs(move_pt_local.z), 1.0)

            # check that robot is facing in only one direction; pitch and yaw cannot both be non-zero
            is_pitch_nonzero = not math.isclose(abs(pitch), 0.0, abs_tol=tol)
            is_yaw_nonzero = not math.isclose(abs(yaw), 0.0, abs_tol=tol)
            self.assertTrue(not (is_pitch_nonzero and is_yaw_nonzero))

            # print("Previous pose: " + str(prev_pose))
            # print("Next pose: " + str(new_pose))
            # print(new_pose.rot.matrix)

            # check that bearing in world frame corresponds to correct one-vertex movement
            check_correct_bearing_and_movement(prev_pose, new_pose, tol)

            # print("New robot pose: " + str(robot.pose))
            # print()


if __name__ == "__main__":
    unittest.main()