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
    def test_sim_measurements(self):
        def check_rot_is_manhattan(pose: SE3Pose, tol: float = 1e-2):
            is_manhattan = True
            for angle in pose.rot.angles:
                self.assertTrue(math.isclose(abs(angle), 0.0, abs_tol=tol) or 
                                math.isclose(abs(angle), math.pi/2, abs_tol=tol) or
                                math.isclose(abs(angle), math.pi, abs_tol=tol))
                # if not (math.isclose(abs(angle), 0.0, abs_tol=tol) or math.isclose(abs(angle), math.pi/2, abs_tol=tol) or math.isclose(abs(angle), math.pi, abs_tol=tol)):
                #    is_manhattan = False
            # return is_manhattan
                
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
        man_env = ManhattanWorld(
            dim=DIM.THREE,
            grid_vertices_shape=(4, 4, 4),
            z_steps_to_intersection=2,
            y_steps_to_intersection=2,
            x_steps_to_intersection=2,
            cell_scale=1.0,
        )
        num_iters = 50
        tol = 1e-2

        for _ in range(num_iters):
            # Doesn't account for when robot has no more moves (must include vertices not behind robot)
            possible_moves = man_env.get_neighboring_robot_vertices_not_behind_robot(robot)
            for pt, rot in possible_moves:
                print(str(pt) + "   ---->   " + str(rot))
                self.assertFalse(bearing_is_behind_robot(rot[1], rot[2], tol))
            next_trans = choice(possible_moves)

            move_pt = next_trans[0]
            roll, pitch, yaw = next_trans[1]
            move_frame_name = f"{robot.name}{robot.timestep+1}"
            move_pt_local = robot.pose.transform_base_point_to_local(move_pt)

            print("Chosen move: " + str(move_pt))
            print("Chosen move local: " + str(move_pt_local))

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

            prev_x = robot.pose.point.x
            prev_y = robot.pose.point.y
            prev_z = robot.pose.point.z

            # move the robot and store the measurement and new pose
            robot.move(
                move_transform, True
            )

            cur_x = robot.pose.point.x
            cur_y = robot.pose.point.y
            cur_z = robot.pose.point.z

            # check that the robot moved correctly
            check_rot_is_manhattan(robot.pose)
            # if (not check_rot_is_manhattan(robot.pose)):
            #    print("Pose is not manhattan")

            # self.assertAlmostEqual(cur_x, prev_x + move_pt_local.x)
            # self.assertAlmostEqual(cur_y, prev_y + move_pt_local.y)
            # self.assertAlmostEqual(cur_z, prev_z + move_pt_local.z)


if __name__ == "__main__":
    unittest.main()