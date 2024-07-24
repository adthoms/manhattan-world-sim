import sys
sys.path.append('../')

import math
import unittest
import numpy as np

from manhattan.geometry.Elements import Point, Point2, Point3, Rot, Rot2, Rot3, SEPose, SE2Pose, SE3Pose
from manhattan.environment.environment import ManhattanWorld
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