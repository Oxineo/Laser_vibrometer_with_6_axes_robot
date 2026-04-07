#%% START THE ROBOT
import aim_laser as las
import rclpy
import numpy as np
import time
import os
import json
from datetime import datetime


rclpy.init(args=None)

node = las.Point_Aimer_Ur7e()

#%% MODIFY THE PLATE CORNER POINTS

node.point_0_0 = [(-0.3883628114459706, 0.48226363134787964, 0.03114266515166833)]
node.point_1_0 = [(-0.6424912989280012, 0.5788997615295917, 0.03257093693609156)]
node.point_1_1 = [(-0.7076709435247521, 0.4223741258515613, 0.03266498893429043)]
node.point_0_1 = [(-0.4552818906331031, 0.3238727868571719, 0.0326745860350256)]

node.setup_plate_geometry()

#%% MOVE THE ROBOT MANUALLY
node.aim_UR7e(0,1)

#%% % STOP AND FREE THE ROBOT

node.destroy_node()
rclpy.shutdown()