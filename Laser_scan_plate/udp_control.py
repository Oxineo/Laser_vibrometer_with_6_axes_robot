#%%
import aim_laser as las
import rclpy
import numpy as np
import time
import os
import json
from datetime import datetime


rclpy.init(args=None)

node = las.Point_Aimer_Ur7e()

#%%

node.point_0_0 = [(1.0,0.0,1.0)]
node.setup_plate_geometry()

#%%
node.aim_UR7e(0.5, 0.8)

#%%

node.destroy_node()
rclpy.shutdown()