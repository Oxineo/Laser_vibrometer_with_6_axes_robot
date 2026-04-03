#!/bin/bash

gnome-terminal --title="Ur Driver" -- bash -c "ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur7e robot_ip:=192.168.186.141 launch_rviz:=false" & { sleep 10 ; gnome-terminal --title="Ur MoveIt" -- bash -c "ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur7e launch_rviz:=true" ; sleep 5 ; }

WORK_DIR="Robot_control_system/robot_model"

gnome-terminal --title="Laser 3D Model" --working-directory=$WORK_DIR -- bash -c "/bin/python3 $WORK_DIR/Import_3D_model_with_payload.py" 
