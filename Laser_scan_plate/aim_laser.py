import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory
import numpy as np
import math
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import threading

class Point_Aimer_Ur7e(Node):
    
    ### Variable ###
    # Distance du laser avec la surface de la plaque
    dis_z = 0.3

    # Points de la plaque (4 coins) 

    point_0_0 = [(-0.6599719682230593, -0.0915199219877611, 0.6387315748197623)]
    point_1_0 = [(-0.6644354257049206, -0.09186204541361304, 0.400642836541432)]
    point_1_1 = [(-0.8003312881568884, 0.1777877687219231, 0.403701239780513)]
    point_0_1 = [(-0.7978981766977311, 0.1782856354359193, 0.6433591317769192)]
    

    def __init__(self):
        super().__init__('aimer_ur7e_preview')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path') # Service de MoveIt pour calculer un chemin cartésien
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory') # Action de MoveIt pour exécuter une trajectoire
        self.display_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10) # Publisher pour visualiser la trajectoire dans RViz
        
        # On calcule la géométrie de la plaque une seule fois au démarrage
        self.setup_plate_geometry()
        return 
    
    def setup_plate_geometry(self):
        """ Calcule et stocke les vecteurs et l'origine de la plaque définitivement. """

        ### Les points sont noté avec point_ex_ey avec 1 le bord de plaque

        point_0_0 = self.point_0_0
        point_1_0 = self.point_1_0
        point_1_1 = self.point_1_1
        point_0_1 = self.point_0_1

        point_0_0_m = np.mean(point_0_0, axis=0)
        point_1_0_m = np.mean(point_1_0, axis=0)
        point_1_1_m = np.mean(point_1_1, axis=0)
        point_0_1_m = np.mean(point_0_1, axis=0)
        
        # Normal de la surface
        P_diag_00_11 = point_0_0_m - point_1_1_m

        P_diag_01_10 = point_0_1_m - point_1_0_m

        P_Z = np.cross(P_diag_00_11, P_diag_01_10)
        self.P_Z = P_Z / np.linalg.norm(P_Z)

        self.get_logger().info(f"Vecteur P_Z calculé : {self.P_Z}")

        # Coin x = 0 , y = 0   
        dist_point_0_0 = np.dot(point_0_0_m , P_Z)
        dist_point_1_0 = np.dot(point_1_0_m , P_Z)
        dist_point_1_1 = np.dot(point_1_1_m , P_Z)
        dist_point_0_1 = np.dot(point_0_1_m , P_Z)

        dist_surface = [dist_point_0_0, dist_point_1_0, dist_point_1_1, dist_point_0_1]

        p_00 = point_0_0_m # - np.mean(dist_surface) * P_Z
        p_01 = point_0_1_m # - np.mean(dist_surface) * P_Z
        p_10 = point_1_0_m # - np.mean(dist_surface) * P_Z
        p_11 = point_1_1_m # - np.mean(dist_surface) * P_Z

        self.p_00 = p_00
        self.p_01 = p_01
        self.p_10 = p_10
        self.p_11 = p_11

        P_X = p_10 - p_00
        self.P_X = P_X / np.linalg.norm(P_X)

        P_Y = np.cross(self.P_X, P_Z)
        self.P_Y = P_Y / np.linalg.norm(P_Y)

        self.corner = p_00




    def rot2quat(self, R):
        m00, m01, m02 = R[0,0], R[0,1], R[0,2]
        m10, m11, m12 = R[1,0], R[1,1], R[1,2]
        m20, m21, m22 = R[2,0], R[2,1], R[2,2]
        tr = m00 + m11 + m22
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            return [(m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S, 0.25 * S]
        elif (m00 > m11) and (m00 > m22):
            S = math.sqrt(1.0 + m00 - m11 - m22) * 2
            return [0.25 * S, (m01 + m10) / S, (m02 + m20) / S, (m21 - m12) / S]
        elif m11 > m22:
            S = math.sqrt(1.0 + m11 - m00 - m22) * 2
            return [(m01 + m10) / S, 0.25 * S, (m12 + m21) / S, (m02 - m20) / S]
        else:
            S = math.sqrt(1.0 + m22 - m00 - m11) * 2
            return [(m02 + m20) / S, (m12 + m21) / S, 0.25 * S, (m10 - m01) / S]

    def calculate_tool0_pose(self, tcp_pos, tcp_R):
        R_local = np.eye(3) 
        t_local = np.array([0.29119,-0.00164,0.0625]) 
        T_local = np.eye(4)
        T_local[:3, :3] = R_local
        T_local[:3, 3] = t_local
        
        T_target = np.eye(4)
        T_target[:3, :3] = tcp_R
        T_target[:3, 3] = tcp_pos
        
        T_tool0 = T_target @ np.linalg.inv(T_local)
        return T_tool0[:3, 3], T_tool0[:3, :3]


    def aim_UR7e(self, X, Y):
        """ Déplace le robot sur un rapport X = [0,1] et Y = [0,1] de la plaque."""

        if X < -0.2 or X > 1.2 or Y < -0.2 or Y > 1.2:
            print("Hors limites de la plaque.")
            return False

        tcp_x = self.P_Z  
        tcp_y = self.P_Y
        tcp_z = np.cross(tcp_x, tcp_y) 
        R_cible = np.column_stack((tcp_x, tcp_y, tcp_z))
        
        center_pt = self.p_00*(1-X)*(1-Y) + self.p_10*(1-X)*Y + self.p_11*X*Y + self.p_01*X*(1-Y)

        hover_pt = center_pt - (self.dis_z * self.P_Z)

        pos_tool0, R_tool0 = self.calculate_tool0_pose(hover_pt, R_cible)
        q_tool0 = self.rot2quat(R_tool0)

        target_pose = Pose()
        target_pose.position.x = pos_tool0[0]
        target_pose.position.y = pos_tool0[1]
        target_pose.position.z = pos_tool0[2]
        target_pose.orientation.x = q_tool0[0]
        target_pose.orientation.y = q_tool0[1]
        target_pose.orientation.z = q_tool0[2]
        target_pose.orientation.w = q_tool0[3]

        while not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Attente de MoveIt...')

        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.group_name = 'ur_manipulator'
        req.link_name = 'tool0'  
        req.waypoints = [target_pose] 
        req.max_step = 0.01 
        req.jump_threshold = 0.0

        req.max_velocity_scaling_factor = 0.1 # Vitesse de déplacement
        req.max_acceleration_scaling_factor = 0.1 # Accélération

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response.fraction < 0.99:
            self.get_logger().error(f"Impossible de calculer le chemin cartésien.")
            return
            
        display_msg = DisplayTrajectory()
        display_msg.trajectory.append(response.solution)
        self.display_pub.publish(display_msg)

        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = response.solution
        self.execute_client.wait_for_server()
        
        send_goal_future = self.execute_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Mouvement refusé par le contrôleur.")
            return False

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        
        return True


def main(args=None):
    rclpy.init(args=args)

    ## Initialisation 
    node = Point_Aimer_Ur7e()

    ## Viser de la plaque avec des rapports X et Y
    # aim_UR7e(X, Y) avec X et Y entre 0 et 1
    # X = 0 correspond au bord de la plaque avec point_0_0 et point_0_1
    # Y = 0 correspond au bord de la plaque avec point_0_0 et point_1_0

    node.aim_UR7e(0.5 , 0.5)

    ## Supression des connexions et fermeture du node
    node.destroy_node()
    rclpy.shutdown()
    return

if __name__ == '__main__':
    main()