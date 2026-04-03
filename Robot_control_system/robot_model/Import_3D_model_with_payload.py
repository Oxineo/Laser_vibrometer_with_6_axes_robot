import rclpy
from rclpy.node import Node
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point, Pose, Vector3
from ur_msgs.srv import SetPayload 
import time
import os
import math

class LaserMeshAttacher(Node):
    def __init__(self):
        super().__init__('laser_mesh_attacher')
        
        # 1. Publisher pour l'objet 3D
        self.publisher = self.create_publisher(
            AttachedCollisionObject, 
            '/attached_collision_object', 
            10
        )
        
        # 2. Client pour le service de poids (On utilise le bon canal actif !)
        self.payload_client = self.create_client(
            SetPayload, 
            '/io_and_status_controller/set_payload'
        )

    def set_robot_payload(self):
        """ Envoie la masse et le centre de gravité au robot UR """
        self.get_logger().info('Connexion au robot pour le réglage du poids...')
        
        while not self.payload_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Attente du service io_and_status_controller...')
        
        req = SetPayload.Request()
        req.mass = 4.830  # Ton outil pèse 4.83 kg
        req.center_of_gravity = Vector3(x=0.021, y=0.0, z=0.05) 
        
        # Envoi de la requête et attente de la confirmation
        future = self.payload_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f"Poids du robot configuré avec succès : {req.mass} kg !")
        else:
            self.get_logger().error("Le robot a refusé la configuration du poids.")

    def load_ascii_stl(self, filepath):
        mesh_msg = Mesh()
        vertices = []
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4 and parts[0] == 'vertex':
                    p = Point()
                    p.x = float(parts[1])
                    p.y = float(parts[2])
                    p.z = float(parts[3])
                    vertices.append(p)
        
        for i in range(0, len(vertices), 3):
            mesh_msg.vertices.extend([vertices[i], vertices[i+1], vertices[i+2]])
            tri = MeshTriangle()
            tri.vertex_indices = [i, i+1, i+2]
            mesh_msg.triangles.append(tri)
            
        return mesh_msg

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    def attach_laser(self):
        file_path = "Laser_Platine.stl" 
        
        if not os.path.exists(file_path):
            self.get_logger().error(f"Impossible de trouver {file_path}. Lance le script depuis le même dossier !")
            return

        self.get_logger().info(f"Lecture du fichier 3D {file_path} en cours...")
        
        attached_laser = AttachedCollisionObject()
        attached_laser.link_name = "tool0" 
        
        # On dit à MoveIt d'ignorer les collisions entre le laser et la fin du bras
        attached_laser.touch_links = ['tool0', 'wrist_3_link'] 
        
        obj = CollisionObject()
        obj.header.frame_id = "tool0"
        obj.id = "laser_3d_complet"
        
        mesh = self.load_ascii_stl(file_path)
        
        # Positionnement et Rotation
        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = 0.0 
        
        angle_x = -90.0  # Rotation autour de X
        angle_y = 0.0    # Rotation autour de Y
        angle_z = 45.0   # Rotation autour de Z
        
        q = self.get_quaternion_from_euler(
            math.radians(angle_x), 
            math.radians(angle_y), 
            math.radians(angle_z)
        )
        
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        
        obj.meshes.append(mesh)
        obj.mesh_poses.append(pose)
        obj.operation = CollisionObject.ADD
        
        attached_laser.object = obj
        
        # Pause courte pour laisser ROS s'abonner
        time.sleep(1.0)
        self.publisher.publish(attached_laser)
        self.get_logger().info("Laser 3D attaché avec succès dans MoveIt")

def main(args=None):
    rclpy.init(args=args)
    node = LaserMeshAttacher()
    
    # 1. Envoi du poids au vrai robot
    node.set_robot_payload()
    
    # 2. Envoi du modèle 3D à MoveIt
    node.attach_laser()
    
# On garde le script allumé indéfiniment pour que MoveIt reçoive bien le modèle !
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()