import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import threading
import math
import pyperclip

class LaserTCPNode(Node):
    def __init__(self):
        super().__init__('laser_tcp_node')
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_transforms()

    def get_quaternion(self, roll, pitch, yaw):
        """Convertit les degrés (en radians) en quaternion"""
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    def publish_transforms(self):
        # ==========================================
        # 1. ORIENTATION (La base du laser)
        # ==========================================
        t_base = TransformStamped()
        t_base.header.stamp = self.get_clock().now().to_msg()
        t_base.header.frame_id = 'tool0'
        t_base.child_frame_id = 'laser_base'
        
        t_base.transform.translation.x = 0.191
        t_base.transform.translation.y = 0.0
        t_base.transform.translation.z = 0.048
        
        q_base = self.get_quaternion(0.0, 0.0, 0.0)
        t_base.transform.rotation.x = q_base[0]
        t_base.transform.rotation.y = q_base[1]
        t_base.transform.rotation.z = q_base[2]
        t_base.transform.rotation.w = q_base[3]

        # ==========================================
        # 2. LA POINTE
        # ==========================================
        t_tip = TransformStamped()
        t_tip.header.stamp = self.get_clock().now().to_msg()
        t_tip.header.frame_id = 'laser_base'
        t_tip.child_frame_id = 'laser_tcp'
        
        # Attention ici : ton commentaire dit 40 cm (0.4m) mais tes valeurs sont d'environ 10 cm
        t_tip.transform.translation.x = 0.10019
        t_tip.transform.translation.y = -0.000164
        t_tip.transform.translation.z = 0.0145 
        
        t_tip.transform.rotation.x = 0.0
        t_tip.transform.rotation.y = 0.0
        t_tip.transform.rotation.z = 0.0
        t_tip.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform([t_base, t_tip])

class PointRecorder(Node):
    def __init__(self):
        super().__init__('point_recorder')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.points = []

    def record_point(self):
        try:
            # lookup_transform a parfois besoin d'un court délai pour que l'arbre TF se remplisse
            t = self.tf_buffer.lookup_transform('base_link', 'laser_tcp', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            z = t.transform.translation.z
            
            self.points.append((x, y, z))
            print(f"📍 Point {len(self.points)} enregistré : X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
            
        except TransformException as ex:
            print(f"❌ Impossible de lire la position. Erreur : {ex}")


def main():
    rclpy.init()
    
    # Création des deux nœuds
    node_recorder = PointRecorder()
    node_laser = LaserTCPNode()
    
    # Utilisation d'un Executor pour gérer plusieurs nœuds
    executor = SingleThreadedExecutor()
    executor.add_node(node_recorder)
    executor.add_node(node_laser)
    
    # Fait tourner l'Executor (ROS 2) en arrière-plan
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    print("\n--- 🛠️ ENREGISTREUR DE POINTS UR7e (MODE MANUEL) ---")
    print("1. Déplace le robot manuellement jusqu'au coin de ta plaque")
    print("2. Appuie sur [Entrée] pour sauvegarder la position")
    print("3. Tape 'q' puis [Entrée] pour quitter et afficher le résumé\n")
    
    try:
        # La boucle d'input utilisateur est maintenant libre de s'exécuter
        while rclpy.ok():
            user_input = input("Appuie sur [Entrée] pour enregistrer... (ou 'q' pour quitter) : ")
            if user_input.lower() == 'q':
                break
            
            node_recorder.record_point()
            
    except KeyboardInterrupt:
        pass
        
    print("\n--- 📋 RÉCAPITULATIF DES POINTS ---")
    if not node_recorder.points:
        print("Aucun point enregistré.")
    else:
        for i, p in enumerate(node_recorder.points):
            print(f"Point {i+1} : X={p[0]:.4f}, Y={p[1]:.4f}, Z={p[2]:.4f}")
        print("\n------------------------------------------\n")
        print(f"Liste de point copié dans le presse-papier : \n{node_recorder.points}")
        

        # Envoyer du texte au presse-papier
        pyperclip.copy(str(node_recorder.points))

        
    # Nettoyage propre
    executor.shutdown()
    node_recorder.destroy_node()
    node_laser.destroy_node()
    rclpy.shutdown()
    spin_thread.join()

if __name__ == '__main__':
    main()