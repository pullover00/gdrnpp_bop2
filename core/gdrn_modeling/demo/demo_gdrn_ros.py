# inference with detector, gdrn, and refiner
import os.path as osp
import sys
import torch
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

from predictor_gdrn import GdrnPredictor
import os

import time

import cv2
import numpy as np

import rospy
from std_msgs.msg import Header
from object_detector_msgs.msg import BoundingBox, Detection, Detections, PoseWithConfidence
from object_detector_msgs.srv import detectron2_service_server, estimate_poses, estimate_posesResponse
from geometry_msgs.msg import Pose, Point, Quaternion
from cv_bridge import CvBridge, CvBridgeError
import tf

class GDRN_ROS:
    def __init__(self):
            intrinsics = np.asarray(rospy.get_param('/pose_estimator/intrinsics'))
            self.frame_id = rospy.get_param('/pose_estimator/color_frame_id')
            self.gdrn_predictor = GdrnPredictor(
                config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/ycbv/ycbv_inference.py"),
                ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/ycbv/gdrnpp_ycbv_weights.pth"),
                #camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/camera_uw.json"),
                camera_intrinsics=intrinsics,
                path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models")
            )

            rospy.init_node("gdrn_estimation")
            s = rospy.Service("estimate_poses", estimate_poses, self.estimate_pose)
            print("Pose Estimation with GDRNet is ready.")

            rospy.spin()

    def estimate_pose(self, req):
        print("request detection...")
        start_time = time.time()

        # === IN ===
        # --- rgb
        detection = req.det
        rgb = req.rgb
        depth = req.depth

        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480

        try:
            image = CvBridge().imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            depth.encoding = "mono16"
            depth_img = CvBridge().imgmsg_to_cv2(depth, "mono8")
        except CvBridgeError as e:
            print(e)

        print("obj_id: ", detection.name)
        print("conf_score: ", detection.score)
        print("bbox: ", detection.bbox)

        obj_id = -1
        for number in self.gdrn_predictor.objs:
            if self.gdrn_predictor.objs[number] == detection.name:
                obj_id = int(number) 
                break
        assert obj_id > 0
        
        outputs = torch.tensor([float(detection.bbox.ymin), float(detection.bbox.xmin), float(detection.bbox.ymax), float(detection.bbox.xmax),  detection.score, detection.score, float(obj_id - 1)])
        outputs = list((outputs.unsqueeze(0)))

        data_dict = self.gdrn_predictor.preprocessing(outputs=outputs, image=image, depth_img=None)
        out_dict = self.gdrn_predictor.inference(data_dict)
        poses = self.gdrn_predictor.postprocessing(data_dict, out_dict)
        #self.gdrn_predictor.gdrn_visualization(batch=data_dict, out_dict=out_dict, image=image)

        obj_name = self.gdrn_predictor.objs[int(obj_id)]

        R_0 = np.eye(4,4)
        R = poses[obj_name][ 0:3,0:3 ]
        R_0[ 0:3,0:3 ] = R
        t = poses[obj_name][ 0:3,3:4 ].ravel()

        rot_quat = tf.transformations.quaternion_from_matrix(R_0)
        print("R: ", R_0)
        print("t: ", t)
        print()

        br = tf.TransformBroadcaster()
        br.sendTransform((poses[obj_name][0][3], poses[obj_name][1][3], poses[obj_name][2][3]),
                     rot_quat,
                     rospy.Time.now(),
                     "pose",
                     self.frame_id)

        estimates = []
        estimate = PoseWithConfidence()
        estimate.name = detection.name
        estimate.confidence = detection.score
        estimate.pose = Pose()
        estimate.pose.position.x = poses[obj_name][0][3]
        estimate.pose.position.y = poses[obj_name][1][3]
        estimate.pose.position.z = poses[obj_name][2][3]
        estimate.pose.orientation.x = rot_quat[0]
        estimate.pose.orientation.y = rot_quat[1]
        estimate.pose.orientation.z = rot_quat[2]
        estimate.pose.orientation.w = rot_quat[3]
        estimates.append(estimate)
        response = estimate_posesResponse()
        response.poses = estimates

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        print()
        return response 
    
if __name__ == "__main__":
    GDRN_ROS()

