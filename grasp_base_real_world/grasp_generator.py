# conda环境需要导入的包
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 同目录下导入的包——>后续需要将其他目录下的包也导入进来
from camera_data import CameraData
# 硬件类
# from hardware.camera import RealSenseCamera
from utils.hardware.device import get_device
# 其他目录下的功能包
from grcnn.inference.post_process import post_process_output
from grasp_transformer.utils.dataset_processing.grasp import detect_grasps
from grcnn.utils.visualisation.plot import plot_grasp


class GraspGenerator:
    def __init__(self, saved_model_path, cam_id, visualize=False):

        # 设置训练好的模型的路径 + 是否使用GPU
        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None

        # 1.相机设备输入 和 图像预处理
        # self.camera = RealSenseCamera(device_id=cam_id)
        self.cam_data = CameraData(include_depth=True, include_rgb=True)
        # Connect to camera
        # self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        # 2.手眼标定结果倒入
        self.cam_pose = np.loadtxt('saved_data/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('saved_data/camera_depth_scale.txt', delimiter=' ')

        # 3.应该是将最后算出来的抓取结果放到对应文件中
        # homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        # self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        # self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        # self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")

        # 4.可视化
        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    def load_model(self):
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self):
        # Get RGB-D image from camera
        # image_bundle = self.camera.get_image_bundle()
        # rgb = image_bundle['rgb']
        # depth = image_bundle['aligned_depth']

        # 获取rgb图和深度图
        # 图片文件路径，这里替换为你实际的png图片文件的路径
        rgb_path = "/home/junhaohu/dataset/test_model/RGB.png"
        # 使用Pillow库的Image.open函数打开图片文件
        rgb_image = Image.open(rgb_path)
        # 将图片数据转换为numpy数组，这里使用np.asanyarray
        rgb = np.asanyarray(rgb_image)
        depth_path = "/home/junhaohu/dataset/test_model/mask.png"
        depth_image = Image.open(depth_path)
        depth = np.asanyarray(depth_image)

        # 在类中对输入的图像做处理得到x
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            # 将x送到模型里面得到预测
            xc = x.to(self.device)
            pred = self.model.predict(xc)
        # 预测结果输出q是质量图片，ang是角度，wid是宽度
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        # 送到检测抓取函数获得抓取姿态
        grasps = detect_grasps(q_img, ang_img, width_img)

        # Get grasp position from model output
        # 可以加一个判断：如果检测抓取姿态为0，则放弃抓取
        # 将图像的点与实际进行对应，具体转换关系得再细看————转换关系->图像位姿和实际位姿映射
        # pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale - 0.04
        # pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
        #                     pos_z / self.camera.intrinsics.fx)
        # pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
        #                     pos_z / self.camera.intrinsics.fy)
        #
        # if pos_z == 0:
        #     return
        #
        # target = np.asarray([pos_x, pos_y, pos_z])
        # target.shape = (3, 1)
        # print('target: ', target)
        #
        # # Convert camera to robot coordinates机械臂坐标下的位置
        # camera2robot = self.cam_pose
        # target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
        # target_position = target_position[0:3, 0]
        #
        # # Convert camera to robot angle机械臂坐标下的抓取姿态，可能有问题，需要看一下
        # angle = np.asarray([0, 0, grasps[0].angle])
        # angle.shape = (3, 1)
        # target_angle = np.dot(camera2robot[0:3, 0:3], angle)
        #
        # # 加上抓取的宽度
        #
        # # Concatenate grasp pose with grasp angle
        # grasp_pose = np.append(target_position, target_angle[2])
        #
        # print('grasp_pose: ', grasp_pose)
        #
        # np.save(self.grasp_pose, grasp_pose)

        if self.fig:
            plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)

    def run(self):
        while True:
            if np.load(self.grasp_request):
                self.generate()
                np.save(self.grasp_request, 0)
                np.save(self.grasp_available, 1)
            else:
                time.sleep(0.1)
