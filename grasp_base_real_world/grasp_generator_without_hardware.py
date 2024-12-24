# conda环境需要导入的包
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 导入模型
from models.grasp_transformer.swin import SwinTransformerSys
from models.grcnn.grconvnet import GenerativeResnet

# 同目录下导入的包——>后续需要将其他目录下的包也导入进来
from camera_data import CameraData
from utils.hardware.device import get_device
# 硬件类
# from hardware.camera import RealSenseCamera
from utils.hardware.device import get_device
# 其他目录下的功能包
from utils.inference.post_process import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp
from utils.dataset_processing import image

# 定义之外的函数是为了更好处理数据集
def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))

def get_depth(out_putsize, depth_file_name):
    depth_img = image.DepthImage.from_tiff(depth_file_name)
    depth_img.normalise()
    tuple_output = (out_putsize,out_putsize)
    depth_img.resize(tuple_output)
    return depth_img.img


def get_rgb(out_putsize, rgb_file_name, normalise=True):
    rgb_img = image.Image.from_file(rgb_file_name)
    tuple_output = (out_putsize,out_putsize)
    rgb_img.resize(tuple_output)
    if normalise:
        rgb_img.normalise()
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))
    return rgb_img.img

class GraspGenerator:
    def __init__(self, saved_model_path, visualize=False):

        # 设置训练好的模型的路径 + 是否使用GPU
        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None

        # 1.相机设备输入 和 图像预处理
        # self.camera = RealSenseCamera(device_id=cam_id)
        self.cam_data = CameraData(include_depth=True, include_rgb=True)

        # 4.可视化
        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    def load_model(self):
        print('Loading model... ')
        model = SwinTransformerSys(in_chans=4, embed_dim=48, num_heads=[1, 2, 4, 8])
        device = torch.device("cuda:0")
        model = model.to(device)
        model.load_state_dict(torch.load(self.saved_model_path))
        model.eval()
        self.model = model
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self):

        # 获取rgb图和深度图
        # 图片文件路径，这里替换为你实际的png图片文件的路径
        rgb_path = "/home/junhaohu/dataset/test_model/depth_and_RGB/7.png"
        rgb_img = get_rgb(448,rgb_path)

        # 深度图需要从tiff文件中获取
        depth_path = "/home/junhaohu/dataset/test_model/depth_and_RGB/7.tiff"
        depth_img = get_depth(448,depth_path)

        # 在类中对输入的图像做处理得到x
        x = numpy_to_torch(
            np.expand_dims(np.concatenate(
                (np.expand_dims(depth_img, 0),
                 rgb_img),
                0
            ),0)
        )


        # rgb_img = rgb_img.transpose((1, 2, 0))
        # depth_img = depth_img.transpose((1, 2, 0))
        #
        # cv2.imshow('depth', depth_img)
        # cv2.imshow('rgb', rgb_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # Predict the grasp pose using the saved model
        with torch.no_grad():
            # 将x送到模型里面得到预测
            xc = x.to(self.device)
            pred = self.model.predict(xc)
        # 预测结果输出q是质量图片，ang是角度，wid是宽度
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        # cv2.imshow('q', q_img)
        # cv2.imshow('a', ang_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 送到检测抓取函数获得抓取姿态
        grasps = detect_grasps(q_img, ang_img, width_img)
        print(grasps)

        if self.fig:
            plot_grasp(fig=self.fig, rgb_img=get_rgb(448,rgb_path,False), grasps=grasps, save=True)

    def run(self):
        print("test begining")
        self.generate()

