import os
import torch.nn as nn
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
import cv2
from PIL import Image


class myDataset(Data.Dataset):
    def __init__(self, root, label):
        # 1. 初始化文件路径或文件名列表。
        # 也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        self.root = root
        self.label = label
        self.path = os.path.join(self.root, self.label)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        #1.从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        #2.预处理数据（例如torchvision.Transform）。
        #3.返回数据对（例如图像和标签）。
        # 这里需要注意的是，第一步：read one data，是一个data
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root, self.label, img_name)
        img = Image.open(img_item_path)
        label = self.label
        return img, label

    def __len__(self):
        # 应该将0更改为数据集的总大小。
        return len(self.img_path)




