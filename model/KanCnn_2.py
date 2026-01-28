import torch
import torch.nn as nn
import torch.nn.functional as F
from .KANLinear import KANLinear
import math

class CNNKAN(nn.Module):
    def __init__(self, input_size=32):
        super(CNNKAN, self).__init__()

        # 卷积与池化结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.5)

        # 自动计算卷积后尺寸
        conv_out_size = self._get_conv_output_size(input_size)

        # 动态 KAN 层
        self.kan1 = KANLinear(conv_out_size, 256)
        self.kan2 = KANLinear(256, 2)  # 二分类任务

        # Grad-CAM 相关
        self.feature_maps = None
        self.gradients = None

    def _get_conv_output_size(self, input_size):
        """根据输入尺寸动态计算卷积层输出维度"""
        size = input_size
        for _ in range(3):  # 三次池化，每次减半
            size = math.floor(size / 2)
        return 128 * size * size

    def save_gradient(self, grad):
        self.gradients = grad

    def get_activations(self):
        return self.feature_maps

    def get_activations_gradient(self):
        return self.gradients

    def forward(self, x, cam=False):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.selu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # 保存特征图
        self.feature_maps = x
        if cam:
            x.register_hook(self.save_gradient)

        x = x.view(x.size(0), -1)
        x = self.kan1(x)
        x = self.kan2(x)
        return x
