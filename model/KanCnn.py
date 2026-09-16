import torch
import torch.nn as nn
import torch.nn.functional as F
from .KANLinear import KANLinear


class CNNKAN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNKAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.5)

        self.kan1 = None
        self.hidden_dim = 256
        self.kan2 = KANLinear(self.hidden_dim, num_classes)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)

        # 第一次 forward 时动态构建 kan1
        if self.kan1 is None:
            in_features = x.size(1)  # 自动获取展平后的维度
            self.kan1 = KANLinear(in_features, self.hidden_dim).to(x.device)

        x = self.kan1(x)
        x = self.kan2(x)
        return x
