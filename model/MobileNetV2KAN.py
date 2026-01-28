import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from .KANLinear import KANLinear


class MobileNetV2KAN(nn.Module):
    def __init__(
        self,
        input_size=64,
        num_classes=2,
        hidden_dim=256,
        dropout_p=0.3,
        bottleneck_dim=512,
    ):
        super().__init__()

        self.input_size = input_size

        # Backbone
        self.backbone = mobilenet_v2(weights=None).features

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        # 推导 Flatten 维度（避免 forward 中动态建层）
        self.flatten_dim = self._infer_flatten_dim(input_size)

        # Bottleneck + Norm（提升稳定性与抗噪）
        self.proj = nn.Linear(self.flatten_dim, bottleneck_dim)
        self.proj_act = nn.GELU()
        self.norm = nn.LayerNorm(bottleneck_dim)

        # KAN Head
        self.kan1 = KANLinear(bottleneck_dim, hidden_dim)
        self.kan2 = KANLinear(hidden_dim, num_classes)

    def _infer_flatten_dim(self, input_size):
        training_state = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size, input_size)
            x = self.backbone(x)
            dim = x.view(1, -1).size(1)
        self.backbone.train(training_state)
        return dim

    def forward(self, x):
        assert x.shape[2] == self.input_size and x.shape[3] == self.input_size, \
            f"输入尺寸必须为 {self.input_size}×{self.input_size}"

        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        x = self.proj_act(self.proj(x))
        x = self.norm(x)

        x = self.dropout(x)
        x = self.kan1(x)
        x = self.kan2(x)
        return x
