import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer, input_size=None):
        self.model = model
        self.target_layer = target_layer
        self.input_size = input_size

        self.activations = None
        self.gradients = None

        # 只注册 forward hook；梯度用 tensor.register_hook 获取（避免 backward hook）
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        # 有些层可能返回 tuple/list，取第一个 tensor
        if isinstance(output, (tuple, list)):
            output = output[0]

        # 保存激活
        self.activations = output.detach()

        # 在该层输出 tensor 上注册梯度 hook（
        def _grad_hook(grad):
            self.gradients = grad.detach()

        output.register_hook(_grad_hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def generate(self, x, class_idx=None):
        """
        x: (1,3,H,W)
        return: cam numpy (H,W) in [0,1]
        """
        self.model.eval()

        # 清空缓存
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)

            # 需要对 forward 建图
            if not x.requires_grad:
                x = x.requires_grad_(True)

            logits = self.model(x)

            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())

            score = logits[:, class_idx].sum()
            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "GradCAM failed to capture activations/gradients. "
                    "Please check if target_layer is correct and is used in forward."
                )

            acts = self.activations          # (N,C,h,w)
            grads = self.gradients           # (N,C,h,w)

            if acts.dim() != 4 or grads.dim() != 4:
                raise RuntimeError(
                    "Target layer output has no spatial dims (not N,C,H,W). "
                    "Please choose a convolutional feature layer."
                )

            # GAP on gradients -> weights
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (N,C,1,1)
            cam = (weights * acts).sum(dim=1, keepdim=True) # (N,1,h,w)
            cam = F.relu(cam)

            # resize to input size
            cam = F.interpolate(
                cam,
                size=(x.shape[2], x.shape[3]),
                mode="bilinear",
                align_corners=False
            )

            cam = cam.squeeze(0).squeeze(0)  # (H,W)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            return cam.detach().cpu().numpy()
