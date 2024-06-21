import torch
import torch.nn as nn
from resnet import ResNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 创建ResNet-18模型
model = ResNet(dataset='cifar100', depth=18, num_classes=100, bottleneck=False)

# 打印参数量
print(f"Total parameters: {count_parameters(model)}")

# 可选：打印每一层的参数量
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.numel())
