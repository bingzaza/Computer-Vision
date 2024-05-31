import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_classes=200):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
