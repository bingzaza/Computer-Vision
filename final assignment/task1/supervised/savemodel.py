import torch
import torchvision.models as models

# 定义模型
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 100)

# 将模型移动到GPU
model = model.cuda()

# 加载训练好的模型参数
model_path = './runs/cifar100_experiment_1/events.out.tfevents.1718188709.DESKTOP-6N2C3LU.6508.0'  # 这是假设模型参数已经保存在这里
model.load_state_dict(torch.load(model_path))
model.eval()

# 保存模型参数
torch.save(model.state_dict(), './resnet18_cifar100_final.pth')
print('Model parameters saved to resnet18_cifar100_final.pth')
