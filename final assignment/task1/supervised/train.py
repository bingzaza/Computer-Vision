import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 数据集相对路径
data_root = './data/cifar-100-python'

def load_cifar100_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'fine_labels']
        X = X.reshape(-1, 3, 32, 32).astype("float")
        Y = np.array(Y)
    return X, Y

def load_cifar100_train(data_root):
    # CIFAR-100 训练集只有一个文件
    filename = os.path.join(data_root, 'train')
    return load_cifar100_batch(filename)

def load_cifar100_test(data_root):
    # CIFAR-100 测试集只有一个文件
    filename = os.path.join(data_root, 'test')
    return load_cifar100_batch(filename)

class CIFAR100Dataset(Dataset):
    def __init__(self, data_root, train=True, transform=None):
        self.transform = transform
        if train:
            self.data, self.labels = load_cifar100_train(data_root)
        else:
            self.data, self.labels = load_cifar100_test(data_root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = np.transpose(img, (1, 2, 0))  # 转换为HWC格式
        img = Image.fromarray(img.astype('uint8'))  # 转换为PIL图像

        if self.transform:
            img = self.transform(img)

        return img, label

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载自定义CIFAR-100数据集
train_dataset = CIFAR100Dataset(data_root=data_root, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CIFAR100Dataset(data_root=data_root, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 加载预训练的ResNet-18模型
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(pretrained=True)

# 冻结卷积层的参数
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# 替换最后的全连接层
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)

# 将模型移动到GPU
model = model.cuda()

# 定义损失函数和优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 创建SummaryWriter
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/cifar100_experiment_1')

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        labels = labels.long()  # 将标签转换为 LongTensor

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

    # 记录训练损失
    writer.add_scalar('Training Loss', epoch_loss, epoch)

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        labels = labels.long()  # 将标签转换为 LongTensor

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')

# 记录准确率
writer.add_scalar('Test Accuracy', accuracy, num_epochs)

# 关闭SummaryWriter
writer.close()
