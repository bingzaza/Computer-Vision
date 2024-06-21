import os
import pickle

data_root = './data/cifar-100-python'

# 检查路径
if not os.path.exists(data_root):
    print(f"路径不存在: {data_root}")
else:
    print(f"路径存在: {data_root}")
    print("文件列表:", os.listdir(data_root))

# 检查 meta 文件
meta_file = os.path.join(data_root, 'meta')
if os.path.exists(meta_file):
    with open(meta_file, 'rb') as f:
        meta_data = pickle.load(f, encoding='bytes')
    print("meta 文件加载成功")
    print("meta 文件内容:", meta_data)
else:
    print("meta 文件不存在")

# 检查 train 文件
train_file = os.path.join(data_root, 'train')
if os.path.exists(train_file):
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
    print("train 文件加载成功")
    print("train 文件内容:", train_data.keys())
else:
    print("train 文件不存在")

# 检查 test 文件
test_file = os.path.join(data_root, 'test')
if os.path.exists(test_file):
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    print("test 文件加载成功")
    print("test 文件内容:", test_data.keys())
else:
    print("test 文件不存在")
