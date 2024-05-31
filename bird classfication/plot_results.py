import matplotlib.pyplot as plt
import pandas as pd
import os

# 定义结果文件的目录
results_dir = '/logs'

# 读取CSV文件
def read_results(file_path):
    return pd.read_csv(file_path)

# 绘制训练集和验证集上的loss曲线
def plot_loss(results, label):
    plt.plot(results['epoch'], results['loss'], label=label)

# 绘制验证集上的accuracy变化
def plot_accuracy(results, label):
    plt.plot(results['epoch'], results['accuracy'], label=label)

# 主函数
def main():
    # 获取所有结果文件
    results_files = [
        'lr_0.001_ftlr_0.001_bs_32_epochs_25_results.csv',
        'lr_0.001_ftlr_0.001_bs_32_epochs_50_results.csv',
        'lr_0.01_ftlr_0.001_bs_32_epochs_25_results.csv',
        'lr_0.01_ftlr_0.001_bs_32_epochs_50_results.csv',
        'lr_0.01_ftlr_0.001_bs_64_epochs_25_results.csv',
        'lr_0.01_ftlr_0.001_bs_64_epochs_50_results.csv'
    ]
    
    # 绘制loss曲线
    plt.figure(figsize=(12, 6))
    for file_name in results_files:
        results = read_results(os.path.join(results_dir, file_name))
        plot_loss(results, label=file_name)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制accuracy曲线
    plt.figure(figsize=(12, 6))
    for file_name in results_files:
        results = read_results(os.path.join(results_dir, file_name))
        plot_accuracy(results, label=file_name)
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
