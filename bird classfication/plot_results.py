import matplotlib.pyplot as plt
import pandas as pd
import os

# �������ļ���Ŀ¼
results_dir = '/logs'

# ��ȡCSV�ļ�
def read_results(file_path):
    return pd.read_csv(file_path)

# ����ѵ��������֤���ϵ�loss����
def plot_loss(results, label):
    plt.plot(results['epoch'], results['loss'], label=label)

# ������֤���ϵ�accuracy�仯
def plot_accuracy(results, label):
    plt.plot(results['epoch'], results['accuracy'], label=label)

# ������
def main():
    # ��ȡ���н���ļ�
    results_files = [
        'lr_0.001_ftlr_0.001_bs_32_epochs_25_results.csv',
        'lr_0.001_ftlr_0.001_bs_32_epochs_50_results.csv',
        'lr_0.01_ftlr_0.001_bs_32_epochs_25_results.csv',
        'lr_0.01_ftlr_0.001_bs_32_epochs_50_results.csv',
        'lr_0.01_ftlr_0.001_bs_64_epochs_25_results.csv',
        'lr_0.01_ftlr_0.001_bs_64_epochs_50_results.csv'
    ]
    
    # ����loss����
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

    # ����accuracy����
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
