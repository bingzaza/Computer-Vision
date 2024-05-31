import os

learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64]
epochs = [25, 50]

for lr in learning_rates:
    for bs in batch_sizes:
        for ep in epochs:
            os.system(f'python main.py --learning_rate {lr} --batch_size {bs} --epochs {ep}')
