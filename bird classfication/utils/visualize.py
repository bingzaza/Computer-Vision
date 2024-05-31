import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('logs/results.csv')

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['loss'], label='Loss')
plt.plot(df['epoch'], df['accuracy'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.title('Training Results')
plt.show()
