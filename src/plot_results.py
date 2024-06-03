import matplotlib.pyplot as plt
import pandas as pd


data = {
    'Set size': ["18000", "21000", "24000", "27000"],
    'Best val accuracy': [0.50, 0.51, 0.5767, 0.68],
    'Test accuracy': [0.48, 0.48, .5505, 0.68]
}


df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))

bar_width = 0.35
index = df['Set size']

plt.bar(index, df['Best val accuracy'], bar_width, label='Best val accuracy')
plt.bar(index, df['Test accuracy'], bar_width, bottom=df['Best val accuracy'], label='Test accuracy')

plt.xlabel('Set size')
plt.ylabel('Accuracy')
plt.title('Training Results')
plt.legend()

for i in range(len(df)):
    plt.text(i, df['Best val accuracy'][i] / 2, f"{df['Best val accuracy'][i]:.2f}", ha='center', va='bottom', color='white')
    plt.text(i, df['Best val accuracy'][i] + df['Test accuracy'][i] / 2, f"{df['Test accuracy'][i]:.2f}", ha='center', va='bottom', color='white')

plt.show()
plt.savefig('acc_plots.png')
