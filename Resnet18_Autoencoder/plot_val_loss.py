# import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('./save/L2_val_loss.pkl', "rb") as f:
    l2_val_loss = pickle.load(f)

with open('./save/H1_val_loss.pkl', "rb") as f:
    h1_val_loss = pickle.load(f)

print(np.argsort(np.array(h1_val_loss) - np.array(l2_val_loss))[:5])
plt.plot(l2_val_loss, color='blue', label='$L_2$ Training', alpha=.7)
plt.plot(h1_val_loss, color='red', label='$H_1$ Training', alpha=.7)
plt.legend(fontsize=12)
plt.xlabel('Iteration', fontsize=15)
plt.ylabel('Test loss', fontsize=15)
plt.yscale('log')
plt.grid()

plt.tight_layout()
plt.savefig('./save/val_loss_plot.png', dpi=300)

plt.show()