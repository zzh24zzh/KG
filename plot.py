import matplotlib.pyplot as plt
import pickle
import numpy as np
with open('processing/investigated_hms.txt','r') as f:
    hms=f.read().splitlines()
with open('processing/investigated_tfs.txt','r') as f:
    tfs=f.read().splitlines()
all_chips=tfs+hms

# fig, ax = plt.subplots(figsize=(50,5))
# ax.set_xticks([i for i in range(len(all_chips))])
# temp=list(np.array(all_chips)[idx])
# ax.set_xticklabels(temp, fontsize=8,rotation=30)
# ax.plot(a[idx])
# plt.show()