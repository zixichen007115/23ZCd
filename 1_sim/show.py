import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_kind', type=str, default='pseran', choices=['ran', 'pseran'])
config = parser.parse_args()

data = np.load("../0_files/data_" + config.data_kind + ".npz")

pos_list = data["pos_list"]
dir_list = data["dir_list"]
act_list = data["act_list"].T
# pos_list: segment, 3, steps
# dir_list: segment, 3, 3, steps
# pos_list: segment*2, steps
num_seg = np.shape(pos_list)[0]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([0.0, 0.2])
ax.set_xticks(np.arange(-0.1, 0.15, 0.05))
ax.set_yticks(np.arange(-0.1, 0.15, 0.05))
ax.set_zticks(np.arange(0.0, 0.25, 0.05))
ax.tick_params(labelsize=15)
label_font = 15
ax.set_xlabel('x(m)', fontsize=label_font)
ax.set_ylabel('y(m)', fontsize=label_font)
ax.set_zlabel('z(m)', fontsize=label_font)
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20
# ax.set_title(config.data_kind)
for i in range(num_seg):
    ax.scatter(pos_list[i, 0], pos_list[i, 1], pos_list[i, 2])
plt.legend()
plt.savefig('../0_files/dataset_' + config.data_kind)
plt.show()
