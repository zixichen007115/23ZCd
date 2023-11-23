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

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-0.1, 0.1])
ax.set_xlabel('x')
ax.set_ylim([-0.1, 0.1])
ax.set_ylabel('x')
ax.set_zlim([0.0, 0.2])
ax.set_zlabel('z')
ax.set_title(config.data_kind)
for i in range(4):
    ax.scatter(pos_list[i, 0], pos_list[i, 1], pos_list[i, 2])
plt.savefig('../0_files/dataset_' + config.data_kind)
plt.show()
