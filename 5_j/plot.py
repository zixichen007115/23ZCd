import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import argparse
import imageio
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='spiral', choices=['spiral', 'obs', 'straight'])
config = parser.parse_args()

data = np.load('../0_files/data_j/data_' + config.task + ".npz")

pos_list = data["pos_list"]
dir_list = data["dir_list"]
act_list = data["act_list"].T
real_list = data["real_list"]
shape_list = data["shape_list"]

img_dir = "../0_files/data_j/img_" + config.task
if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.mkdir(img_dir)

num_seg = np.shape(pos_list)[0]
ctrl_step = np.shape(pos_list)[2]

for step in range(0, ctrl_step, 5):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0.0, 0.2])
    ax.set_xticks(np.arange(-0.1, 0.15, 0.05))
    ax.set_yticks(np.arange(-0.1, 0.15, 0.05))
    ax.set_zticks(np.arange(0.0, 0.25, 0.05))
    ax.tick_params(labelsize=15)
    colors = ['b', 'g', 'k']
    ax.scatter(0, 0, 0, c='r')
    ax.plot([0, 0.01], [0, 0], [0, 0], c=colors[0])
    ax.plot([0, 0], [0, 0.01], [0, 0], c=colors[1])
    ax.plot([0, 0], [0, 0], [0, 0.01], c=colors[2])

    for i in range(num_seg * 5 + 1):
        ax.scatter(shape_list[i, 0, step], shape_list[i, 1, step], shape_list[i, 2, step], c='r')

    for i in range(num_seg):
        ax.scatter(pos_list[i, 0, step], pos_list[i, 1, step], pos_list[i, 2, step], c='k')
        vec = np.zeros([3, 1, 3])
        vec[0, 0, 0] = 1
        vec[1, 0, 1] = 1
        vec[2, 0, 2] = 1

        for j in range(3):
            start = [pos_list[i, 0, step], pos_list[i, 1, step], pos_list[i, 2, step]]
            end = start + np.matmul(dir_list[i, :, :, step].T, vec[:, :, j])[:, 0] * 0.01
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c=colors[j])

    plt.savefig(img_dir + "/%03d.png" % (step + 1))
    plt.close()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

for step in range(0, ctrl_step, int(ctrl_step / 5)):
    print(step)

    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0.0, 0.2])
    ax.set_xticks(np.arange(-0.1, 0.15, 0.05))
    ax.set_yticks(np.arange(-0.1, 0.15, 0.05))
    ax.set_zticks(np.arange(0.0, 0.25, 0.05))
    ax.tick_params(labelsize=15)
    colors = ['b', 'g', 'k']
    ax.scatter(0, 0, 0, c='r')
    ax.plot([0, 0.01], [0, 0], [0, 0], c=colors[0])
    ax.plot([0, 0], [0, 0.01], [0, 0], c=colors[1])
    ax.plot([0, 0], [0, 0], [0, 0.01], c=colors[2])

    for i in range(num_seg * 5 + 1):
        ax.scatter(shape_list[i, 0, step], shape_list[i, 1, step], shape_list[i, 2, step], c='r',
                   alpha=0.2 + step * 0.8 / ctrl_step)

    for i in range(num_seg):
        ax.scatter(pos_list[i, 0, step], pos_list[i, 1, step], pos_list[i, 2, step], c='k',
                   alpha=0.2 + step * 0.8 / ctrl_step)

        if i == num_seg - 1:
            vec = np.zeros([3, 1, 3])
            vec[0, 0, 0] = 1
            vec[1, 0, 1] = 1
            vec[2, 0, 2] = 1

            for j in range(3):
                start = [pos_list[i, 0, step], pos_list[i, 1, step], pos_list[i, 2, step]]
                end = start + np.matmul(dir_list[i, :, :, step].T, vec[:, :, j])[:, 0] * 0.01
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c=colors[j],
                        alpha=0.2 + step * 0.8 / ctrl_step)

plt.savefig("../0_files/data_j/" + config.task + "_diagram.png")
plt.close()

imgs = []
file = sorted(os.listdir(img_dir))
for f in file:
    real_url = path.join(img_dir, f)
    imgs.append(real_url)

frames = []
for image_name in imgs:
    frames.append(imageio.imread(image_name))

imageio.mimsave("../0_files/data_j/" + config.task, frames, 'GIF', duration=0.1)
