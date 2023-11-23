import numpy as np
import matplotlib.pyplot as plt
from trajectory import trajectory_generation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='spiral', choices=['spiral', 'obs', 'straight'])
config = parser.parse_args()

if config.task == 'spiral':
    tra_name = 'A'
elif config.task == 'obs':
    tra_name = 'B'
else:
    tra_name = 'C'

data = np.load('../0_files/data_six/data_' + config.task + ".npz")

pos_list = data["pos_list"]
dir_list = data["dir_list"]
act_list = data["act_list"].T
real_list = data["real_list"]
shape_list = data["shape_list"]
ori_dir_list = data["ori_dir_list"]
# pos_list: segment, 3, steps
# dir_list: segment, 3, 3, steps
# act_list: 2, segment, steps
# real_list: segment, 3, steps
# shape_list: 21, 3, steps

print(np.shape(pos_list))
print(np.shape(dir_list))
print(np.shape(act_list))
print(np.shape(real_list))

num_seg = np.shape(pos_list)[0]
ctrl_step = np.shape(pos_list)[2]

alpha_tar, sin_tar, cos_tar = trajectory_generation(ctrl_step, num_seg, config.task)

err = np.zeros([num_seg, ctrl_step])
for i in range(num_seg):
    for j in range(ctrl_step):
        error = np.sqrt(
            np.square(real_list[i, 0, j] - alpha_tar[j, i]) + np.square(real_list[i, 1, j] - sin_tar[j, i]) + np.square(
                real_list[i, 2, j] - cos_tar[j, i]))
        err[i, j] = error
    print("%.2f" % np.mean(err[i] * 100))

plt.figure(figsize=(9, 6))
for test_seg in range(num_seg):
    plt.subplot(num_seg, 3, 1 + test_seg * 3)

    plt.plot(alpha_tar[:ctrl_step, test_seg], c='red', linewidth=3)
    plt.plot(real_list[test_seg, 0, :], c='blue')
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 250)
    plt.ylabel("m_" + str(test_seg + 1))
    if test_seg == 0:
        plt.title("V_z")

    plt.subplot(num_seg, 3, 2 + test_seg * 3)

    plt.plot(sin_tar[:ctrl_step, test_seg], c='red', linewidth=3)
    plt.plot(real_list[test_seg, 1, :], c='blue')
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 250)
    if test_seg == 0:
        plt.title("V_x")

    plt.subplot(num_seg, 3, 3 + test_seg * 3)
    plt.plot(cos_tar[:ctrl_step, test_seg], c='red', linewidth=3)
    plt.plot(real_list[test_seg, 2, :], c='blue')
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 250)
    if test_seg == 0:
        plt.title("V_y")

plt.suptitle("LSTM-" + tra_name)
plt.savefig('../0_files/data_six/conf-LSTM-' + tra_name)
plt.show()

plt.figure(figsize=(6, 6))
for test_seg in range(num_seg):
    plt.subplot(num_seg, 2, 1 + test_seg * 2)
    plt.plot(act_list[0, test_seg, :], c='blue')
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 250)
    plt.ylabel("m_" + str(test_seg + 1))
    if test_seg == 0:
        plt.title("a_0")

    plt.subplot(num_seg, 2, 2 + test_seg * 2)
    plt.plot(act_list[1, test_seg, :], c='blue')
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 250)
    if test_seg == 0:
        plt.title("a_1")

plt.suptitle("LSTM-" + tra_name)
plt.savefig('../0_files/data_six/act-LSTM-' + tra_name)
plt.show()

if config.task == 'obs':
    err = 0
    for i in range(ctrl_step):
        error = np.sqrt(np.square(pos_list[-1, 0, i]) + np.square(pos_list[-1, 1, i]))
        err += error
    print(err / ctrl_step / 0.2 * 100)

elif config.task == 'straight':
    err = 0
    ori_vec = np.zeros([3, 1])
    ori_vec[2] = 1
    for i in range(ctrl_step):
        vec = np.matmul(dir_list[-1, :, :, i].T, ori_vec)[:, 0]
        error = np.arccos(vec[2])
        err += error
    print(err / ctrl_step)
