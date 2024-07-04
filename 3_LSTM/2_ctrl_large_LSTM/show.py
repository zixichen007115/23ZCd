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

data = np.load('../../0_files/data_large/data_' + config.task + ".npz")

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
        error = np.linalg.norm(real_list[i, :, j] - [alpha_tar[j, i], sin_tar[j, i], cos_tar[j, i]])
        err[i, j] = error
    if config.task == 'obs':
        err[i, 0] = 0
    print("err: %.2f+-%.2f" % (np.mean(err[i] * 100), np.std(err[i] * 100)))

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
    elif test_seg == num_seg - 1:
        plt.xlabel('step')

    plt.subplot(num_seg, 3, 3 + test_seg * 3)
    plt.plot(cos_tar[:ctrl_step, test_seg], c='red', linewidth=3)
    plt.plot(real_list[test_seg, 2, :], c='blue')
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 250)
    if test_seg == 0:
        plt.title("V_y")

plt.suptitle("LSTM-" + tra_name)
plt.savefig('../../0_files/data_large/conf-LSTM-' + tra_name)
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
    elif test_seg == num_seg - 1:
        plt.xlabel('step')

    plt.subplot(num_seg, 2, 2 + test_seg * 2)
    plt.plot(act_list[1, test_seg, :], c='blue')
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 250)
    if test_seg == 0:
        plt.title("a_1")
    elif test_seg == num_seg - 1:
        plt.xlabel('step')


plt.suptitle("LSTM-" + tra_name)
plt.savefig('../../0_files/data_large/act-LSTM-' + tra_name)
plt.show()

dataset = np.load("../../0_files/data_pseran.npz")
pos_list_dataset = dataset["pos_list"]

if config.task == 'obs':
    err = 0
    plt.figure(figsize=(6, 6))
    plt.scatter(pos_list_dataset[-1, 0], pos_list_dataset[-1, 1], c='red', s=5)
    plt.scatter(pos_list[-1, 0], pos_list[-1, 1], c='blue', s=5)
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)
    plt.show()
    for i in range(ctrl_step):
        error = np.sqrt(np.square(pos_list[-1, 0, i]) + np.square(pos_list[-1, 1, i]))
        err += error
    print(err / ctrl_step / 0.2 * 100)

elif config.task == 'straight':
    err = 0
    plt.figure(figsize=(6, 6))
    plt.scatter(pos_list_dataset[-1, 0], pos_list_dataset[-1, 1], c='red', s=5)
    plt.plot(pos_list[-1, 0], pos_list[-1, 1], c='blue')
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)
    plt.show()
    ori_vec = np.zeros([3, 1])
    ori_vec[2] = 1
    for i in range(ctrl_step):
        vec = np.matmul(dir_list[-1, :, :, i].T, ori_vec)[:, 0]
        error = np.arccos(vec[2])
        err += error
    print(err / ctrl_step)
