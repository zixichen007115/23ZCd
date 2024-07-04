import sys

import numpy as np
from run_simulation_dynamic import main
import argparse


def two2three(act_2d):
    # act_2d: num_seg * 2;  [-1, 1]
    # act_3d: (num_seg, 3); [0, 1]

    num_seg = int(len(act_2d) / 2)
    act_3d = np.zeros((num_seg, 3))
    for i in range(num_seg):
        normed_act_2d = act_2d[2 * i:2 * i + 2] / np.linalg.norm(act_2d[2 * i:2 * i + 2])
        if normed_act_2d[0] > - 0.5 and normed_act_2d[1] > 0:
            # 0 ~ 2 * pi / 3
            act_3d[i, 0] = normed_act_2d[0] + normed_act_2d[1] / np.sqrt(3)
            act_3d[i, 1] = 2 / np.sqrt(3) * normed_act_2d[1]

        elif normed_act_2d[0] > - 0.5 and normed_act_2d[1] <= 0:
            # 4 * pi / 3  ~ 2 * pi
            act_3d[i, 0] = normed_act_2d[0] - normed_act_2d[1] / np.sqrt(3)
            act_3d[i, 2] = -2 / np.sqrt(3) * normed_act_2d[1]
        else:
            # 2 * pi / 3  ~ 4 * pi / 3
            act_3d[i, 1] = (-2 * normed_act_2d[0] + normed_act_2d[1]) / 2
            act_3d[i, 2] = (-2 * normed_act_2d[0] - normed_act_2d[1]) / 2
        act_3d[i] *= np.linalg.norm(act_2d[2 * i:2 * i + 2])
    return act_3d


parser = argparse.ArgumentParser()
parser.add_argument('--data_kind', type=str, default='pseran', choices=['ran', 'pseran'])
config = parser.parse_args()

ctrl_step = 12000
num_seg = 4

act_list = np.random.rand(ctrl_step, num_seg * 2) * 2 - 1
if config.data_kind == 'pseran':
    for i in range(int(ctrl_step / num_seg)):
        # 0-ctrl_step/4
        act_list[i, 2] = act_list[i, 0]
        act_list[i, 4] = act_list[i, 0]
        act_list[i, 6] = act_list[i, 0]

        act_list[i, 3] = act_list[i, 1]
        act_list[i, 5] = act_list[i, 1]
        act_list[i, 7] = act_list[i, 1]

        # ctrl_step/4-ctrl_step*2/4
        act_list[i + int(ctrl_step / num_seg), 2] = act_list[i, 0]
        act_list[i + int(ctrl_step / num_seg), 4] = act_list[i, 0]

        act_list[i + int(ctrl_step / num_seg), 3] = act_list[i, 1]
        act_list[i + int(ctrl_step / num_seg), 5] = act_list[i, 1]

        # ctrl_step*2/4-ctrl_step*3/4
        act_list[i + int(ctrl_step / num_seg * 2), 2] = act_list[i, 0]

        act_list[i + int(ctrl_step / num_seg * 2), 3] = act_list[i, 1]

act_3d_list = np.zeros([ctrl_step, num_seg, 3])
for i in range(ctrl_step):
    act_3d_list[i] = two2three(act_list[i])
print("%.3f_%.3f" % (np.min(act_list), np.max(act_list)))
print("%.3f_%.3f" % (np.min(act_3d_list), np.max(act_3d_list)))


pos_list, dir_list, act_3d_list = main(ctrl_step=ctrl_step, num_seg=4, act_list=act_3d_list)

np.savez('../0_files/data_' + config.data_kind, pos_list=pos_list, dir_list=dir_list, act_list=act_list)

print("%.3f_%.3f" % (np.min(pos_list[:, 0]), np.max(pos_list[:, 0])))
print("%.3f_%.3f" % (np.min(pos_list[:, 1]), np.max(pos_list[:, 1])))
print("%.3f_%.3f" % (np.min(pos_list[:, 2]), np.max(pos_list[:, 2])))
print("%.3f_%.3f" % (np.min(act_list), np.max(act_list)))
