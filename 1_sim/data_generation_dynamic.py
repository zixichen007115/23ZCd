import numpy as np
from run_simulation_dynamic import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_kind', type=str, default='pseran', choices=['ran', 'pseran'])
config = parser.parse_args()

ctrl_step = 16000

act_list = np.random.rand(ctrl_step, 8) * 2 - 1

if config.data_kind == 'pseran':
    for i in range(int(ctrl_step / 4)):
        # 0-ctrl_step/4
        act_list[i, 2] = act_list[i, 0]
        act_list[i, 4] = act_list[i, 0]
        act_list[i, 6] = act_list[i, 0]
        act_list[i, 3] = act_list[i, 1]
        act_list[i, 5] = act_list[i, 1]
        act_list[i, 7] = act_list[i, 1]
        # ctrl_step/4-ctrl_step/2
        act_list[i + int(ctrl_step / 4), 2] = act_list[i + int(ctrl_step / 4), 0]
        act_list[i + int(ctrl_step / 4), 4] = act_list[i + int(ctrl_step / 4), 0]
        act_list[i + int(ctrl_step / 4), 3] = act_list[i + int(ctrl_step / 4), 1]
        act_list[i + int(ctrl_step / 4), 5] = act_list[i + int(ctrl_step / 4), 1]
        # ctrl_step/2-ctrl_step*3/4
        act_list[i + int(ctrl_step / 2), 2] = act_list[i + int(ctrl_step / 2), 0]
        act_list[i + int(ctrl_step / 2), 3] = act_list[i + int(ctrl_step / 2), 1]
        # ctrl_step*3/4-ctrl_step

pos_list, dir_list, act_list = main(ctrl_step=ctrl_step, num_seg=4, act_list=act_list)

np.savez('../0_files/data_' + config.data_kind, pos_list=pos_list, dir_list=dir_list, act_list=act_list)

print("%.3f_%.3f" % (np.min(pos_list[:, 0]), np.max(pos_list[:, 0])))
print("%.3f_%.3f" % (np.min(pos_list[:, 1]), np.max(pos_list[:, 1])))
print("%.3f_%.3f" % (np.min(pos_list[:, 2]), np.max(pos_list[:, 2])))
print("%.3f_%.3f" % (np.min(act_list), np.max(act_list)))
