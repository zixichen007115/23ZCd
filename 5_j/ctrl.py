import numpy as np
from run_simulation_dynamic import main_
from trajectory import trajectory_generation
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='spiral', choices=['spiral', 'obs', 'straight'])
config = parser.parse_args()

ctrl_step = 250
num_seg = 4

if not os.path.exists("../0_files/data_j"):
    os.mkdir("../0_files/data_j")

alpha_tar, sin_tar, cos_tar = trajectory_generation(ctrl_step, num_seg, config.task)

pos_list, dir_list, act_list, real_list, shape_list, ori_dir_list = (
    main_(ctrl_step=ctrl_step, num_seg=num_seg, tar_list=(alpha_tar, sin_tar, cos_tar)))

np.savez('../0_files/data_j/data_'+config.task, pos_list=pos_list, dir_list=dir_list, act_list=act_list,
         real_list=real_list, shape_list=shape_list, ori_dir_list=ori_dir_list)
