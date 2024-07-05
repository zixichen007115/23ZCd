import sys

import numpy as np
from tqdm import tqdm
from set_arm_environment import ArmEnvironment
import torch
from j_ctrl import j_ctrl


def two2three(act_2d):
    # act_2d: (num_seg, 2);  [-1, 1]
    # act_3d: (num_seg, 3); [0, 1]

    num_seg = len(act_2d)
    act_3d = np.zeros((num_seg, 3))
    for i in range(num_seg):
        normed_act_2d = act_2d[i] / np.linalg.norm(act_2d[i])
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
        act_3d[i] *= np.linalg.norm(act_2d[i])
    return act_3d




class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict]

    def setup(self):
        self.set_arm()


def main_(ctrl_step=1, num_seg=4, tar_list=None):
    """ Create simulation environment """
    time_step = 2.5e-4
    controller_Hz = 1
    final_time = int(ctrl_step / controller_Hz)

    env = Environment(final_time, time_step=time_step, num_seg=num_seg)
    total_steps, systems = env.reset()

    controller_step_skip = int(1.0 / (controller_Hz * env.time_step))

    """ Read arm params """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(
            np.zeros(env.muscle_groups[m].activation.shape)
        )

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)

    ctrl_num = 0
    t_step = 5

    ori_vec = np.zeros([3, 1])
    ori_vec[2] = 1

    alpha_tar, sin_tar, cos_tar = tar_list
    pos_list = np.zeros((num_seg, 3, ctrl_step))
    shape_list = np.zeros((21, 3, ctrl_step))
    ori_dir_list = np.zeros((num_seg, 3, 3, ctrl_step))
    dir_list = np.zeros((num_seg, 3, 3, ctrl_step))
    real_list = np.zeros((num_seg, 3, ctrl_step))
    act_list = np.zeros((ctrl_step, 4, 2))

    vec_seq = np.zeros([t_step, num_seg * 3])
    for i in range(num_seg):
        vec_seq[:, i * 3] = 1
    act_seq = np.zeros([t_step - 1, num_seg * 2])

    pre_act = np.zeros([num_seg, 2])

    j = np.load('../0_files/jac_np.npy')

    for k_sim in tqdm(range(total_steps)):
        if (k_sim % controller_step_skip) == 0:

            for i in range(21):
                shape_list[i, :, ctrl_num] = env.shearable_rod.position_collection[:, 5 * i]

            for i_col in range(num_seg):
                pos_list[i_col, :, ctrl_num] = env.shearable_rod.position_collection[:, 25 + i_col * 25]
                ori_dir_list[i_col, :, :, ctrl_num] = env.shearable_rod.director_collection[:, :, 24 + i_col * 25]

            dir_list = np.copy(ori_dir_list)
            for i_col in range(num_seg - 1):
                i_col = 3 - i_col
                dir_list[i_col, :, :, ctrl_num] = np.matmul(ori_dir_list[i_col, :, :, ctrl_num],
                                                            np.linalg.inv(ori_dir_list[i_col - 1, :, :, ctrl_num]))

            for i_col in range(num_seg):
                vec = np.matmul(dir_list[i_col, :, :, ctrl_num].T, ori_vec)[:, 0]
                real_list[i_col, :, ctrl_num] = np.array([vec[2], vec[0], vec[1]])

            vec_seq[:-1] = np.copy(vec_seq[1:])
            vec_seq[-2] = real_list[:, :, ctrl_num].reshape(num_seg * 3)
            vec_seq[-1] = np.array([alpha_tar[ctrl_num + 1], sin_tar[ctrl_num + 1], cos_tar[ctrl_num + 1]]).T.reshape(num_seg * 3)

            act_seq[:-1] = np.copy(act_seq[1:])
            act_seq[-1]  = pre_act.reshape(num_seg * 2)

            act, j = j_ctrl(vec_seq, act_seq, j)
            act = act.reshape(num_seg, 2)

            act_3d = two2three(act)

            act_list[ctrl_num, :, :] = act
            pre_act = np.copy(act)

            activations[0] = np.concatenate(
                (np.ones(25) * act_3d[0, 0], np.ones(25) * act_3d[1, 0], np.ones(25) * act_3d[2, 0],
                 np.ones(25) * act_3d[3, 0]))
            activations[1] = np.concatenate(
                (np.ones(25) * act_3d[0, 1], np.ones(25) * act_3d[1, 1], np.ones(25) * act_3d[2, 1],
                 np.ones(25) * act_3d[3, 1]))
            activations[2] = np.concatenate(
                (np.ones(25) * act_3d[0, 2], np.ones(25) * act_3d[1, 2], np.ones(25) * act_3d[2, 2],
                 np.ones(25) * act_3d[3, 2]))

            ctrl_num = ctrl_num + 1

        time, systems, done = env.step(time, activations)
    return pos_list, dir_list, act_list, real_list, shape_list, ori_dir_list
