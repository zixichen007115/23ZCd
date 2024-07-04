import numpy as np
from tqdm import tqdm
from set_arm_environment import ArmEnvironment
import torch


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


def restore_model(input_size=20, hidden_size=128, num_layers=4, output_size=8):
    from model import LSTM
    LSTM = LSTM(input_size, hidden_size, num_layers, output_size, device=torch.device('cpu'))
    LSTM_path = '../../0_files/LSTM-large.ckpt'
    LSTM.load_state_dict(torch.load(LSTM_path, map_location=lambda storage, loc: storage))
    return LSTM


class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict]

    def setup(self):
        self.set_arm()


def main(ctrl_step=1, num_seg=4, tar_list=None):
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

    lstm = restore_model()
    seg_input = np.zeros([t_step, 5 * num_seg])
    # 3 for state, 2 for actuation

    act_list = np.zeros((ctrl_step, num_seg, 2))
    pre_act = np.zeros([num_seg, 2])

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

            # vec = np.zeros([num_seg,3])
            for i in range(t_step - 1):
                seg_input[i] = np.copy(seg_input[i + 1])

            for i_col in range(num_seg):
                vec = np.matmul(dir_list[i_col, :, :, ctrl_num].T, ori_vec)[:, 0]
                seg_input[-2, i_col * 5] = vec[2]
                seg_input[-2, i_col * 5 + 1] = vec[0]
                seg_input[-2, i_col * 5 + 2] = vec[1]

                seg_input[-1, i_col * 5] = alpha_tar[ctrl_num + 1, i_col]
                seg_input[-1, i_col * 5 + 1] = sin_tar[ctrl_num + 1, i_col]
                seg_input[-1, i_col * 5 + 2] = cos_tar[ctrl_num + 1, i_col]
                seg_input[-1, i_col * 5 + 3:i_col * 5 + 5] = pre_act[i_col]

                real_list[i_col, :, ctrl_num] = seg_input[-2, i_col * 5:i_col * 5 + 3]

            nn_input_tensor = torch.Tensor(np.array([seg_input]))

            with torch.no_grad():
                nn_input_tensor.to(torch.device('cpu'))
                out = lstm(nn_input_tensor)
                out = out.cpu().numpy()
            act = out[0, -1, :]

            act_max = 1
            for i in range(len(act)):
                if np.abs(act[i]) > act_max:
                    act[i] = act_max * act[i] / np.abs(act[i])
            act_3d = two2three(act)
            act = act.reshape([num_seg, 2])

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
