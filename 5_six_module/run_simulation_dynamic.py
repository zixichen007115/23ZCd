import numpy as np
from tqdm import tqdm
from set_arm_environment import ArmEnvironment
import torch


def restore_model(t_step=5, hidden_size=128, num_layers=4, output_size=2):
    from model import LSTM
    input_size = t_step * 5 + 1
    LSTM = LSTM(input_size, hidden_size, num_layers, output_size, device=torch.device('cpu'))
    LSTM_path = '../0_files/LSTM-bi.ckpt'
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
    shape_list = np.zeros((num_seg * 5 + 1, 3, ctrl_step))
    ori_dir_list = np.zeros((num_seg, 3, 3, ctrl_step))
    dir_list = np.zeros((num_seg, 3, 3, ctrl_step))
    real_list = np.zeros((num_seg, 3, ctrl_step))

    lstm = restore_model(t_step=t_step)
    seg_input = np.zeros([num_seg, t_step * 5 + 1])

    act_list = np.zeros((ctrl_step, num_seg, 2))
    pre_act = np.zeros([num_seg, 2])

    for k_sim in tqdm(range(total_steps)):
        if (k_sim % controller_step_skip) == 0:

            for i in range(num_seg * 5 + 1):
                shape_list[i, :, ctrl_num] = env.shearable_rod.position_collection[:, 5 * i]

            for i_col in range(num_seg):
                pos_list[i_col, :, ctrl_num] = env.shearable_rod.position_collection[:, 25 + i_col * 25]
                ori_dir_list[i_col, :, :, ctrl_num] = env.shearable_rod.director_collection[:, :, 24 + i_col * 25]

            dir_list = np.copy(ori_dir_list)
            for i_col in range(num_seg - 1):
                i_col = num_seg - 1 - i_col
                dir_list[i_col, :, :, ctrl_num] = np.matmul(ori_dir_list[i_col, :, :, ctrl_num],
                                                            np.linalg.inv(ori_dir_list[i_col - 1, :, :, ctrl_num]))

            for i_col in range(num_seg):
                vec = np.matmul(dir_list[i_col, :, :, ctrl_num].T, ori_vec)[:, 0]

                for j in range(t_step - 1):
                    seg_input[i_col, j * 5:j * 5 + 5] = np.copy(seg_input[i_col, j * 5 + 5:j * 5 + 10])

                seg_input[i_col, -11] = vec[2]
                seg_input[i_col, -10] = vec[0]
                seg_input[i_col, -9] = vec[1]

                seg_input[i_col, -6] = alpha_tar[ctrl_num + 1, i_col]
                seg_input[i_col, -5] = sin_tar[ctrl_num + 1, i_col]
                seg_input[i_col, -4] = cos_tar[ctrl_num + 1, i_col]
                seg_input[i_col, -3: -1] = np.copy(pre_act[i_col])

                seg_input[i_col, -1] = 1 - 2 * i_col / (num_seg - 1)

                real_list[i_col, :, ctrl_num] = seg_input[i_col, -11:-8]

            nn_input_tensor = torch.Tensor(np.array([seg_input]))

            with torch.no_grad():
                nn_input_tensor.to(torch.device('cpu'))
                out = lstm(nn_input_tensor)
                out = out.cpu().numpy()
            act = out[0, :, :]

            act_max = 1
            for i in range(num_seg):
                for j in range(2):
                    if np.abs(act[i, j]) > act_max:
                        act[i, j] = act_max * act[i, j] / np.abs(act[i, j])

            act_list[ctrl_num, :, :] = act
            pre_act = np.copy(act)

            for i in range(4):
                act_real = np.ones(25 * num_seg)
                for j in range(num_seg):
                    if i < 2:
                        des_act = np.max([0, act_list[ctrl_num, j, int(i % 2)]])
                    else:
                        des_act = np.max([0, -act_list[ctrl_num, j, int(i % 2)]])
                    act_real[j * 25: j * 25 + 25] = des_act

                activations[i] = act_real

            ctrl_num = ctrl_num + 1

        time, systems, done = env.step(time, activations)
    return pos_list, dir_list, act_list, real_list, shape_list, ori_dir_list
