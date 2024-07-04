import sys

import numpy as np
from tqdm import tqdm
from set_arm_environment import ArmEnvironment


class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict]

    def setup(self):
        self.set_arm()


def main(ctrl_step=1, num_seg=4, act_list=None):
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

    pos_list = np.zeros((num_seg, 3, len(act_list[:, 0])))
    dir_list = np.zeros((num_seg, 3, 3, len(act_list[:, 0])))
    ctrl_num = 0

    for k_sim in tqdm(range(total_steps)):

        if (k_sim % controller_step_skip) == 0:

            act_3d = act_list[ctrl_num]
            for i in range(3):
                activations[i] = np.concatenate(
                    (np.ones(25) * act_3d[0, i], np.ones(25) * act_3d[1, i],
                     np.ones(25) * act_3d[2, i], np.ones(25) * act_3d[3, i]))

            for i_col in range(num_seg):
                pos_list[i_col, :, ctrl_num] = env.shearable_rod.position_collection[:, 25 + i_col * 25]
                dir_list[i_col, :, :, ctrl_num] = env.shearable_rod.director_collection[:, :, 24 + i_col * 25]

            ctrl_num = ctrl_num + 1

        time, systems, done = env.step(time, activations)
    print(ctrl_num)
    return pos_list, dir_list, act_list
