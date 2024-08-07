import numpy as np


def trajectory_generation(ctrl_step=250, num_seg=6, task='spiral'):
    ctrl_step = ctrl_step + 1
    alpha_tar = np.zeros((ctrl_step, num_seg))
    sin_tar = np.zeros((ctrl_step, num_seg))
    cos_tar = np.zeros((ctrl_step, num_seg))

    if task == 'spiral':
        alpha_list = np.array([0.998, 0.995, 0.950, 0.850, 0.800, 0.650])

        alpha_tar[:, 0] = np.linspace(1, alpha_list[0], ctrl_step)
        alpha_tar[:, 1] = np.linspace(1, alpha_list[1], ctrl_step)
        alpha_tar[:, 2] = np.linspace(1, alpha_list[2], ctrl_step)
        alpha_tar[:, 3] = np.linspace(1, alpha_list[3], ctrl_step)
        alpha_tar[:, 4] = np.linspace(1, alpha_list[4], ctrl_step)
        alpha_tar[:, 5] = np.linspace(1, alpha_list[5], ctrl_step)

        sin_tar[:, 0] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 1] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 2] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 3] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 4] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 5] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))

        cos_tar[:, 0] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 1] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 2] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 3] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 4] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 5] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))

    elif task == 'obs':
        alpha_list = np.array([0.999, 0.999, 0.999, 0.999, 0.980, 0.700])

        alpha_tar[:, 0] = alpha_list[0]
        alpha_tar[:, 1] = alpha_list[1]
        alpha_tar[:, 2] = alpha_list[2]
        alpha_tar[:, 3] = alpha_list[3]
        alpha_tar[:, 4] = alpha_list[4]
        alpha_tar[:, 5] = alpha_list[5]

        sin_tar[:, 0] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 1] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 2] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 3] = np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 4] = -np.sin(np.linspace(0, np.pi * 2, ctrl_step))
        sin_tar[:, 5] = -np.sin(np.linspace(0, np.pi * 2, ctrl_step))

        cos_tar[:, 0] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 1] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 2] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 3] = np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 4] = -np.cos(np.linspace(0, np.pi * 2, ctrl_step))
        cos_tar[:, 5] = -np.cos(np.linspace(0, np.pi * 2, ctrl_step))

    else:
        alpha_list = np.array([0.999, 0.996, 0.985, 0.975, 0.925, 0.600])
        # alpha_list = np.array([0.999, 0.996, 0.985, 0.975, 0.925, 0.600])


        fir = int(ctrl_step / 5)

        alpha_tar[:fir, 0] = np.linspace(1, alpha_list[0], fir)
        alpha_tar[:fir, 1] = np.linspace(1, alpha_list[1], fir)
        alpha_tar[:fir, 2] = np.linspace(1, alpha_list[2], fir)
        alpha_tar[:fir, 3] = np.linspace(1, alpha_list[3], fir)
        alpha_tar[:fir, 4] = np.linspace(1, alpha_list[4], fir)
        alpha_tar[:fir, 5] = np.linspace(1, alpha_list[5], fir)

        sin_tar[:fir, 0] = 0
        sin_tar[:fir, 1] = 0
        sin_tar[:fir, 2] = 0
        sin_tar[:fir, 3] = 0
        sin_tar[:fir, 4] = 0
        sin_tar[:fir, 5] = 0

        cos_tar[:fir, 0] = 1
        cos_tar[:fir, 1] = 1
        cos_tar[:fir, 2] = 1
        cos_tar[:fir, 3] = 1
        cos_tar[:fir, 4] = 1
        cos_tar[:fir, 5] = -1

        fir_ = ctrl_step - fir
        alpha_tar[fir:, 0] = alpha_list[0]
        alpha_tar[fir:, 1] = alpha_list[1]
        alpha_tar[fir:, 2] = alpha_list[2]
        alpha_tar[fir:, 3] = alpha_list[3]
        alpha_tar[fir:, 4] = alpha_list[4]
        alpha_tar[fir:, 5] = alpha_list[5]

        sin_tar[fir:, 0] = np.sin(np.linspace(0, np.pi * 2, fir_))
        sin_tar[fir:, 1] = np.sin(np.linspace(0, np.pi * 2, fir_))
        sin_tar[fir:, 2] = np.sin(np.linspace(0, np.pi * 2, fir_))
        sin_tar[fir:, 3] = np.sin(np.linspace(0, np.pi * 2, fir_))
        sin_tar[fir:, 4] = np.sin(np.linspace(0, np.pi * 2, fir_))
        sin_tar[fir:, 5] = -np.sin(np.linspace(0, np.pi * 2, fir_))

        cos_tar[fir:, 0] = np.cos(np.linspace(0, np.pi * 2, fir_))
        cos_tar[fir:, 1] = np.cos(np.linspace(0, np.pi * 2, fir_))
        cos_tar[fir:, 2] = np.cos(np.linspace(0, np.pi * 2, fir_))
        cos_tar[fir:, 3] = np.cos(np.linspace(0, np.pi * 2, fir_))
        cos_tar[fir:, 4] = np.cos(np.linspace(0, np.pi * 2, fir_))
        cos_tar[fir:, 5] = -np.cos(np.linspace(0, np.pi * 2, fir_))

    for i in range(ctrl_step):
        for j in range(num_seg):
            rest = 1 - np.square(alpha_tar[i, j])
            if rest == 0:
                sin_tar[i, j] = 0
                cos_tar[i, j] = 0
            else:
                sin_tar[i, j] = np.sqrt(np.square(sin_tar[i, j]) * rest) * np.sign(sin_tar[i, j])
                cos_tar[i, j] = np.sqrt(np.square(cos_tar[i, j]) * rest) * np.sign(cos_tar[i, j])

    return alpha_tar, sin_tar, cos_tar
