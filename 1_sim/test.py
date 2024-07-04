import numpy as np
def two2three(act_2d):
    # act_2d: 8, [-1, 1]
    num_seg = int(len(act_2d) / 2)
    act_3d = np.zeros((num_seg, 3))
    for i in range(num_seg):
        if act_2d[2 * i] > - 0.5 and act_2d[2 * i + 1] > 0:
            # 0 ~ 2 * pi / 3
            act_3d[i, 0] = act_2d[2 * i] + act_2d[2 * i + 1] / np.sqrt(3)
            act_3d[i, 1] = 2 / np.sqrt(3) * act_2d[2 * i + 1]

        elif act_2d[2 * i] > - 0.5 and act_2d[2 * i + 1] <= 0:
            # 4 * pi / 3  ~ 2 * pi
            act_3d[i, 0] = act_2d[2 * i] - act_2d[2 * i + 1] / np.sqrt(3)
            act_3d[i, 2] = -2 / np.sqrt(3) * act_2d[2 * i + 1]
        else:
            act_3d[i, 1] = (-2 * act_2d[2 * i] + act_2d[2 * i + 1])/2
            act_3d[i, 2] = (-2 * act_2d[2 * i] - act_2d[2 * i + 1])/2
    return act_3d

act_2d = np.zeros(8)
act_2d[0] = 1 / 2
act_2d[1] = np.sqrt(3) / 2
act_2d[0] = 1 / 2
act_2d[1] = -np.sqrt(3) / 2
# act_2d[2] = -1
# act_2d[5] = 1
# act_2d[7] = -1
print(two2three(act_2d))