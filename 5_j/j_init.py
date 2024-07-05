import sys

import numpy as np
import torch

data = np.load("../0_files/data_ran.npz")
pos_list = data["pos_list"]
dir_list = data["dir_list"]
act_list = data["act_list"].T
# pos_list: segment, 3, steps
# dir_list: segment, 3, 3, steps
# act_list: segment * 2, steps

print("position  list shape:{}".format(np.shape(pos_list)))
print("direction list shape:{}".format(np.shape(dir_list)))
print("action    list shape:{}".format(np.shape(act_list)))

print("action range:%.3f_%.3f" % (np.min(act_list), np.max(act_list)))

num_seg = np.shape(pos_list)[0]
list_length = np.shape(act_list)[1]

ori_vec = np.zeros([3, 1])
ori_vec[2] = 1

for i in range(list_length):
    for j in range(num_seg - 1):
        j = 3 - j
        dir_list[j, :, :, i] = np.matmul(dir_list[j, :, :, i], np.linalg.inv(dir_list[j - 1, :, :, i]))

vec = np.zeros([list_length, num_seg, 3])
for i in range(list_length):
    for j in range(num_seg):
        vec[i, j] = np.matmul(dir_list[j, :, :, i].T, ori_vec)[:, 0]

j_np = np.zeros([num_seg * 3, num_seg * 2])

dif_vec = np.copy(vec)
for i in range(15):
    dif_vec[i + 1] -= vec[i]
dif_vec_op = (dif_vec[2:12].reshape(10, num_seg * 3)).T
print(dif_vec_op.shape)
# num_seg * 3, 10

dif_act = np.copy(act_list)
for i in range(15):
    dif_act[:, i + 1] -= act_list[:, i]
dif_act_op = dif_act[:, 1:11]
print(dif_act_op.shape)
# print(dif_act_op)
# sys.exit()

# num_seg * 2, 12

j_tensor = torch.tensor(j_np, requires_grad=True)
optim_m = torch.optim.Adam([j_tensor], 0.05, [0.9, 0.999])

dif_poses_tensor = torch.tensor(dif_vec_op, requires_grad=False)
dif_acts_tensor = torch.tensor(dif_act_op, requires_grad=False)

for it in range(100):
    loss = dif_poses_tensor - j_tensor @ dif_acts_tensor
    loss_sum = loss.norm()
    print(loss_sum)
    optim_m.zero_grad()
    loss_sum.backward()
    optim_m.step()

jac_np = j_tensor.detach().numpy()

np.save("../0_files/jac_np", jac_np)
