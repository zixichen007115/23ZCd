import sys

from torch.utils import data
import random
import numpy as np


class Data_sim(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, mode, model_name):
        """Initialize and preprocess the CelebA dataset."""
        self.mode = mode
        self.model_name = model_name
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.preprocess()

    def preprocess(self):

        data = np.load("../0_files/data_pseran.npz")
        pos_list = data["pos_list"]
        dir_list = data["dir_list"]
        act_list = data["act_list"].T
        # pos_list: segment, 3, steps
        # dir_list: segment, 3, 3, steps
        # act_list: segment*2, steps

        print("position  list shape:{}".format(np.shape(pos_list)))
        print("direction list shape:{}".format(np.shape(dir_list)))
        print("action    list shape:{}".format(np.shape(act_list)))
        print("action range:%.3f_%.3f" % (np.min(act_list), np.max(act_list)))

        random.seed(1)
        num_seg = np.shape(pos_list)[0]
        list_length = np.shape(act_list)[1]

        val_test_sample = random.sample(range(list_length), int(list_length * 0.3))
        val_sample = val_test_sample[:int(list_length * 0.1)]
        test_sample = val_test_sample[int(list_length * 0.1):]

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

        t_step = 5
        j = self.model_name

        for i in range(1, list_length - t_step):
            seg_input = np.zeros([t_step, 5])
            # 2 for actuation, 3 for previous state
            output = np.zeros([t_step, 2])
            for k in range(t_step):
                seg_input[k, 0] = vec[i + k + 1, j, 2]
                seg_input[k, 1] = vec[i + k + 1, j, 0]
                seg_input[k, 2] = vec[i + k + 1, j, 1]
                seg_input[k, 3:5] = act_list[2 * j:2 * j + 2, i + k - 1]
                output[k] = act_list[2 * j:2 * j + 2, i + k]

            if i in val_sample:
                self.val_dataset.append([seg_input.transpose(), output.transpose()])
            elif i in test_sample:
                self.test_dataset.append([seg_input.transpose(), output.transpose()])
            else:
                self.train_dataset.append([seg_input.transpose(), output.transpose()])

        print('Finished preprocessing the dataset...')
        print('train sample number: %d.' % len(self.train_dataset))
        print('validation sample number: %d.' % len(self.val_dataset))
        print('test sample number: %d.' % len(self.test_dataset))

    def __getitem__(self, index):
        if self.mode == 'train':
            dataset = self.train_dataset
        elif self.mode == 'test':
            dataset = self.test_dataset
        else:
            dataset = self.val_dataset
        seg_input, output = dataset[index]
        return seg_input.transpose(), output.transpose()

    def __len__(self):
        """Return the number of images."""
        if self.mode == 'train':
            return len(self.train_dataset)
        elif self.mode == 'test':
            return len(self.test_dataset)
        else:
            return len(self.val_dataset)


def get_loader(batch_size=32, mode='train', model_name=1, num_workers=1):
    """Build and return a data loader."""
    dataset = Data_sim(mode=mode, model_name=model_name)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True if mode == 'train' else False,
                                  num_workers=num_workers)
    return data_loader
