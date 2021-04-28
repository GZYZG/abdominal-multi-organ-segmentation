"""
随机取样方式下的数据集
"""

import os
import random
import scipy.ndimage as ndimage
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from torch.utils.data import IterableDataset
from config.config import config

on_server = True
size = config.slice_num
slice_thickness = 3
down_scale = 1


def split_ct_to_slice_batch(ct_path, batch_size=48, padding="zero"):
    """
    将一个CT数据分割为多个batch。CT数据shape为：(slice_num, w, h)
    :param ct_path: CT数据的路径
    :param batch_size: batch大小，即每个batch里slice的数量
    :param padding: 当CT数据的slice num不能被batch_size整除时的padding方式

    :return : 划分好的batch，shape为(batch_num, batch_size, w, h)
    """
    # print(f"processing {ct_path} ...")
    img = sitk.ReadImage(ct_path)
    data = sitk.GetArrayFromImage(img)
    #     print(data.shape)
    n, w, h = data.shape
    batch_num = n // batch_size
    if batch_num * batch_size != n:
        batch_num += 1
        padding_size = batch_num * batch_size - n
        if padding == "zero":
            padding = np.zeros(shape=(padding_size, w, h))
        elif padding == "shift":
            padding = data[:padding_size, :, :]
        #         print(padding.shape)
        data = np.concatenate([data, padding], axis=0)

    #     print(data.shape)
    batches = np.zeros(shape=(batch_num, batch_size, w, h))
    for i in range(batch_num):
        s = i * batch_size
        #         e = (i+1) * batch_size
        batches[i, :, :, :] = data[s:s + batch_size, :, :]

    #     print(batches.shape)
    return batches


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('img', 'label'), self.ct_list))  # 以ct中的样本文件名为基础替换为标签文件名

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))  # 拼接标签文件的地址

    def __getitem__(self, index):
        """
        :param index:
        :return: torch.Size([B, 1, 48, 256, 256]) torch.Size([B, 48, 256, 256])
        """

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # 在slice平面内随机选取48张slice
        start_slice = random.randint(0, ct_array.shape[0] - size)
        end_slice = start_slice + size - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        # 对CT使用双三次算法进行插值，插值之后的array依然是int16
        # ct_array = ndimage.zoom(ct_array, (1, down_scale, down_scale), order=3)
        # seg_array = ndimage.zoom(seg_array, (1, down_scale, down_scale), order=3)

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)


class SlicesDataset(IterableDataset):
    def __init__(self, ct_dir, seg_dir, slice_num):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('img', 'label'), self.ct_list))  # 以ct中的样本文件名为基础替换为标签文件名

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))  # 拼接标签文件的地址
        self.slice_num = slice_num

    def __iter__(self):
        """
        :param index:
        :return: torch.Size([B, 1, 48, 256, 256]) torch.Size([B, 48, 256, 256])
        """
        for index, file in enumerate(self.ct_list):
            ct_path = self.ct_list[index]
            seg_path = self.seg_list[index]

            ct_batches = split_ct_to_slice_batch(ct_path, batch_size=self.slice_num)
            seg_batches = split_ct_to_slice_batch(seg_path, batch_size=self.slice_num)

            batch_num = ct_batches.shape[0]
            for b in range(batch_num):
                ct_array = torch.FloatTensor(ct_batches[b]).unsqueeze(0)
                seg_array = torch.FloatTensor(seg_batches[b])
                yield ct_array, seg_array


ct_dir = os.path.join(config.prep_train_dataset_dir, "CT")
seg_dir = os.path.join(config.prep_train_dataset_dir, "GT")

# train_ds = Dataset(ct_dir, seg_dir)
train_ds = SlicesDataset(ct_dir, seg_dir, slice_num=config.slice_num)


# Obtain a pointer and download the data
# syn3379050 = syn.get(entity='syn3379050 '  )

# Get the path to the local copy of the data file
# filepath = syn3379050.path

# # 测试代码
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, 1)
    for index, (ct, seg) in enumerate(train_dl):
        print(index, ct.size(), seg.size())
        print('----------------')
