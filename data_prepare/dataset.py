"""
随机取样方式下的数据集
"""

import os
import random
import scipy.ndimage as ndimage
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from config import config

on_server = True
size = config.slice_num
slice_thickness = 3
down_scale = .5


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
        ct_array = ndimage.zoom(ct_array, (1, down_scale, down_scale), order=3)
        seg_array = ndimage.zoom(seg_array, (1, down_scale, down_scale), order=3)

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)


ct_dir = '../train/CT/' \
    if on_server is False else os.path.join(config.train_dataset_dir, "CT")  # '/home/gzy/medical/abdominal-multi-organ-segmentation/train/CT/'
seg_dir = '../train/GT/' \
    if on_server is False else os.path.join(config.train_dataset_dir, "GT")  # '/home/gzy/medical/abdominal-multi-organ-segmentation/train/GT/'

train_ds = Dataset(ct_dir, seg_dir)



# Obtain a pointer and download the data
# syn3379050 = syn.get(entity='syn3379050 '  )

# Get the path to the local copy of the data file
# filepath = syn3379050.path

# # 测试代码
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, 6, True)
    for index, (ct, seg) in enumerate(train_dl):

        print(index, ct.size(), seg.size())
        print('----------------')
