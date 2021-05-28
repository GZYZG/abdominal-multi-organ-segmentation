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
from utils.utils import split_ct_to_slice_batch

on_server = True
size = config.slice_num
slice_thickness = 3
down_scale = 1


class MaskedSynapseDataset(IterableDataset):
    def __init__(self, ct_dir, seg_dir, slice_num, img_size=512, visible_class=None, transform=None):
        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('img', 'label'), self.ct_list))  # 以ct中的样本文件名为基础替换为标签文件名

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))  # 拼接标签文件的地址
        self.slice_num = slice_num
        assert visible_class is None or isinstance(visible_class, list)
        self.visible = visible_class if not visible_class else sorted(visible_class)

        self.transform = transform
        self.img_size = img_size

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
                ct_array = ct_batches[b]
                seg_array = seg_batches[b]

                if self.visible is not None:
                    _seg = np.zeros_like(seg_array)
                    for idx, cls in enumerate(self.visible):
                        _seg[seg_array == cls] = idx + 1
                    seg_array = _seg

                w, h = ct_array.shape[-2:]
                ct_array = ndimage.zoom(ct_array, (1, w / self.img_size, h / self.img_size), order=3)
                seg_array = ndimage.zoom(seg_array, (1, w / self.img_size, h / self.img_size), order=0)

                if self.transform:
                    ct_array = self.transform(ct_array)
                    seg_array = self.transform(seg_array)

                ct_tensor = torch.FloatTensor(ct_array).unsqueeze(0)
                seg_tensor = torch.FloatTensor(seg_array)

                yield ct_tensor, seg_tensor


# # 测试代码
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ct_dir = os.path.join(config.prep_train_dataset_dir, "CT")
    seg_dir = os.path.join(config.prep_train_dataset_dir, "GT")

    train_ds = MaskedSynapseDataset(ct_dir, seg_dir, slice_num=config.slice_num, visible_class=[1, 2, 3, 4])
    train_dl = DataLoader(train_ds, 1)
    for index, (ct, seg) in enumerate(train_dl):
        print(index, ct.size(), seg.size())
        print('----------------')
