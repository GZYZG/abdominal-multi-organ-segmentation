"""
在五例随机挑选的数据上做测试
共13种器官＋背景
(0) 背景
(1) spleen 脾
(2) right kidney 右肾
(3) left kidney 左肾
(4) gallbladder 胆囊
(5) esophagus 食管
(6) liver 肝脏
(7) stomach 胃
(8) aorta 大动脉
(9) inferior vena cava 下腔静脉
(10) portal vein and splenic vein 门静脉和脾静脉
(11) pancreas 胰腺
(12) right adrenal gland 右肾上腺
(13) left adrenal gland 左肾上腺
"""

import os
from time import time
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import xlsxwriter as xw
import scipy.ndimage as ndimage
from collections import OrderedDict
from utils.utils import load_model

from net.ResUnet_dice import Net
from config.config import config


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dataset_dir = config.prep_val_dataset_dir

test_ct_dir = os.path.join(dataset_dir, "CT")  # './val/CT/'
# val_seg_dir = os.path.join(config.val_dataset_dir, "GT")  # './val/GT/'

organ_pred_dir = os.path.join(dataset_dir, "pred")  # './val/pred/'
if not os.path.exists(organ_pred_dir):
    os.mkdir(organ_pred_dir)

module_dir = config.test_model_path  # 'output/module/net2480-0.718-0.812.pth'  # './module/net170-0.943-1.055.pth'

upper = 350
lower = -upper
down_scale = 0.5
size = config.slice_num
slice_thickness = 3


organ_list = [
    'spleen',
    'right kidney',
    'left kidney',
    'gallbladder',
    'esophagus',
    'liver',
    'stomach',
    'aorta',
    'inferior vena cava',
    'portal vein and splenic vein',
    'pancreas',
    'right adrenal gland',
    'left adrenal gland',
]

# 定义网络并加载参数
net = Net(training=False)
net.to(config.device)
if config.on_gpu:
    net = torch.nn.DataParallel(net).cuda(0)
# net = torch.nn.DataParallel(Net(training=False)).cuda()
state_dict = torch.load(module_dir, map_location=config.device)['model_state_dict']
# state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
net.load_state_dict(state_dict)
net.eval()
print(f"Testing model is loaded from {module_dir} ")

# 开始正式进行测试
for file_index, file in enumerate(os.listdir(test_ct_dir)):
    start_time = time()

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(test_ct_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    # 将灰度值在阈值之外的截断掉
    # ct_array[ct_array > upper] = upper
    # ct_array[ct_array < lower] = lower

    # 对CT使用双三次算法进行插值，插值之后的array依然是int16
    # ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)

    # 在轴向上进行切块取样
    flag = False
    start_slice = 0
    end_slice = start_slice + size - 1
    ct_array_list = []

    while end_slice <= ct_array.shape[0] - 1:
        ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])

        start_slice = end_slice + 1
        end_slice = start_slice + size - 1

    # 当无法整除的时候反向取最后一个block
    if end_slice is not ct_array.shape[0] - 1:
        flag = True
        count = ct_array.shape[0] - start_slice
        ct_array_list.append(ct_array[-size:, :, :])

    outputs_list = []
    with torch.no_grad():
        for ct_array in ct_array_list:

            ct_tensor = torch.FloatTensor(ct_array).to(config.device)  #.cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            ct_tensor = ct_tensor.unsqueeze(dim=0)

            outputs = net(ct_tensor)
            outputs = outputs.squeeze()

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            outputs_list.append(outputs.cpu().detach().numpy())
            del outputs

    # 执行完之后开始拼接结果
    pred_seg = np.concatenate(outputs_list[0:-1], axis=1)
    if flag is False:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)
    else:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count:, :, :]], axis=1)

    # 将金标准读入内存来计算dice系数
    # seg = sitk.ReadImage(os.path.join(val_seg_dir, file.replace('img', 'label')), sitk.sitkUInt8)
    # seg_array = sitk.GetArrayFromImage(seg)

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    # pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0)
    # pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
    pred_seg = np.argmax(pred_seg, axis=0)
    pred_seg = np.round(pred_seg).astype(np.uint8)

    print('size of pred: ', pred_seg.shape)
    # print('size of GT: ', seg_array.shape)




    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(organ_pred_dir, file))
    del pred_seg

    speed = time() - start_time

    print('{} inference finished, this case use {:.3f} s'.format(file, speed))
    print('-----------------------')
