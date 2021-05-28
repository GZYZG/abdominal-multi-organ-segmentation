import torch
import SimpleITK as sitk
import numpy as np
from medpy import metric


def save_model(epoch, model, optimizer, lr_schedule, path):
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_schedule': lr_schedule.state_dict()
        }, path)
        return True
    except Exception as exp:
        print(f"Error occurs while saving model to {path}. Error info: {exp}")
        return False


def load_model(model, path, optimizer=None, lr_schedule=None):
    """从checkpoint文件加载模型

    Parameters
    ----------
    model : nn.Module, 创建好的未加载参数的模型
    optimizer : 优化器，可选的
    lr_schedule : 学习率scheduler，可选的
    path : str, checkpoint文件的路径
    """
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        if lr_schedule is not None:
            lr_schedule.load_state_dict(checkpoint['lr_schedule'])

        return dict(zip(["model", "optimizer", "lr_schedule", "epoch"], [model, optimizer, lr_schedule, epoch]))
    except Exception as exp:
        print(f"Error occurs while load model from {path}. Error info: {exp}")
        raise RuntimeError(f"Load Model from {path} failed! Error info: {exp}")
        # return dict(zip(["model", "optimizer", "lr_schedule", "epoch"], [None, None, None, None]))


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


def get_output_dirname(args):
    s = "{dataset}_{batch_size}_{slice_num}_{width}-{height}"
    s = s.format(dataset=args.dataset, batch_size=args.batch_size, slice_num=args.slice_num,
                 width=args.CT_width, height=args.CT_height)

    return s


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def calculate_metric_dice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1
    else:
        return 0


def calculate_metric_hd95(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0, 0
