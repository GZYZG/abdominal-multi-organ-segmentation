import cv2
import SimpleITK as itk
from config.config import config
import os
from sklearn import preprocessing
import numpy as np
"""
- 归一化：
    1) 把数据变成(0, 1)或者（-1, 1）之间的小数。主要是为了数据处理方便提出来的，把数据映射到0～1范围之内处理，更加便捷快速;
    2) 把有量纲表达式变成无量纲表达式，便于不同单位或量级的指标能够进行比较和加权。
        归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量。
常见归一化方法：
    1) min-max normalization: x' = (x - min) / (max - min)
    2) mean normalization: x' = (x - mean) / ( max - min)
    3) non-linear normalization: 
        3.1) 对数函数转换：x' = lg(x) / lg(max(x))
        3.2) l1,l2 normalization: x' = x / l1(x) or x / l2(x)
        
- 标准化：
    在机器学习中，我们可能要处理不同种类的资料，例如，音讯和图片上的像素值，这些资料可能是高维度的，
    资料标准化后会使每个特征中的数值平均变为0(将每个特征的值都减掉原始资料中该特征的平均)、标准差变为1，
    这个方法被广泛的使用在许多机器学习算法中(例如：支持向量机、逻辑回归和类神经网络)。
    
- 中心化：
    平均值为0，对标准差无要求
    
- 归一化和标准化的区别：
    归一化是将样本的特征值转换到同一量纲下把数据映射到[0,1]或者[-1, 1]区间内，
    仅由变量的极值决定，因区间放缩法是归一化的一种。标准化是依照特征矩阵的列处理数据，
    其通过求z-score的方法，转换为标准正态分布，和整体样本分布相关，每个样本点都能对标准化产生影响。
    它们的相同点在于都能取消由于量纲不同引起的误差；都是一种线性变换，都是对向量X按照比例压缩再进行平移。
    
- 标准化和中心化的区别：
    标准化是原始分数减去平均数然后除以标准差，中心化是原始分数减去平均数。 所以一般流程为先中心化再标准化。
    
- 无量纲：
    我的理解就是通过某种方法能去掉实际过程中的单位，从而简化计算，消除单位带来的数值上的尺度差异。

"""


def normalize(X, norm='mean', axis=1):
    """
    对数据进行归一化，X.shape = (n_samples, n_features)
    Parameters:
    ----------
    X : 数据矩阵，需要被归一化的矩阵。
    norm : 归一化的方式。mean表示均值归一化， $X' = (X - X_mean) / (X_max - X_min)$,
            minmax表示最小最大值归一化，$X' = (X - X_min) / (X_max - X_min)$。
    axis : {0, 1}, default=1。表示沿着哪个轴进行归一化，0表示针对样本的各个特征进行归一化；1表示对每个样本进行归一化。

    Returns
    ----------
    X : 归一化后的数据矩阵
    """
    shape = X.shape
    if len(shape) == 1:
        X = np.reshape(X, (1, -1))
    if axis == 0:
        X = X.T
    X_max = X.max(axis=1)
    X_min = X.min(axis=1)
    if norm == "mean":
        "mean归一化后的对比效果更强"
        X_mean = X.mean(axis=1)
        X = (X - X_mean) / (X_max - X_min)
    elif norm == "minmax":
        X = (X - X_min) / (X_max - X_min)

    if axis == 0:
        X = X.T

    return X


if __name__ == "__main__":
    nii_name = "img0004.nii.gz"
    ct_path = os.path.join(config.val_dataset_dir, f"CT/{nii_name}")
    gt_path = os.path.join(config.val_dataset_dir, f"GT/{nii_name.replace('img', 'label')}")
    ct_data = itk.ReadImage(ct_path)
    gt_data = itk.ReadImage(gt_path)
    imgs = itk.GetArrayFromImage(ct_data)
    segs = itk.GetArrayFromImage(gt_data)
    i = 100
    upper = 350
    lower = -upper
    truncated = False
    normalized = True
    img = imgs[i]
    seg = segs[i]

    MAX = img.max()
    MIN = img.min()
    MEAN = img.mean()

    if truncated:
        img[img < lower] = lower
        img[img > upper] = upper
    if normalized:
        img = normalize(img.reshape(1, -1), norm='mean').reshape(-1, 512)
    # img += seg * 100
    cv2.imshow(f"NO.{i}-slice", img)
    # print(img)
    print(img.mean(), img.max(), img.min())

    cv2.waitKey()
