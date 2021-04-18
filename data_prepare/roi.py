import cv2
import SimpleITK as itk
from config.config import config
import os
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
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
        X = (X.T - X_mean) / (X_max - X_min)
    elif norm == "minmax":
        X = (X.T - X_min) / (X_max - X_min)
    X = X.T
    if axis == 0:
        X = X.T

    return X


def mask_body(img, thresh=-1):
    """
    对医学图像进行掩膜，只保留图像中关于身体的部分。img.shape = (width, height)，输入必须为0~255灰度级的图像。
    Parameters:
    ----------
    img : 源图像。
    thresh : 罗阔面积阈值，打印等于的保留，小于等于的舍弃。

    Returns
    ----------
    masks: 掩膜矩阵，masks[n]与源图像一样大，n表示选择的轮廓的数量，masks[n][i][j] = 1表示像素数据身体部分，masks[n][i][j] = 0表示不数据身体部分。
    """
    # img = np.array(img)
    # print(img.dtype.name)
    if img.dtype.name == 'int8':
        raise ValueError(f"Input image's dtype should be int8 but get {img.dtype.name}")

    tmp = img  # cv2.convertScaleAbs(img)

    # 先在源图像中找到边缘
    edge = cv2.Canny(tmp, 0, 255)
    # 在边缘的基础上找到轮廓
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.array([])
    # 计算各个轮廓的面积，找到面积最大的轮廓
    areas = np.zeros(len(contours))
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        areas[idx] = area
        # print(f"{idx}-th contour's area: {area}")
    if thresh == -1:
        selected = [areas.argmax()]
    else:
        selected = areas[areas >= thresh]
    # 生成选定轮廓的掩膜，并提取出源图像中对应的区域
    if len(selected) == 0:
        return np.array([])
    masks = np.zeros(shape=(len(selected), img.shape[0], img.shape[1]), dtype=np.uint8)
    for idx, con in enumerate(selected):
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, idx, (1, 1, 1), -1)
        masks[idx] = mask

    return masks


def preprocess(X, norm="", axis=1, mask=False, truncated=False, upper=None, lower=None, copy=False):
    """
    对数据进行预处理，X.shape = (n_samples, width, height)。该先归一化呢还是先掩膜呢？
    预处理流程：normalization -> mask body
    Parameters:
    ----------
    X : 数据矩阵，需要被预处理的矩阵。
    norm : 归一化的方式。mean表示均值归一化， $X' = (X - X_mean) / (X_max - X_min)$,
            minmax表示最小最大值归一化，$X' = (X - X_min) / (X_max - X_min)$。
    axis : {0, 1}, default=1。表示沿着哪个轴进行归一化，0表示针对样本的各个特征进行归一化；1表示对每个样本进行归一化。
    copy : {True, False}, default=False。表示是否直接修改源图像，为True时先copy一份源图像再进行操作。

    Returns
    ----------
    X : 预处理后的数据矩阵
    """
    if norm in ("mean", "minmax"):
        normalized = True
    else:
        normalized = False

    if copy:
        X = X.copy()

    if truncated:
        if upper is None:
            raise ValueError(f"upper value can't be None when truncated= is True")
        if lower is None:
            lower = -upper
        X[X < lower] = lower
        X[X > upper] = upper

    if np.ndim(X) == 2:
        X = np.expand_dims(X, axis=0)

    if normalized:
        n, w, h = X.shape
        X = X.reshape((n, -1))
        X = normalize(X, norm=norm, axis=axis)
        X = X.reshape((-1, w, h))
    elif mask:
        X_copy = X.copy()
        n, w, h = X_copy.shape
        X_copy= X_copy.reshape((n, -1))
        X_copy = normalize(X_copy, norm='mean', axis=axis)
        X_copy = X_copy.reshape((-1, w, h))

        tmp = X_copy
        for idx, img in enumerate(tmp):
            masks = mask_body(img.copy())
            if len(masks) != 0:
                body = np.zeros_like(X[idx])
                for m in masks:
                    body += np.where(mask == 1, X[idx], m)
                X[idx] = body

    return X


def caleGrayHist(image):
    # 灰度图像的高、宽
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)  # 图像的灰度级范围是0~255
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1

    return grayHist


def show_gray_hist(gray_hist):
    x_range = range(256)
    plt.plot(x_range, gray_hist, 'r', linewidth=1.5, c='orange')
    # 设置坐标轴的范围
    y_maxValue = np.max(gray_hist)
    plt.axis([0, 255, 0, y_maxValue])  # 画图范围
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels")
    plt.show()


def get_start_end(nii_label):
    start = len(nii_label)-1
    end = 0
    for idx, label in enumerate(nii_label):
        if np.alltrue(label == 0):
            start = idx
            break
    for idx in range(len(nii_label)-1, 0):
        if np.alltrue(nii_label[idx] == 0):
            end = idx
            break

    return start, end


if __name__ == "__main__":
    nii_name = "img0004.nii.gz"
    ct_path = os.path.join(config.val_dataset_dir, f"CT/{nii_name}")
    gt_path = os.path.join(config.val_dataset_dir, f"GT/{nii_name.replace('img', 'label')}")
    ct_data = itk.ReadImage(ct_path)
    gt_data = itk.ReadImage(gt_path)
    imgs = itk.GetArrayFromImage(ct_data)
    segs = itk.GetArrayFromImage(gt_data)
    i = 100
    upper = 1000
    lower = -upper
    truncated = True
    normalized = True
    norm = "minmax"

    # imgs_copy = preprocess(imgs.copy(), norm=norm, truncated=True, upper=upper)
    for idx, img in enumerate(imgs):
        X = imgs[idx].copy().reshape(1, -1)
        img = normalize(X, norm=norm).reshape(imgs[idx].shape[0], -1)
        img = (img * 255).astype(np.uint8)
        # 将原CT slice归一化后再转为0~255的灰度图像，再统计灰度值的分布，用直方图展示
        # grayHist = caleGrayHist(img)
        # 画出直方图
        # show_gray_hist(grayHist)

        # 对图像进行gamma变换
        # gamma = 1.5
        # gamma_output = np.power(img, gamma)
        # img_norm = gamma_output

        # 进行线性变换
        linear_out = cv2.normalize(img.copy(), dst=None, alpha=100, beta=10, norm_type=cv2.NORM_MINMAX)

        img_norm = linear_out

        # 对图像进行预处理
        masks = mask_body(img.copy(), thresh=10)
        img_processed = np.zeros_like(img)
        if len(masks) != 0:
            for mask in masks:
                img_processed += np.where(mask == 1, img, mask)
        # img = preprocess(img_norm, norm=norm, truncated=False, upper=upper)[0]

        win_name1 = f"Input-NO.{idx}"
        win_name2 = f"Output-NO.{idx}"
        win_name3 = f"NO.{idx}-slice preprocessing"
        cv2.imshow(win_name1, imgs[idx])
        cv2.imshow(win_name2, img_norm)
        cv2.imshow(win_name3, img)

        cv2.moveWindow(win_name1, x=0, y=0)
        cv2.moveWindow(win_name2, x=400, y=0)
        cv2.moveWindow(win_name3, x=850, y=0)

        cv2.waitKey()
        cv2.destroyAllWindows()




