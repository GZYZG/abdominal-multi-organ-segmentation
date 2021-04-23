import numpy as np
import cv2
from scipy import ndimage, misc
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import SimpleITK as itk
import os
from config.config import config


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


def rotate(X, angle, axes=(1, 0), order=1, reshape=False, cval=0) -> np.ndarray:
    """
    对 X 进行旋转
    核心是scipy.ndimage.rotate函数，关于干函数的详解可参考：
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#scipy.ndimage.rotate
    Parameters
    ----------
    angle : number，旋转的角度，如果为正则是逆时针旋转，否则为顺时针旋转
    axes : tuple of 2 ints, optional。定义旋转平面的轴的索引，默认是前两个轴。
        当数据的维度超过三维时，需要自己设定。比如批量地对图像进行旋转
    reshape : bool，由于旋转后可能会超出原来的边界，reshape控制是否对 X 进行缩放以完整保留旋转后的 X
    cval : number，用于填充
    order : int，样条插值的阶数，取值范围是 0-5.

    Returns
    ----------
    ret_X : ndarray，经过旋转后的 X
    """

    ret_X = ndimage.rotate(X, angle, axes=axes, order=order, reshape=reshape, mode='constant', cval=cval)
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(10, 3))
    # ax1, ax2 = fig.subplots(1, 2)
    # img = misc.ascent()
    # print(img.max())
    # img_45 = ndimage.rotate(img, angle, axes=axes, reshape=reshape, mode='constant', cval=cval)
    # print(img.shape)
    # print(img_45.shape)
    # ax1.imshow(img, cmap='gray')
    # ax1.set_axis_off()
    # ax2.imshow(img_45, cmap='gray')
    # ax2.set_axis_off()
    # fig.set_tight_layout(True)
    # plt.show()

    return ret_X


def translate(X, offset, order=1, cval=0) -> np.ndarray:
    """
    对 X 进行平移
    核心是scipy.ndimage.shift，关于干函数的详解可参考：
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html#scipy.ndimage.shift
    Parameters
    ----------
    X : 待平移的数据
    offset : float or sequence，X 的各个维度上的平移的值。如果是int，则每个维度平移相同的值，否则需要为每个维度指定平移的值
    order : int，样条插值的阶数，取值范围是 0-5
    cval : number，用于填充

    Returns
    ---------
    trans_X : np.ndarray，平移后的X
    """
    trans_X = ndimage.shift(X, shift=offset, order=order, cval=cval)
    return trans_X


def test_elastic_deformation():
    # Load images
    nii_name = "img0001.nii.gz"
    slice_n = 110
    ct = itk.ReadImage(os.path.join(config.train_dataset_dir, f"CT/{nii_name}"))
    gt = itk.ReadImage(os.path.join(config.train_dataset_dir, f"GT/{nii_name.replace('img', 'label')}"))

    im = itk.GetArrayFromImage(ct)[slice_n]
    im_mask = itk.GetArrayFromImage(gt)[slice_n]

    # im = cv2.imread("./10_1.tif", -1)
    # im_mask = cv2.imread("./10_1_mask.tif", -1)
    # Draw grid lines
    draw_grid(im, 50)
    draw_grid(im_mask, 50)

    # Merge images into separate channels (shape will be (cols, rols, 2))
    im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
    # Apply transformation on image
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

    # Split image and mask
    im_t = im_merge_t[..., 0]
    im_mask_t = im_merge_t[..., 1]
    im_mask_t = im_mask_t.astype(np.uint8)
    # Display result
    plt.figure(figsize=(16, 14))
    plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
    plt.show()

    cv2.imshow('im', im_mask)
    cv2.imshow('im_mask', im_mask_t)
    cv2.waitKey()


if __name__ == "__main__":
    # test_elastic_deformation()
    X = misc.ascent()  # np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    ro_X = rotate(X, -45, reshape=False, cval=255)
    tr_X = translate(X, (10, 10), cval=255)
    print(f"{X}\n{ro_X}")
    cv2.imshow("origin", X.astype(np.uint8))
    cv2.imshow("processed", ro_X.astype(np.uint8))
    cv2.imshow("translated", tr_X.astype(np.uint8))
    cv2.waitKey()
