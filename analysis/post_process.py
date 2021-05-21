from utils.augmentation import fill_hole
import SimpleITK as sitk
import os
import numpy as np
from config.config import config

if __name__ == '__main__':
    # 处理分割结果中的空洞
    base_dir = "D:/projs/dicom/TransUNet-main/predictions/TU_Synapse512/" \
               "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs2_512"
    nii = '4_pred.nii.gz'
    nii_path = os.path.join(base_dir, nii)
    image = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(image)
    np.swapaxes(array, 0, 1)

    for dim in range(len(array.shape)):
        n = array.shape[dim]
        for i in range(n):
            slice = array.take(indices=i, axis=dim)
            # 对各个类别的空洞进行填充
            for c in range(1, 14):
                cls = (c == slice).astype(np.int8)
                if cls.sum() == 0:
                    continue
                filled = fill_hole(cls)
                slice[filled == 1] = c
            # array[i] = slice
            slice = np.expand_dims(slice, axis=dim)
            indices = np.array([[[i]]])
            np.put_along_axis(array, indices=indices, values=slice, axis=dim)

    filled_img = sitk.GetImageFromArray(array)
    filled_img.SetOrigin(image.GetOrigin())
    filled_img.SetDirection(image.GetDirection())
    filled_img.SetSpacing(image.GetSpacing())

    nii = f"{nii.replace('.nii.gz', '')}_filled.nii.gz"
    sitk.WriteImage(filled_img, os.path.join(base_dir, nii))
