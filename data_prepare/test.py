from pre_process import resampling
from config.config import config
import os
import SimpleITK as sitk


def test_resampling():
    data = "img0001.nii.gz"
    ct_path = os.path.join(config.train_dataset_dir, f"CT/{data}")
    gt_path = os.path.join(config.train_dataset_dir, f"GT/{data.replace('img', 'label')}")
    ct_image = sitk.ReadImage(ct_path)
    gt_image = sitk.ReadImage(gt_path)
    new_spacing = [*ct_image.GetSpacing()]
    new_spacing[-1] = 1
    resampled_ct_image = resampling(ct_image, label=False, new_spacing=new_spacing)
    resampled_gt_image = resampling(gt_image, label=True, new_spacing=new_spacing)

    sitk.WriteImage(resampled_ct_image, f"./resampled_{data}")
    sitk.WriteImage(resampled_gt_image, f"./resampled_{data.replace('img', 'label')}")


if __name__ == "__main__":
    test_resampling()
