import SimpleITK as sitk
import pydicom
import os


def dcm2nii(dcms_path, nii_path, spacing=None):
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image2 = reader.Execute()
    # 2.将整合后的数据转为array，并获取dicom文件基本信息。将原Dicom中的关键信息保留下来
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z (-170.10000610351562, -5.599999904632568, -323.7699890136719)
    if spacing is None:
        spacing = image2.GetSpacing()  # x, y, z (0.68359375, 0.68359375, 2.5)
    direction = image2.GetDirection()  # x, y, z (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)


def dicom_analysis(path):
    data = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]
    # 读取一个dicom文件夹时，可以读取每个.dcm文件的SlicePosition来确定每张dcm在扫描数据中的位置
    slice_location = [e['SliceLocation'] for e in data]
    pass


if __name__ == '__main__':
    dcms_path = r'../dataset/4'  # dicom序列文件所在路径
    nii_path = r'../dataset/test/CT/4.nii.gz'  # 所需.nii.gz文件保存路径
    dcm2nii(dcms_path, nii_path)
    dicom_analysis(dcms_path)
