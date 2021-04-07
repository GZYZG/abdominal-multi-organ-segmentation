import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk


def nii2dcm_2(nii_path: str, dcm_path: str):
    image = sitk.ImageSeriesReader_GetGDCMSeriesIDs().ReadImage("01.dcm")  # 读取一个含有头信息的dicom格式的医学图像
    keys = image.GetMetaDataKeys()  # 获取它的头信息

    image2 = sitk.ReadImage(nii_path)  # 读取要转换格式的图像
    for key in keys:
        image2.SetMetaData(key, image.GetMetaData(key))  # 把第一张图像的头信息，插入到第二张图像
    sitk.WriteImage(image2, '*****.dcm')  # 把插入过头信息的图像输出为dicom格式


def nii2dcm(nii_path: str, dcm_path: str):
    """
    :param nii_path: .nii 文件路径，应为一个文件
    :param dcm_path: dcm 文件保存路径，应为文件夹
    """
    filedir = nii_path
    outdir = dcm_path
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    img = sitk.ReadImage(filedir)
    img = sitk.GetArrayFromImage(img)
    for i in range(img.shape[0]):
        select_img = sitk.GetImageFromArray(img[i])
        sitk.WriteImage(select_img, outdir + str(img.shape[0] - i) + '.dcm')
        plt.imshow(img[i])
        plt.show()


def viz_NII_3D_VR(nii_path):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path)
    reader.Update()

    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(reader.GetOutput())

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)

    property = vtk.vtkVolumeProperty()

    popacity = vtk.vtkPiecewiseFunction()
    popacity.AddPoint(1000, 0.0)
    popacity.AddPoint(4000, 0.68)
    popacity.AddPoint(7000, 0.83)

    color = vtk.vtkColorTransferFunction()
    color.AddHSVPoint(1000, 0.042, 0.73, 0.55)
    color.AddHSVPoint(2500, 0.042, 0.73, 0.55, 0.5, 0.92)
    color.AddHSVPoint(4000, 0.088, 0.67, 0.88)
    color.AddHSVPoint(5500, 0.088, 0.67, 0.88, 0.33, 0.45)
    color.AddHSVPoint(7000, 0.95, 0.063, 1.0)

    property.SetColor(color)
    property.SetScalarOpacity(popacity)
    property.ShadeOn()
    property.SetInterpolationTypeToLinear()
    property.SetShade(0, 1)
    property.SetDiffuse(0.9)
    property.SetAmbient(0.1)
    property.SetSpecular(0.2)
    property.SetSpecularPower(10.0)
    property.SetComponentWeight(0, 1)
    property.SetDisableGradientOpacity(1)
    property.DisableGradientOpacityOn()
    property.SetScalarOpacityUnitDistance(0.891927)

    volume.SetProperty(property)

    ren = vtk.vtkRenderer()
    ren.AddActor(volume)
    ren.SetBackground(0.1, 0.2, 0.4)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    renWin.SetSize(600, 600)
    renWin.Render()
    iren.Start()


def viz_NII_3D_SR(nii_path):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path)
    reader.Update()

    surface = vtk.vtkContourFilter()
    surface.SetInputConnection(reader.GetOutputPort())
    for i in range(13):
        surface.SetValue(i, i)

    surface_normals = vtk.vtkPolyDataNormals()
    surface_normals.SetInputConnection(surface.GetOutputPort())
    surface_normals.SetFeatureAngle(60)

    surface_mapper = vtk.vtkPolyDataMapper()
    surface_mapper.SetInputConnection(surface_normals.GetOutputPort())
    surface_mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(surface_mapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(.1, .2, .4)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    renWin.SetSize(600, 600)
    renWin.Render()
    iren.Start()
    pass


if __name__ == "__main__":
    # nii2dcm("../dataset/1.nii.gz", "../dataset/4/")
    viz_NII_3D_SR("../dataset/val/pred/organ0004.nii.gz")