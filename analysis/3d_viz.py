import vtk

x = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [1, 1, 1], [3, 1, 1], [3, 3, 1], [1, 3, 1]]
y = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]

# 把点添加到数据结构中
points = vtk.vtkPoints()
for i in range(8):
    points.InsertPoint(i, x[i])

# 把面添加到数据结构中
ploys = vtk.vtkCellArray()
for i in range(6):
    ploys.InsertNextCell(4, y[i])

# 为每一个点设置一个标量（用来进行颜色映射）
scalars = vtk.vtkFloatArray()
for i in range(8):
    scalars.InsertTuple1(i, i)

# 创建一个多边形结构体，并把数据填充进去
cube = vtk.vtkPolyData()
cube.SetPoints(points)
cube.SetPolys(ploys)
cube.GetPointData().SetScalars(scalars)

# 创建一个颜色映射表，若是不自己定义，系统会使用默认的
pColorTable = vtk.vtkLookupTable()
pColorTable.SetNumberOfColors(6)
pColorTable.SetTableValue(0, 1.0, 0.0, 1.0, 1.0)
pColorTable.SetTableValue(1, 0.0, 1.0, 1.0, 1.0)
pColorTable.SetTableValue(2, 1.0, 1.0, 1.0, 1.0)
pColorTable.SetTableValue(3, 1.0, 0.0, 1.0, 1.0)
pColorTable.SetTableValue(4, 0.0, 0.0, 1.0, 1.0)
pColorTable.SetTableValue(5, 1.0, 1.0, 0.0, 1.0)
# 上面是颜色手动定义6种，下面是自己定义生成的，使用一种就可以
# pColorTable.SetNumberOfColor(256)
# pColorTable.SetHueRange(1.0,0.0)
pColorTable.Build()

# 创建一个舞台和演员（类比理解哈~）
mapper = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(cube)
else:
    mapper.SetInputData(cube)
# 0指最小值，8指最大值。每个点都有自己的标量，根据标量在哪个范围然后映射到颜色表上
mapper.SetScalarRange(0, 8)
mapper.SetLookupTable(pColorTable)
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 定义相机
camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

# 定义渲染等
render = vtk.vtkRenderer()
render.AddActor(actor)
render.SetActiveCamera(camera)
render.ResetCamera()
render.SetBackground(1, 1, 1)
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(render)

# 展示
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
renWin.Render()
iren.Start()
