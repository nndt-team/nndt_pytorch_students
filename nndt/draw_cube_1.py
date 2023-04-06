import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from math import sqrt

from datagen import Datagen


# Optimal
# DELTA = 10
# STEP = 20

# Slow and precise
DELTA = 5
STEP = 1

def len_points(first, second):
    return sqrt(
        (first[0] - second[0])**2 
        + (first[1] - second[1])**2
        + (first[2] - second[2])**2
        )


if __name__ == "__main__":
    file_path = "patient00/colored.obj"
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_path)
    reader.Update()
    geometry = reader.GetOutput()
    heart_points = vtk_to_numpy(geometry.GetPoints().GetData())
    
    d = Datagen(".", ["patient00"])
    d.process_data()
    first_cube = d.xyz_cube_list[0].numpy()[0]
    xyz = []
    for i in range(len(first_cube)):
        for j in range(len(first_cube[i])):
            for k in range(len(first_cube[i][j])):
                xyz.append([])
                for s in range(3):
                    xyz[-1].append(first_cube[i][j][k][s])
    cube = np.array(xyz)

    sample_heart_mask = []
    for first in heart_points:
        for second in cube[::STEP]:
            if len_points(first, second) < DELTA:
                sample_heart_mask.append(0)
                break
        else:
            sample_heart_mask.append(1)
    sample_heart_mask = np.array(sample_heart_mask)

    blue_red_heart = numpy_to_vtk(num_array=sample_heart_mask, deep=True, array_type=vtk.VTK_FLOAT)
    blue_red_heart.SetName('sample_cube')
    geometry.GetPointData().AddArray(blue_red_heart)    

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("result.vtp")
    writer.SetInputData(geometry)
    writer.Write()