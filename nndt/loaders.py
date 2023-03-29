import torch
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.util.numpy_support import numpy_to_vtk
from pykdtree.kdtree import KDTree

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class SDTLoader():
    """
    Load signed distance tensor file.
    Args:
        filepath (str): path to the file
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_load = False
        self._sdt = None
        self._sdt_threshold_level = 0.0
        
        
        self._scale = 50
        self._ps_center = None
        self._ns_center = (0.0, 0.0,0.0)
              
        
        

    def calc_bbox(self):
        """Return the boundary box size of the object.
        Returns:
            (tuple), (tuple): boundary box: (Xmin, Xmax, Ymin), (Ymax, Zmin, Zmax)
        """

        mask_arr = self.sdt.cpu().numpy() <= self._sdt_threshold_level
        Xmin = float(np.argmax(np.any(mask_arr, axis=(1, 2))))
        Ymin = float(np.argmax(np.any(mask_arr, axis=(0, 2))))
        Zmin = float(np.argmax(np.any(mask_arr, axis=(0, 1))))

        Xmax = float(
            self.sdt.shape[0] - np.argmax(np.any(mask_arr, axis=(1, 2))[::-1])
        )
        Ymax = float(
            self.sdt.shape[1] - np.argmax(np.any(mask_arr, axis=(0, 2))[::-1])
        )
        Zmax = float(
            self.sdt.shape[2] - np.argmax(np.any(mask_arr, axis=(0, 1))[::-1])
        )

        return (Xmin, Ymin, Zmin), (Xmax, Ymax, Zmax)

    @property
    def sdt(self):
        """Return full STD data.
        Returns:
            jnp.ndarray: STD
        """

        if not self.is_load:
            self.load_data()
        return self._sdt
    
    
    @property
    def ps_center(self):
        """Return full STD data.
        Returns:
            jnp.ndarray: STD
        """
        if self._ps_center is None:
            ps_bbox = self.calc_bbox()

            _ps_center = ((ps_bbox[0][0] + ps_bbox[1][0])/2,
                (ps_bbox[0][1] + ps_bbox[1][1])/2,
                (ps_bbox[0][2] + ps_bbox[1][2])/2)  
            
            self._ps_center = _ps_center
            
        return self._ps_center
    

    def load_data(self):
        """Load data from the file"""

        self._sdt = torch.tensor(np.load(self.filepath), device=device)
        self.is_load = True

    def request(self, ps_xyz: torch.tensor) -> torch.tensor:
        """
        Calculate the distance from points to the surface
        :param ps_xyz: points in the normalized space
        :return: distances in SDT form
        """

        assert ps_xyz.ndim >= 1
        assert ps_xyz.shape[-1] == 3

        if ps_xyz.ndim == 1:
            p_array_ = ps_xyz.unsqueeze(0)
        else:
            p_array_ = ps_xyz
    
        
        p_array_ = p_array_.reshape((-1, 3)).to(device)
        
        
        req_x = p_array_[:, 0]
        req_y = p_array_[:, 1]
        req_z = p_array_[:, 2]
        

        x = torch.round(torch.clip(req_x, 0, self.sdt.shape[0] - 1)).int()
        y = torch.round(torch.clip(req_y, 0, self.sdt.shape[1] - 1)).int()
        z = torch.round(torch.clip(req_z, 0, self.sdt.shape[2] - 1)).int()

        adv_x = req_x - x
        adv_y = req_y - y
        adv_z = req_z - z

        
        
        result = self.sdt[x.long(), y.long(), z.long()]
        result = result + torch.sqrt(adv_x**2 + adv_y**2 + adv_z**2)

        ret_shape = list(ps_xyz.shape)
        
        ret_shape[-1] = 1

        result = result.reshape(ret_shape)

        return result

    def unload_data(self):
        self._sdt = None
        self.is_load = False

    def is_load(self) -> bool:
        return self.is_load
    


def calc_ret_shape(array, last_axis):
    ret_shape = list(array.shape)
    if len(ret_shape) == 1:
        ret_shape.append(last_axis)
    else:
        ret_shape[-1] = last_axis
    ret_shape = tuple(ret_shape)
    return ret_shape



class object_loader():
    def __init__(self, filepath):
        self.filepath = filepath
        self.is_loaded = False
        self._points = None
        self._colors = None
        self._kdtree = None
        self._mesh = None
        
    def load_data(self):
        self._load_colors()
        self._load_points()
        self.is_loaded = True
        
        
    def _load_colors(self):
        red = []
        green = []
        blue = []
        alpha = []

        with open(self.filepath, "r") as fl:
            for line in fl:
                if "v" in line:
                    tokens = line.split(" ")
                    if ("v" == tokens[0]) and (len(tokens) >= 7):
                        red.append(float(tokens[4].replace(",", ".")))
                        green.append(float(tokens[5].replace(",", ".")))
                        blue.append(float(tokens[6].replace(",", ".")))
                        alpha.append(1.0)

        red = torch.tensor(red, device=device)
        green = torch.tensor(green, device=device)
        blue = torch.tensor(blue, device=device)
        alpha = torch.tensor(alpha, device=device)
        
        self._colors = torch.column_stack([red, green, blue, alpha])

    
    def _load_points(self):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(self.filepath)
        reader.Update()
        _mesh = reader.GetOutput()
        self._points = torch.tensor(vtk_to_numpy(_mesh.GetPoints().GetData()), device=device)
        
        self._kdtree = KDTree(self._points.cpu().numpy())
        self._mesh = _mesh
        
    def get_data(self):
        if self.is_loaded:
            return self._points, self._colors
        else:
            self.load_data()
            return self._points, self._colors
        
        
    def xyz2ind(self, xyz): # request to kdtree
        

        assert xyz.shape[-1] == 3    
        ret_shape = calc_ret_shape(xyz, 1)
        xyz_flat = xyz.reshape((-1, 3)) # flatten to [N, 3]
        dist, ind = self._kdtree.query(xyz_flat.cpu().numpy()) # querying
    
        ind = torch.tensor(ind.astype(np.int32), device = device).view(ret_shape)

        return ind#, dist
        
        
    def xyz2rgba(self, xyz):
        """
        Convert coordinates of points to colors of the nearest vertex on the surface.
        Data transformation
        surface_xyz2rgba(ns_xyz[..,3]) -> rgba[..,4]
        :param ns_xyz: coordinates of points in normalized space
        :return: colors in RGBA format
        """
        assert xyz.shape[-1] == 3
        ret_shape = calc_ret_shape(xyz, 4)
        
        ind = self.xyz2ind(xyz)
        
        color = torch.take_along_dim(self._colors, ind.long(), dim=0)

        color = color.view(ret_shape)
        return color
    
    def save_mesh(self, filepath, name_value:dict):
        """
        Save a surface mesh to .vtp file.
        Dictionary may include data for storage. The dictionary key is an array name, the dictionary value is an array for storage.
        Data transformation
        save_mesh(filepath, {name, array})
        :param filepath: Path to the .vtp file
        :param name_value: Dictionary with name of vtk-arrays and data for the storage.
        :return:
        """
        surface = self._mesh

        for keys, values in name_value.items():
            if isinstance(values, (np.ndarray)):
                if values.ndim == 1:
                    data_ = numpy_to_vtk(
                        num_array=values, deep=True, array_type=vtk.VTK_FLOAT
                    )
                    data_.SetName(keys)
                    surface.GetPointData().AddArray(data_)
                else:
                    raise NotImplementedError
            elif values is list:
                data_ = numpy_to_vtk(
                    num_array=values, deep=True, array_type=vtk.VTK_FLOAT
                )
                data_.SetName(keys)
                surface.GetPointData().AddArray(data_)
            else:
                raise NotImplementedError

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(surface)
        writer.Update()
        writer.Write()

