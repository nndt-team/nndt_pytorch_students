import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,  DataLoader
from loaders import SDTLoader, object_loader


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")
    
print('device = ', device)



# all augmentations 
def _rotation_matrix(yaw, pitch, roll):
    
    Rz = torch.tensor(
        [
            [torch.cos(yaw), -torch.sin(yaw), 0.0],
            [torch.sin(yaw), torch.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ], device=device
    )
    Ry = torch.tensor(
        [
            [torch.cos(pitch), 0, torch.sin(pitch)],
            [0, 1, 0],
            [-torch.sin(pitch), 0.0, torch.cos(pitch)],
        ], device=device
    )
    Rx = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(roll), -torch.sin(roll)],
            [0.0, torch.sin(roll), torch.cos(roll)],
        ], device=device
    )

    return Rz @ Ry @ Rx


def _scale_xyz(cube, scale):
    return scale * cube


def _rotate_xyz(cube, M):
    M = _rotation_matrix(M[0], M[1], M[2]).T.double()
    return cube @ M


def _shift_xyz(cube, shift):
    return cube + shift


def _scale_rotate_shift(
    cube : torch.Tensor, scale=torch.tensor([1.2, 1.2, 1.2]),
    rotation=torch.tensor([10.0, 10.0, 10.0]),
    shift=torch.tensor([5.0, 5.0, 5.0])
):
    
    
    cube = _scale_xyz(cube, scale.float().to(device))
    cube = _rotate_xyz(cube, rotation.float().to(device))
    cube = _shift_xyz(cube, shift.float().to(device))
    return cube

def grid_in_cube(
    spacing=(2, 2, 2), scale=2.0, center_shift=(0.0, 0.0, 0.0)
) -> np.ndarray:
    """Draw samples from the uniform grid that is defined inside a bounding box
    with the center in the `center_shift` and size of `scale`
    Parameters
    ----------
    spacing : tuple, optional
        Number of sections along X, Y, and Z axes (default is (2, 2, 2))
    scale : float, optional
        The scaling factor defines the size of the bounding box (default is 2.)
    center_shift : tuple, optional
        A tuple of ints of coordinates by which to modify the center of the cube (default is (0., 0., 0.))
    Returns
    -------
    ndarray
        3D mesh-grid with shape (spacing[0], spacing[1], spacing[2], 3)
    """

    center_shift_ = np.array(center_shift)
    cube = np.mgrid[
        0 : 1 : spacing[0] * 1j, 0 : 1 : spacing[1] * 1j, 0 : 1 : spacing[2] * 1j
    ].transpose((1, 2, 3, 0))

    return torch.tensor(scale * (cube - 0.5) + center_shift_, device=device)

	

def xyz_to_ns(xyz, ps_center, scale, ns_center):
    return ((xyz-torch.tensor(ps_center, device=device)) / scale) + torch.tensor(ns_center, device=device)

def xyz_to_ps(xyz, ps_center, scale, ns_center):
    return ((xyz-torch.tensor(ns_center, device=device)) * scale) + torch.tensor(ps_center, device=device)

def sdf_to_ps(sdf, scale):
    return sdf*scale

def sdf_to_ns(sdf, scale):
    return sdf/scale



def load_one_obj(data_folder, name, scale=False):    
    
    for file in os.listdir(f'{data_folder}/{name}/'):
        if file.endswith('.obj'):
            trg_obj = (f'{data_folder}/{name}/{file}')

    obj = object_loader(trg_obj)

    return obj



def load_one_sdt(data_folder, name):
    for file in os.listdir(f'{data_folder}/{name}/'):
        if file.endswith('.npy'):
            trg_sdf = (f'{data_folder}/{name}/{file}')

    sdt = SDTLoader(filepath=trg_sdf)
    return sdt


class Datagen():
    def __init__(self, data_dir, data_names, train=True, 
                cube_spacing = (16,16,16),
               cube_scale = 1.0,
               count = 200,
               step = 154, # not used
               shift_sigma = 0,
               scale_range = 0.03,
               rotate_angle= 0.5,
               shift_mul = 4,
                batch=128,
                inference = False):

        self.data_dir = data_dir
        self.data_names = data_names
        self.train = train
        self.cube_spacing = cube_spacing
        self.cube_scale = cube_scale
        self.count = count
        self.step = step
        self.shift_sigma = shift_sigma
        self.scale_range = scale_range
        self.rotate_angle = rotate_angle
        self.shift_mul = shift_mul
        self.inference = inference
        self.batch = batch
        self.xyz_cube_list = []
        
        
        # if inference:
        #     self.train=False
        
        if not train:
            self.cube_scale = 1.0
            self.shift_sigma = 0
            self.scale_range = 0
            self.rotate_angle = 0
            
        
        
    def _make_a_cube(self):
        one_cube = grid_in_cube(
                    spacing=self.cube_spacing, scale=self.cube_scale, center_shift=(0.0, 0.0, 0.0)
                )
        
        basic_cube = basic_cube = one_cube.unsqueeze(0).repeat(self.count, 1,1,1,1)
        
        return basic_cube
    
    def _get_augmentation_values(self):

        if self.train:
            if self.scale_range == 0:
                scale = torch.ones(self.count, 3)
            else: 
                scale = torch.distributions.uniform.Uniform(
                    low=1.0 - self.scale_range,
                    high=1.0 + self.scale_range).sample((self.count,3))
            if self.rotate_angle ==0:
                rotate = torch.zeros(self.count,3)
            else:
                rotate = torch.distributions.uniform.Uniform(
                    low=0 - self.rotate_angle,
                    high=0 + self.rotate_angle).sample((self.count,3))
                
            if self.shift_sigma == 0:
                shift = torch.zeros(self.count, 3)
            else:
                shift = torch.normal(1, 0.5, size=(self.count, 3)) * self.shift_sigma
        
        else:
            scale = torch.ones(self.count, 3)
            rotate = torch.zeros(self.count,3)
            shift = torch.zeros(self.count, 3)
            

        return scale, rotate, shift
    
            

    def __len__(self):
        return len(self.data_names)
    
    def shuffle(self, sdt_:torch.Tensor, colors_:torch.Tensor):

        idx = torch.randperm(sdt_.shape[0])

        sdt_list = sdt_[idx, :,:,:,:]
        colors_list = colors_[idx, :]

        return sdt_, colors_
    
    def make_batches(self, sdt_:torch.Tensor, colors:torch.Tensor ):
        
        size = sdt_.shape[0]
        
        n_batches = size//self.batch
        if n_batches == 0:
            n_batches=1
        
        return sdt_.chunk(n_batches, dim=0), colors.chunk(n_batches, dim=0)

    def sample_n_points(self, tensor):
        perm = torch.randperm(tensor.shape[0])
        idx = perm[:self.count]
        samples = tensor[idx]

        return samples
    
    
    def process_data(self):
        
        sdt_list = []
        colors_list = []
        
        for name in self.data_names:
            obj_loader = load_one_obj(self.data_dir, name)
            sdt_loader = load_one_sdt(self.data_dir, name)

            points, colors = obj_loader.get_data()
            if not self.inference:
                points = self.sample_n_points(points)
                
            else:
                self.count = points.shape[0]

            
            
            scale, rotate, shift = self._get_augmentation_values()
            
            #return points,shift
            points = xyz_to_ns(points.to(device), sdt_loader.ps_center, sdt_loader._scale, sdt_loader._ns_center)

            shift = torch.add(points, shift.to(device)).squeeze()
        
            basic_cube = self._make_a_cube()
            
            # no vmap in pytorch
            _xyz_cube = []
            for i in range(len(basic_cube)):
                _xyz_cube.append(_scale_rotate_shift(basic_cube[i], scale[i], rotate[i], shift[i]))
            _xyz_cube = torch.stack(_xyz_cube, dim=0)

            _xyz_cube = xyz_to_ps(_xyz_cube.to(device), sdt_loader.ps_center, sdt_loader._scale, sdt_loader._ns_center)
            
            self.xyz_cube_list.append(_xyz_cube)

            sdt = sdt_loader.request(_xyz_cube)

            new_points = xyz_to_ps(shift, sdt_loader.ps_center, sdt_loader._scale, sdt_loader._ns_center)
            colors = obj_loader.xyz2rgba(new_points)
            color_class = torch.argmax(colors[:,0:3], dim=1)
            
            norm_sdt = sdf_to_ns(sdt, sdt_loader._scale)
            colors_list.append(color_class)
            sdt_list.append(norm_sdt)


            
        
        sdt_list = torch.cat(sdt_list)
        colors_list = torch.cat(colors_list)

        sdt_list = torch.permute(sdt_list, [0, 4, 1,2,3])
        colors_list = torch.nn.functional.one_hot(colors_list, num_classes=3).double().to(device)
        
        if self.train:
            sdt_list, colors_list = self.shuffle(sdt_list, colors_list)
            
        sdt_list, colors_list = self.make_batches(sdt_list, colors_list)

        
        return sdt_list, colors_list


def make_generators(datafolder, test_size=0.8, random_state=42, batch=128,
                    cube_spacing = (16,16,16), cube_scale = 1.0, count = 200, step = 154, # not used
               shift_sigma = 0, scale_range = 0.03, rotate_angle= 0.5, shift_mul = 4,):
    
    names = os.listdir(datafolder)
    
    train, test = train_test_split(names, test_size=test_size, random_state=random_state)

    train_dg = Datagen(data_dir = datafolder, data_names= train, batch=batch, cube_spacing = cube_spacing, cube_scale = cube_scale,
                      count = count, step = step, shift_sigma = shift_sigma, scale_range = scale_range, rotate_angle = rotate_angle,shift_mul = shift_mul)
    
    test_dg = Datagen(data_dir = datafolder, data_names= test, train=False, batch=batch)
    return train_dg, test_dg
