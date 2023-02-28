import os
import numpy as np
import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

print('device = ', device)

import torch.nn as nn


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
        cube: torch.Tensor, scale=torch.tensor([1.2, 1.2, 1.2]),
        rotation=torch.tensor([10.0, 10.0, 10.0]),
        shift=torch.tensor([5.0, 5.0, 5.0])
):
    assert cube.dtype == torch.float64, 'TypeError'
    assert cube.get_device() == device.index, 'DeviceError'

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
           0: 1: spacing[0] * 1j, 0: 1: spacing[1] * 1j, 0: 1: spacing[2] * 1j
           ].transpose((1, 2, 3, 0))

    return torch.tensor(scale * (cube - 0.5) + center_shift_, device=device)


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
        self._ns_center = (0.0, 0.0, 0.0)

    def calc_bbox(self) -> ((float, float, float), (float, float, float)):
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

            _ps_center = ((ps_bbox[0][0] + ps_bbox[1][0]) / 2,
                          (ps_bbox[0][1] + ps_bbox[1][1]) / 2,
                          (ps_bbox[0][2] + ps_bbox[1][2]) / 2)

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

        x = torch.round(torch.clip(req_x, 0, self.sdt.shape[0] - 1)).int()  # .astype(int)
        y = torch.round(torch.clip(req_y, 0, self.sdt.shape[1] - 1)).int()  # .astype(int)
        z = torch.round(torch.clip(req_z, 0, self.sdt.shape[2] - 1)).int()  # .astype(int)

        adv_x = req_x - x
        adv_y = req_y - y
        adv_z = req_z - z

        result = self.sdt[x.long(), y.long(), z.long()]
        result = result + torch.sqrt(adv_x ** 2 + adv_y ** 2 + adv_z ** 2)

        ret_shape = list(ps_xyz.shape)

        ret_shape[-1] = 1

        result = result.reshape(ret_shape)

        return result

    def unload_data(self):
        self._sdt = None
        self.is_load = False

    def is_load(self) -> bool:
        return self.is_load


def xyz_to_ns(xyz, ps_center, scale, ns_center):
    return ((xyz - torch.tensor(ps_center, device=device)) / scale) + torch.tensor(ns_center, device=device)


def xyz_to_ps(xyz, ps_center, scale, ns_center):
    return ((xyz - torch.tensor(ns_center, device=device)) * scale) + torch.tensor(ps_center, device=device)


def sdf_to_ps(sdf, scale):
    return sdf * scale


def sdf_to_ns(sdf, scale):
    return sdf / scale


def load_one_obj(data_folder, name, scale=False):
    for file in os.listdir(f'{data_folder}/{name}/'):
        if file.endswith('.obj'):
            trg_obj = (f'{data_folder}/{name}/{file}')

    verts, faces, aux = load_obj(trg_obj)
    color = aux[0]

    faces_idx = faces.verts_idx
    verts = verts

    if scale:
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

    mesh = Meshes(verts=[verts], faces=[faces_idx], verts_normals=[color])

    return mesh


def load_one_sdt(data_folder, name):
    for file in os.listdir(f'{data_folder}/{name}/'):
        if file.endswith('.npy'):
            trg_sdf = (f'{data_folder}/{name}/{file}')

    sdt = SDTLoader(filepath=trg_sdf)
    return sdt


class Datagen():
    def __init__(self, data_dir, data_names, train=True,
                 cube_spacing=(16, 16, 16),
                 cube_scale=1.0,
                 count=200,
                 step=154,  # not used
                 shift_sigma=0,
                 scale_range=0.03,
                 rotate_angle=0.5,
                 shift_mul=4,
                 batch=128,
                 inference=False):

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

        if inference:
            self.train = False

        if not train:
            self.cube_scale = 1.0
            self.shift_sigma = 0
            self.scale_range = 0
            self.rotate_angle = 0

    def _make_a_cube(self):
        one_cube = grid_in_cube(
            spacing=self.cube_spacing, scale=self.cube_scale, center_shift=(0.0, 0.0, 0.0)
        )

        basic_cube = basic_cube = one_cube.unsqueeze(0).repeat(self.count, 1, 1, 1, 1)

        return basic_cube

    def _get_augmentation_values(self):

        if self.train:
            scale = torch.distributions.uniform.Uniform(
                low=1.0 - self.scale_range,
                high=1.0 + self.scale_range).sample((self.count, 3))

            rotate = torch.distributions.uniform.Uniform(
                low=0 - self.rotate_angle,
                high=0 + self.rotate_angle).sample((self.count, 3))

            shift = torch.normal(1, 0.5, size=(self.count, 3)) * self.shift_sigma

        else:
            scale = torch.ones(self.count, 3)
            rotate = torch.zeros(self.count, 3)
            shift = torch.zeros(self.count, 3)

        return scale, rotate, shift

    def __len__(self):
        return len(self.data_names)

    def shuffle(self, sdt_: torch.Tensor, colors_: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        idx = torch.randperm(sdt_.shape[0])

        sdt_list = sdt_[idx, :, :, :, :]
        colors_list = colors_[idx, :]

        return sdt_, colors_

    def make_batches(self, sdt_: torch.Tensor, colors: torch.Tensor) -> (list, list):

        size = sdt_.shape[0]

        n_batches = size // self.batch
        if n_batches == 0:
            n_batches = 1

        return sdt_.chunk(n_batches, dim=0), colors.chunk(n_batches, dim=0)

    def process_data(self):

        sdt_list = []
        colors_list = []

        for name in tqdm(self.data_names):
            mesh = load_one_obj(self.data_dir, name)
            loader = load_one_sdt(self.data_dir, name)

            if self.inference:
                points = mesh.verts_list()[0]
                colors = mesh.verts_normals_list()[0]

                color_class = torch.argmax(colors[:, 0:3], dim=1)
                self.count = points.shape[0]

            else:
                points, colors = sample_points_from_meshes(mesh, self.count, return_normals=True)
                color_class = torch.argmax(colors[0][:, 0:3], dim=1)

            scale, rotate, shift = self._get_augmentation_values()

            # return points,shift
            points = xyz_to_ns(points.to(device), loader.ps_center, loader._scale, loader._ns_center)

            shift = torch.add(points, shift.to(device)).squeeze()

            basic_cube = self._make_a_cube()

            # no vmap in pytorch
            _xyz_cube = []
            for i in range(len(basic_cube)):
                _xyz_cube.append(_scale_rotate_shift(basic_cube[i], scale[i], rotate[i], shift[i]))
            _xyz_cube = torch.stack(_xyz_cube, dim=0)

            _xyz_cube = xyz_to_ps(_xyz_cube.to(device), loader.ps_center, loader._scale, loader._ns_center)

            sdt = loader.request(_xyz_cube)

            norm_sdt = sdf_to_ns(sdt, loader._scale)

            colors_list.append(color_class)
            sdt_list.append(norm_sdt)

        sdt_list = torch.cat(sdt_list)
        colors_list = torch.cat(colors_list)

        sdt_list = torch.permute(sdt_list, [0, 4, 1, 2, 3])
        colors_list = torch.nn.functional.one_hot(colors_list, num_classes=3).double().to(device)

        if self.train:
            sdt_list, colors_list = self.shuffle(sdt_list, colors_list)

        sdt_list, colors_list = self.make_batches(sdt_list, colors_list)

        return sdt_list, colors_list


def make_generators(datafolder, test_size=0.8, random_state=42, batch=128,
                    cube_spacing=(16, 16, 16), cube_scale=1.0, count=200, step=154,  # not used
                    shift_sigma=0, scale_range=0.03, rotate_angle=0.5, shift_mul=4, ):
    names = os.listdir(datafolder)

    train, test = train_test_split(names, test_size=test_size, random_state=random_state)

    train_dg = Datagen(data_dir=datafolder, data_names=train, batch=batch, cube_spacing=cube_spacing,
                       cube_scale=cube_scale,
                       count=count, step=step, shift_sigma=shift_sigma, scale_range=scale_range,
                       rotate_angle=rotate_angle, shift_mul=shift_mul)

    test_dg = Datagen(data_dir=datafolder, data_names=test, train=False, batch=batch)
    return train_dg, test_dg