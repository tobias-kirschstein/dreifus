from unittest import TestCase

import pyvista as pv
from elias.util import load_img
from elias.util.io import resize_img
from pyvista.examples import download_mars_jpg

from dreifus.matrix import Pose, Intrinsics
from dreifus.pyvista import add_camera_frustum, add_coordinate_axes


class PyVistaTest(TestCase):

    def test_add_camera_frustum(self):
        img_mars = resize_img(load_img(download_mars_jpg()), 1 / 4)
        W = img_mars.shape[1]
        H = img_mars.shape[0]

        pose = Pose()
        intrinsics = Intrinsics(1000, 1000, W / 2, H / 2)

        p = pv.Plotter()
        add_coordinate_axes(p, scale=0.1)
        add_camera_frustum(p, pose, intrinsics, image=img_mars / 255)
        p.show()
