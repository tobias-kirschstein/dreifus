from unittest import TestCase

import numpy as np
import torch

from dreifus.graphics import homogenize


class GraphicsTest(TestCase):

    def _assert_homogenized(self, points, points_hom):
        for i in range(len(points.shape)):
            if i == len(points.shape) - 1:
                self.assertEqual(points_hom.shape[i], points.shape[i] + 1)
            else:
                self.assertEqual(points_hom.shape[i], points.shape[i])
        self.assertTrue((points == points_hom[..., :-1]).all())
        self.assertTrue((points_hom[..., -1] == 1).all())
        self.assertEqual(points.__class__, points_hom.__class__)

    def test_homogenize(self):
        points = np.random.randn(3, 5, 7, 3)
        points_hom = homogenize(points)

        self._assert_homogenized(points, points_hom)

        points = np.random.randn(3, 5, 7, 2)
        points_hom = homogenize(points)

        self._assert_homogenized(points, points_hom)

        points = np.random.randn(7, 3)
        points_hom = homogenize(points)

        self._assert_homogenized(points, points_hom)

        points = torch.randn(3, 5, 7, 3)
        points_hom = homogenize(points)

        self._assert_homogenized(points, points_hom)

        points = torch.randn(3, 5, 7, 2)
        points_hom = homogenize(points)

        self._assert_homogenized(points, points_hom)

        points = torch.randn(7, 2)
        points_hom = homogenize(points)

        self._assert_homogenized(points, points_hom)


