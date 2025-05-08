import pickle
from typing import Type, Union
from unittest import TestCase

import numpy as np
import torch

from dreifus.matrix.intrinsics_numpy import Intrinsics
from dreifus.matrix.intrinsics_torch import TorchIntrinsics


class IntrinsicsTest(TestCase):

    def _test_intrinsics(self, cls: Type[Union[Intrinsics, TorchIntrinsics]]):
        intrinsics = cls()

        if cls == TorchIntrinsics:
            self.assertTrue((intrinsics == torch.eye(3)).all())
        else:
            self.assertTrue((intrinsics == np.eye(3)).all())

        intrinsics = cls(10, 11, 12, 13)
        self.assertEqual(intrinsics.fx, 10)
        self.assertEqual(intrinsics.fy, 11)
        self.assertEqual(intrinsics.cx, 12)
        self.assertEqual(intrinsics.cy, 13)

        intrinsics_homogenized = intrinsics.homogenize()
        self.assertTrue((intrinsics_homogenized[:3, :3] == intrinsics).all())

        if cls == TorchIntrinsics:
            matrix = torch.arange(3 * 3).reshape(3, 3).float()
        else:
            matrix = np.arange(3 * 3).reshape(3, 3)

        intrinsics_l = intrinsics @ matrix
        intrinsics_r = matrix @ intrinsics

        self.assertNotIsInstance(intrinsics_l, cls)
        self.assertNotIsInstance(intrinsics_r, cls)

    def test_intrinsics(self):
        self._test_intrinsics(Intrinsics)

    def test_torch_intrinsics(self):
        self._test_intrinsics(TorchIntrinsics)

    def test_serialize_intrinsics(self):
        intrinsics = Intrinsics(123, 456, 789, 111)
        intrinsics_serialized = pickle.dumps(intrinsics)
        intrinsics_deserialized = pickle.loads(intrinsics_serialized)

        self.assertTrue((intrinsics == intrinsics_deserialized).all())
        for k in intrinsics.__dict__.keys():
            self.assertEqual(getattr(intrinsics, k), getattr(intrinsics_deserialized, k))
        intrinsics_deserialized.rescale(512)
