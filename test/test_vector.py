from typing import Union, Type
from unittest import TestCase

from dreifus.matrix.pose_numpy import Pose
from dreifus.matrix.pose_torch import TorchPose
from dreifus.vector.vector_numpy import Vec2, Vec3, Vec4
from dreifus.vector.vector_torch import TorchVec4, TorchVec3


class VectorTest(TestCase):

    def _test_vec_2(self, cls: Type[Union[Vec2]]):
        # Empty Vector -> (0, 0)
        vec2 = cls()
        self.assertEqual(vec2.x, 0)
        self.assertEqual(vec2.y, 0)

        vec2 = cls(1, 2)
        self.assertEqual(vec2.x, 1)
        self.assertEqual(vec2.y, 2)

        vec2 = cls([1, 2])
        self.assertEqual(vec2.x, 1)
        self.assertEqual(vec2.y, 2)

        with self.assertRaises(AssertionError):
            vec2 = cls([1, 2], [3, 4])

        with self.assertRaises(ValueError):
            vec2 = cls([1, 2, 3])

        # Currently, specifying fewer arguments than vector length implicitly initalizes the remaining positions with 0
        vec2 = cls(1)
        self.assertEqual(vec2.x, 1)
        self.assertEqual(vec2.y, 0)

        # Vector arithmetics
        vec_a = cls(1, 2)
        vec_b = cls(3, 4)

        vec_c = vec_a + vec_b
        self.assertEqual(vec_c.x, 4)
        self.assertEqual(vec_c.y, 6)

        vec_c = vec_a * vec_b
        self.assertEqual(vec_c.x, 3)
        self.assertEqual(vec_c.y, 8)

        dot = vec_a @ vec_b
        self.assertEqual(dot, 11)

    def _test_vec_3(self, cls: Type[Union[Vec3, TorchVec3]]):
        # Empty Vector -> (0, 0)
        vec3 = cls()
        self.assertEqual(vec3.x, 0)
        self.assertEqual(vec3.y, 0)
        self.assertEqual(vec3.z, 0)

        vec3 = cls(1, 2, 3)
        self.assertEqual(vec3.x, 1)
        self.assertEqual(vec3.y, 2)
        self.assertEqual(vec3.z, 3)

        vec3 = cls([1, 2, 3])
        self.assertEqual(vec3.x, 1)
        self.assertEqual(vec3.y, 2)
        self.assertEqual(vec3.z, 3)

        with self.assertRaises(AssertionError):
            vec3 = cls([1, 2, 3], [3, 4, 5])

        with self.assertRaises(ValueError):
            vec3 = cls([1, 2, 3, 4])

        # Currently, specifying fewer arguments than vector length implicitly initalizes the remaining positions with 0
        vec3 = cls(1)
        self.assertEqual(vec3.x, 1)
        self.assertEqual(vec3.y, 0)
        self.assertEqual(vec3.z, 0)

        # Vector arithmetics
        vec_a = cls(1, 2, 3)
        vec_b = cls(4, 5, 6)

        vec_c = vec_a + vec_b
        self.assertEqual(vec_c.x, 5)
        self.assertEqual(vec_c.y, 7)
        self.assertEqual(vec_c.z, 9)

        vec_c = vec_a * vec_b
        self.assertEqual(vec_c.x, 4)
        self.assertEqual(vec_c.y, 10)
        self.assertEqual(vec_c.z, 18)

        dot = vec_a @ vec_b
        self.assertEqual(dot, 32)

        vec_c = vec_a.cross(vec_b)
        self.assertEqual(vec_c.x, -3)
        self.assertEqual(vec_c.y, 6)
        self.assertEqual(vec_c.x, -3)

        vec_c = vec_a.homogenize()
        self.assertEqual(vec_c.w, 1)

    def _test_vec_4(self, cls: Type[Union[Vec4, TorchVec4]]):
        vec4 = cls(1, 2, 3, 4)
        if cls == TorchVec4:
            pose = TorchPose()
        else:
            pose = Pose()

        vec = pose @ vec4
        self.assertIsInstance(vec, cls)

    def test_vec_2(self):
        self._test_vec_2(Vec2)
        # TODO: There is not TorchVec2 yet

    def test_vec_3(self):
        self._test_vec_3(Vec3)
        self._test_vec_3(TorchVec3)

    def test_vec_4(self):
        self._test_vec_4(Vec4)
        self._test_vec_4(TorchVec4)
