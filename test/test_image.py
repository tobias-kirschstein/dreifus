from unittest import TestCase

import numpy as np
import torch

from dreifus.image import Img, ImageClass, ImageRange, ChannelConvention


class ImageTest(TestCase):

    def _get_numpy_image(self) -> np.ndarray:
        return np.arange(5 * 7 * 3).reshape((5, 7, 3)).astype(np.uint8)

    def _get_torch_image(self) -> torch.Tensor:
        return torch.arange(5 * 7 * 3).reshape((3, 5, 7)) / 255.

    def _get_normalized_torch_image(self) -> torch.Tensor:
        return torch.arange(5 * 7 * 3).reshape((3, 5, 7)) / 255. * 2 - 1

    def test_cast(self):
        # numpy -> torch
        numpy_image = self._get_numpy_image()
        image = Img.from_numpy(numpy_image)
        image = image.convert_to(image_class=ImageClass.TORCH)

        assert isinstance(image.img, torch.Tensor)
        assert image.image_class == ImageClass.TORCH
        assert image.image_range == ImageRange.UINT8
        assert image.channel_convention == ChannelConvention.CHANNEL_LAST
        assert np.all(image.img.numpy() == numpy_image)

        # torch -> numpy
        torch_image = self._get_torch_image()
        image = Img.from_torch(torch_image)
        image = image.convert_to(image_class=ImageClass.NUMPY)

        assert isinstance(image.img, np.ndarray)
        assert image.image_class == ImageClass.NUMPY
        assert image.image_range == ImageRange.FLOAT
        assert image.channel_convention == ChannelConvention.CHANNEL_FIRST
        assert np.all(image.img == torch_image.numpy())

    def test_range_conversion(self):
        # [0, 255] -> [0, 1]
        numpy_image = self._get_numpy_image()
        imageV2 = Img.from_numpy(numpy_image)
        image = imageV2.convert_to(image_range=ImageRange.FLOAT)

        assert np.all(image.img == numpy_image / 255.)
        assert image.img.min() >= 0
        assert image.img.max() <= 1

        # [0, 255] -> [-1, 1]
        image = imageV2.convert_to(image_range=ImageRange.NORMALIZED)

        assert np.all(image.img == (numpy_image / 255. * 2 - 1))
        assert image.img.min() >= -1
        assert image.img.max() <= 1

        # [0, 1] -> [0, 255]
        torch_image = self._get_torch_image()
        imageV2 = Img.from_torch(torch_image)
        image = imageV2.convert_to(image_range=ImageRange.UINT8)

        assert (image.img == torch_image * 255).all()
        assert image.img.min() >= 0
        assert image.img.max() <= 255
        assert image.img.dtype == torch.uint8

        # [0, 1] -> [-1, 1]
        image = imageV2.convert_to(image_range=ImageRange.NORMALIZED)

        assert (image.img == torch_image * 2 - 1).all()
        assert image.img.min() >= -1
        assert image.img.max() <= 1

        # [-1, 1] -> [0, 255]
        normalized_torch_image = self._get_normalized_torch_image()
        imageV2 = Img.from_normalized_torch(normalized_torch_image)
        image = imageV2.convert_to(image_range=ImageRange.UINT8)

        assert (image.img / 255. * 2 - 1 == normalized_torch_image).all()
        assert image.img.min() >= 0
        assert image.img.max() <= 255
        assert image.img.dtype == torch.uint8

    def test_channel_conversion(self):
        # CHANNEL_LAST -> CHANNEL_FIRST

        numpy_image = self._get_numpy_image()
        imageV2 = Img.from_numpy(numpy_image)
        image = imageV2.convert_to(channel_convention=ChannelConvention.CHANNEL_FIRST)

        assert image.img.shape[0] == numpy_image.shape[2]

        # CHANNEL_FIRST -> CHANNEL_LAST

        torch_image = self._get_torch_image()
        imageV2 = Img.from_torch(torch_image)
        image = imageV2.convert_to(channel_convention=ChannelConvention.CHANNEL_LAST)

        assert image.img.shape[2] == torch_image.shape[0]

    def test_full_conversions(self):
        # NUMPY -> TORCH
        numpy_image = self._get_numpy_image()
        imageV2 = Img.from_numpy(numpy_image)
        image_numpy_to_torch = imageV2.to_torch()

        assert image_numpy_to_torch.img.min() >= 0
        assert image_numpy_to_torch.img.max() <= 1
        assert image_numpy_to_torch.img.shape[0] == numpy_image.shape[2]

        # NUMPY -> TORCH_NORMALIZED
        image_numpy_to_torch_normalized = imageV2.to_normalized_torch()

        assert image_numpy_to_torch_normalized.img.min() >= -1
        assert image_numpy_to_torch_normalized.img.max() <= 1
        assert image_numpy_to_torch_normalized.img.shape[0] == numpy_image.shape[2]

        # TORCH -> NUMPY
        torch_image = self._get_torch_image()
        imageV2 = Img.from_torch(torch_image)
        image_torch_to_numpy = imageV2.to_numpy()

        assert image_torch_to_numpy.img.min() >= 0
        assert image_torch_to_numpy.img.max() <= 255
        assert image_torch_to_numpy.img.shape[2] == torch_image.shape[0]
        assert image_torch_to_numpy.img.dtype == np.uint8

        # TORCH -> TORCH_NORMALIZED
        image_torch_to_torch_normalized = imageV2.to_normalized_torch()

        assert image_torch_to_torch_normalized.img.min() >= -1
        assert image_torch_to_torch_normalized.img.max() <= 1
        assert image_torch_to_torch_normalized.img.shape == torch_image.shape

        # TORCH_NORMALIZED -> NUMPY
        torch_normalized_image = self._get_normalized_torch_image()
        imageV2 = Img.from_normalized_torch(torch_normalized_image)
        image_torch_normalized_to_numpy = imageV2.to_numpy()

        assert image_torch_normalized_to_numpy.img.min() >= 0
        assert image_torch_normalized_to_numpy.img.max() <= 255
        assert image_torch_normalized_to_numpy.img.shape[2] == torch_image.shape[0]
        assert image_torch_normalized_to_numpy.img.dtype == np.uint8

        # TORCH_NORMALIZED -> TORCH
        image_torch_normalized_to_torch = imageV2.to_torch()

        assert image_torch_normalized_to_torch.img.min() >= 0
        assert image_torch_normalized_to_torch.img.max() <= 1
        assert image_torch_normalized_to_torch.img.shape == torch_normalized_image.shape

        # NUMPY cycle
        image_cycle_numpy = image_numpy_to_torch.to_numpy()
        assert image_cycle_numpy.image_class == image_torch_normalized_to_numpy.image_class
        assert image_cycle_numpy.image_range == image_torch_normalized_to_numpy.image_range
        assert image_cycle_numpy.channel_convention == image_torch_normalized_to_numpy.channel_convention
        assert np.all(image_cycle_numpy.img == numpy_image)

        # TORCH cycle
        image_cycle_torch = image_torch_to_numpy.to_torch()
        assert image_cycle_torch.image_class == image_torch_normalized_to_torch.image_class
        assert image_cycle_torch.image_range == image_torch_normalized_to_torch.image_range
        assert image_cycle_torch.channel_convention == image_torch_normalized_to_torch.channel_convention
        assert (image_cycle_torch.img == torch_image).all()

        # TORCH_NORMALIZED cycle
        image_cycle_torch_normalized = image_torch_normalized_to_numpy.to_normalized_torch()
        assert image_cycle_torch_normalized.image_class == image_torch_to_torch_normalized.image_class
        assert image_cycle_torch_normalized.image_range == image_torch_to_torch_normalized.image_range
        assert image_cycle_torch_normalized.channel_convention == image_torch_to_torch_normalized.channel_convention
        assert (image_cycle_torch_normalized.img == torch_normalized_image).all()

    def test_guess_image_class(self):
        numpy_image = self._get_numpy_image()
        image = Img(numpy_image, ImageRange.UINT8, ChannelConvention.CHANNEL_LAST)
        assert image.image_class == ImageClass.NUMPY

        torch_image = self._get_torch_image()
        image = Img(torch_image, ImageRange.FLOAT, ChannelConvention.CHANNEL_FIRST)
        assert image.image_class == ImageClass.TORCH
