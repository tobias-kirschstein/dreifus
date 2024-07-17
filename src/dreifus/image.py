from enum import Enum, auto
from math import ceil, sqrt, floor
from typing import Union, Optional, Tuple, List

import cv2
import numpy as np
import torch

ImageType = Union[torch.Tensor, np.ndarray]


class ImageRange(Enum):
    NORMALIZED = (-1, 1)  # [-1, 1]
    FLOAT = (0, 1)  # [0, 1]
    UINT8 = (0, 255)  # [0, 255]

    def min(self) -> int:
        return self.value[0]

    def max(self) -> int:
        return self.value[1]

    def convert_to(self,
                   image: ImageType,
                   target_range: 'ImageRange') -> ImageType:
        src_min = self.min()
        src_max = self.max()
        tgt_min = target_range.min()
        tgt_max = target_range.max()

        return (image - src_min) / (src_max - src_min) * (tgt_max - tgt_min) + tgt_min


class ChannelConvention(Enum):
    CHANNEL_FIRST = auto()  # [C, H, W]
    CHANNEL_LAST = auto()  # [H, W, C]

    def get_permutation(self, target_convention: 'ChannelConvention') -> Optional[Tuple[int, int, int]]:
        if self == target_convention:
            # No conversion necessary
            return None

        if self == ChannelConvention.CHANNEL_FIRST:
            if target_convention == ChannelConvention.CHANNEL_LAST:
                # [C, H, W] -> [H, W, C]
                return 1, 2, 0
        elif self == ChannelConvention.CHANNEL_LAST:
            if target_convention == ChannelConvention.CHANNEL_FIRST:
                # [H, W, C] -> [C, H, W]
                return 2, 0, 1

        raise NotImplementedError(f"Channel Conversion {self} -> {target_convention} not implemented")


class ImageClass(Enum):
    TORCH = auto()
    NUMPY = auto()

    @staticmethod
    def guess(image: ImageType) -> 'ImageClass':
        if isinstance(image, torch.Tensor):
            return ImageClass.TORCH
        elif isinstance(image, np.ndarray):
            return ImageClass.NUMPY
        else:
            raise ValueError(f"Unknown image class: {image.__class__}")

    def permute(self,
                image: ImageType,
                permutation: Tuple[int, int, int]) -> ImageType:
        if self == self.TORCH:
            return image.permute(permutation)
        elif self == self.NUMPY:
            return np.transpose(image, permutation)
        else:
            raise ValueError(f"image class {self} not implemented")

    def convert_to(self,
                   image: ImageType,
                   target_class: 'ImageClass') -> ImageType:
        if self == target_class:
            return image

        if self == self.TORCH:
            if target_class == self.NUMPY:
                return image.detach().cpu().numpy()
        elif self == self.NUMPY:
            if target_class == self.TORCH:
                return torch.tensor(image)

        raise NotImplementedError(f"Image class conversion {self} -> {target_class} not implemented")


class Img:

    def __init__(self,
                 image: ImageType,
                 image_range: ImageRange,
                 channel_convention: ChannelConvention,
                 image_class: Optional[ImageClass] = None
                 ):
        self.img = image
        self.image_range = image_range
        self.channel_convention = channel_convention
        if image_class is None:
            self.image_class = ImageClass.guess(image)
        else:
            self.image_class = image_class

    @staticmethod
    def from_normalized_torch(image: torch.Tensor) -> 'Img':
        # Assume channel first
        return Img(image, ImageRange.NORMALIZED, ChannelConvention.CHANNEL_FIRST, image_class=ImageClass.TORCH)

    @staticmethod
    def from_torch(image: torch.Tensor) -> 'Img':
        # Assume channel first and range [0, 1]
        return Img(image, ImageRange.FLOAT, ChannelConvention.CHANNEL_FIRST, image_class=ImageClass.TORCH)

    @staticmethod
    def from_numpy(image: np.ndarray) -> 'Img':
        if image.dtype in [np.uint8, np.int32]:
            return Img(image, ImageRange.UINT8, ChannelConvention.CHANNEL_LAST, image_class=ImageClass.NUMPY)
        else:
            return Img(image, ImageRange.FLOAT, ChannelConvention.CHANNEL_LAST, image_class=ImageClass.NUMPY)

    def convert_to(self,
                   image_range: Optional[ImageRange] = None,
                   channel_convention: Optional[ChannelConvention] = None,
                   image_class: Optional[ImageClass] = None,
                   inplace: bool = False) -> 'Img':

        image = self.img

        # Image Class
        if image_class is not None:
            image = self.image_class.convert_to(image, image_class)
        else:
            image_class = self.image_class

        # Image Range
        if image_range is not None:
            image = self.image_range.convert_to(image, image_range)
            if image_range == ImageRange.UINT8:
                # Only if target is uint8, we need to explicitly cast
                # In all other cases, the conversion is already done by scaling the range
                if image_class == ImageClass.TORCH:
                    image = image.round().clip(0, 255).type(torch.uint8)
                elif image_class == ImageClass.NUMPY:
                    image = np.clip(image.round(), 0, 255).astype(np.uint8)
                else:
                    raise NotImplementedError(f"uint conversion not implemented for {image_class}")
        else:
            image_range = self.image_range

        # Channel Convention
        if channel_convention is not None:
            permutation = self.channel_convention.get_permutation(channel_convention)
            if permutation is not None:
                image = image_class.permute(image, permutation)
        else:
            channel_convention = self.channel_convention

        if inplace:
            self.img = image
            self.image_range = image_range
            self.channel_convention = channel_convention
            self.image_class = image_class

        return Img(image, image_range=image_range, channel_convention=channel_convention, image_class=image_class)

    def to_numpy(self) -> 'Img':
        # to np.ndarray
        return self.convert_to(
            image_range=ImageRange.UINT8,
            channel_convention=ChannelConvention.CHANNEL_LAST,
            image_class=ImageClass.NUMPY
        )

    def to_torch(self) -> 'Img':
        return self.convert_to(
            image_range=ImageRange.FLOAT,
            channel_convention=ChannelConvention.CHANNEL_FIRST,
            image_class=ImageClass.TORCH
        )

    def to_normalized_torch(self) -> 'Img':
        return self.convert_to(
            image_range=ImageRange.NORMALIZED,
            channel_convention=ChannelConvention.CHANNEL_FIRST,
            image_class=ImageClass.TORCH
        )


def torch_to_numpy_img(torch_img: torch.Tensor) -> np.ndarray:
    return Img.from_torch(torch_img.detach().cpu()).to_numpy().img


def normalized_torch_to_numpy_img(torch_img: torch.Tensor) -> np.ndarray:
    return Img.from_normalized_torch(torch_img.detach().cpu()).to_numpy().img


def perform_alpha_blending(img: np.ndarray, bg: np.ndarray, alpha_map: Optional[np.ndarray] = None) -> np.ndarray:
    assert alpha_map is not None or img.shape[-1] == 4, 'If no alpha map is provided, image must have an alpha channel'
    assert img.dtype == np.uint8
    assert bg.dtype == np.uint8
    assert alpha_map is None or alpha_map.dtype == np.uint8

    if len(bg.shape) != len(img.shape):
        for i in range(len(img.shape) - len(bg.shape)):
            bg = np.expand_dims(bg, axis=-2)  # Spawn new axis to cover spatial dimensions

    if alpha_map is None:
        alpha_map = img[..., [-1]]
        img = img[..., :-1]
    alpha_map = alpha_map / 255
    img = img / 255
    bg = bg / 255

    img = alpha_map * img + (1 - alpha_map) * bg
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


def create_image_mosaic(images: List[np.ndarray],
                        nrows: Optional[int] = None,
                        ncols: Optional[int] = None,
                        max_width: Optional[int] = None,
                        max_height: Optional[int] = None) -> np.ndarray:
    """
    Creates a single ROWSxCOLUMNS numpy image from an image list.
    The given images are expected to have the same resolution.

    Parameters
    ----------
        images: (H x W x 3 as np.uint8 in [0, 255])
            the images to stitch together
        nrows:
            how many rows should be in the mosaic. If None, it is automatically inferred
        ncols:
            how many columns should be in the mosaic. If None, it is automatically inferred
        max_width:
            Optionally can resize the final mosaic to not exceed this width
        max_height:
            Optionally can resize the final mosaic to not exceed this height

    Returns
    -------
        A single numpy image containing all the given images in a mosaic fashion.
    """

    n_images = len(images)

    H, W, C = images[0].shape

    if nrows is None and ncols is None:
        # Attempt to create a square mosaic
        nrows = int(ceil(sqrt(n_images)))
        ncols = int(ceil(n_images / nrows))
    elif nrows is None:
        # Infer nrows from specified number of ncols
        nrows = int(ceil(n_images / ncols))
    elif ncols is None:
        # Infer ncols from specified number of nrows
        ncols = int(ceil(n_images / nrows))

    scale_factor = 1.
    if max_width is not None:
        scale_factor = min(scale_factor, max_width / (ncols * W))
    if max_height is not None:
        scale_factor = min(scale_factor, max_height / (nrows * H))

    tile_width = min(int(floor(W * scale_factor / nrows)), W)
    tile_height = min(int(floor(H * scale_factor / nrows)), H)

    tiled_image = np.ones((tile_height * nrows, tile_width * ncols, C), dtype=np.uint8) * 255
    for i, image in enumerate(images):
        row = int(floor(i / ncols))
        col = i % ncols
        resized_image = cv2.resize(image, (tile_width, tile_height))
        if len(resized_image.shape) == 2:
            resized_image = resized_image[..., None]
        tiled_image[row * tile_height: (row + 1) * tile_height,
        col * tile_width: (col + 1) * tile_width] = resized_image

    return tiled_image
