from collections.abc import Iterable
from functools import partial
from math import ceil
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from PIL import Image
from scipy.ndimage import shift
from skimage.morphology import square, disk, dilation, erosion
from skimage.transform import rescale
from skimage.util import img_as_ubyte
from torchvision.transforms import Resize

from aug.configuration import AugmentationMethod


class PadSequence(object):

    def __init__(self, length: int, padwith: int = 0):
        self.length = length
        self.padwith = padwith

    def __call__(self, sequence: torch.Tensor):
        sequenceLength = sequence.shape[0]
        if sequenceLength == self.length:
            return sequence
        targetLength = self.length - sequenceLength
        return F.pad(sequence, pad=(0, targetLength), mode="constant", value=self.padwith)


class ResizeToHeight(Resize):

    def __init__(self, size: int):
        super().__init__(size)
        if isinstance(size, Tuple):
            self.height = size[0]
        else:
            self.height = size

    def forward(self, img: Image):
        oldWidth, oldHeight = img.size
        if oldHeight > oldWidth:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            return tvF.resize(img, [self.height, intermediateWidth], self.interpolation, self.max_size, self.antialias)
        else:
            return super().forward(img)


class ResizeAndPad(object):
    """
    Custom transformation that maintains the image's original aspect ratio by scaling it to the given height and padding
    it to achieve the desired width.
    """

    def __init__(self, height: int, width: int, padwith: int = 1):
        self.width = width
        self.height = height
        self.padwith = padwith

    def __call__(self, img: Image):
        oldWidth, oldHeight = img.size
        if oldWidth == self.width and oldHeight == self.height:
            return img
        else:

            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            if intermediateWidth > self.width:
                intermediateWidth = self.width
            resized = img.resize((intermediateWidth, self.height), resample=Image.BICUBIC)  # Image.Resampling.BICUBIC
            if img.mode == "RGB":
                padValue = (self.padwith, self.padwith, self.padwith)
            else:
                padValue = self.padwith
            preprocessed = Image.new(img.mode, (self.width, self.height), padValue)
            preprocessed.paste(resized)
            return preprocessed

    @classmethod
    def invert(cls, image: np.ndarray, targetShape: Tuple[int, int]) -> np.ndarray:
        # resize so that the height matches, then cut off at width ...
        originalHeight, originalWidth = image.shape
        scaleFactor = targetShape[0] / originalHeight
        resized = rescale(image, scaleFactor)
        return resized[:, :targetShape[1]]

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Downscale:
    def __init__(self, scale: Union[float, List[float], Tuple[float, float]], targetHeight: int = 64,
                 padValue: int = 0):
        if isinstance(scale, Iterable):
            lower = min(scale[0], scale[1])
            upper = max(scale[0], scale[1])
            if upper > 1.0:
                raise ValueError("scale values larger than 1 are not supported")
            if lower == upper:
                self.getScaleValue = partial(__getFixedValue__, lower)
            else:
                self.getScaleValue = partial(__getRandomValue__, lower, upper)
        else:
            if scale > 1.0:
                raise ValueError("scale values larger than 1 are not supported")
            self.getScaleValue = partial(__getFixedValue__, scale)
        self.targetHeight = targetHeight
        self.padValue = padValue

    def __call__(self, img: Image):
        scaleValue = self.getScaleValue()
        if np.isclose(scaleValue, 1.0):
            return img
        img = np.array(img)
        oldHeight, _ = img.shape
        adaptedScaleValue = (self.targetHeight / oldHeight) * scaleValue
        scaled = rescale(img, adaptedScaleValue)
        scaleHeight, scaleWidth = scaled.shape
        pasted = np.ones_like(scaled, shape=(self.targetHeight, scaleWidth)) * self.padValue
        hOffset = 0
        if (self.targetHeight - scaleHeight) > 1:
            hOffset = (self.targetHeight - scaleHeight) // 2
        pasted[hOffset:scaleHeight + hOffset, :scaleWidth] = scaled
        return Image.fromarray(img_as_ubyte(pasted))

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Morph(object):

    def __init__(self, method: AugmentationMethod, seShape: str = "square",
                 size: Union[int, Tuple[int, int], List[int]] = 3):
        if method == AugmentationMethod.DILATION:
            self.method = dilation
        if method == AugmentationMethod.EROSION:
            self.method = erosion
        if seShape == "square":
            self.selem = square
        elif seShape == "disk":
            self.selem = disk

        if isinstance(size, Iterable):
            lower = min(size[0], size[1])
            upper = max(size[0], size[1])
            if lower == upper:
                self.shape = lower
                self.getSelem = self.__fixed_shape__
            else:
                self.shape = [lower, upper]
                self.getSelem = self.__random_shape__
        else:
            self.shape = size
            self.getSelem = self.__fixed_shape__

    def __random_shape__(self):
        return self.selem(np.random.randint(self.shape[0], self.shape[1]))

    def __fixed_shape__(self):
        return self.selem(self.shape)

    def __call__(self, img: Image):
        img = np.array(img)
        out = self.method(img, self.getSelem())
        return Image.fromarray(out)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.method.__name__})'


class MaskLines(object):

    def __init__(self, method: AugmentationMethod, rate: float = 0.1, maskValue: int = 0):
        self.method = method
        self.rate = rate
        self.maskValue = maskValue

    def __call__(self, img: Image):
        img = np.array(img)
        if self.method == AugmentationMethod.MASK_VERTICAL:
            _, width = img.shape
            lineCount = min(width, ceil(width * self.rate))
            img[:, np.random.choice(width, lineCount, replace=False)] = self.maskValue
        else:
            height, _ = img.shape
            lineCount = min(height, ceil(height * self.rate))
            img[np.random.choice(height, lineCount, replace=False), :] = self.maskValue
        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + f'({self.method.__name__})'


def __getFixedValue__(value):
    return value


def __getRandomValue__(lower, upper):
    return np.random.uniform(lower, upper)


class Shift():

    def __init__(self, horizontalShift: Union[int, float, Tuple[int, int], List[int]] = 0,
                 verticalShift: Union[int, float, Tuple[int, int], List[int]] = 0, fillValue: int = 0):
        self.fillValue = fillValue
        if isinstance(horizontalShift, Iterable):
            lower = min(horizontalShift[0], horizontalShift[1])
            upper = max(horizontalShift[0], horizontalShift[1])
            if lower == upper:
                self.getHorizontalValue = partial(__getFixedValue__, lower)
            else:
                self.shape = [lower, upper]
                self.getHorizontalValue = partial(__getRandomValue__, lower, upper)
        else:
            self.getHorizontalValue = partial(__getFixedValue__, horizontalShift)

        if isinstance(verticalShift, Iterable):
            lower = min(verticalShift[0], verticalShift[1])
            upper = max(verticalShift[0], verticalShift[1])
            if lower == upper:
                self.getVerticalValue = partial(__getFixedValue__, lower)
            else:
                self.shape = [lower, upper]
                self.getVerticalValue = partial(__getRandomValue__, lower, upper)
        else:
            self.getVerticalValue = partial(__getFixedValue__, verticalShift)

    def __call__(self, img):
        img = np.array(img)
        return Image.fromarray(shift(img, (self.getVerticalValue(), self.getHorizontalValue()), cval=self.fillValue))

    def __repr__(self):
        return self.__class__.__name__


class ImgAugWrapper(object):

    def __init__(self, imgAugTransform):
        self.transform = imgAugTransform

    def __call__(self, img: Image) -> Image:
        return Image.fromarray(self.transform.augment_image(np.array(img)))

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.transform.__name__})'
