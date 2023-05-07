import logging
from typing import Union, List

from imgaug.augmenters import Dropout
from imgaug.augmenters.geometric import ShearX, ElasticTransformation
from imgaug.augmenters.imgcorruptlike import GaussianNoise, ImpulseNoise, ShotNoise, SpeckleNoise
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Grayscale, GaussianBlur, RandomApply, RandomRotation

from aug.configuration import Configuration, AugmentationMethod, AugmentationMode, Augmentation
from aug.dataset import PAD_token
from aug.utils.transforms import PadSequence, ResizeAndPad, ImgAugWrapper, Morph, MaskLines, Shift, Downscale


def composeTextTransformation(config: Configuration) -> Compose:
    if config.batchSize > 1:
        return transforms.Compose([PadSequence(length=config.transcriptionLength, padwith=PAD_token)])
    return transforms.Compose([])


def composeEvalTransformations(config: Configuration) -> Compose:
    return Compose([Grayscale(num_output_channels=1),
                       ResizeAndPad(height=config.padHeight, width=config.padWidth, padwith=config.padValue),
                       ToTensor()])


def __wrapIfRandom__(config: Configuration, transform):
    if config.augmentationMode == AugmentationMode.PARTIAL:
        return RandomApply([transform], config.augmentationRate)
    else:
        return transform


def __augToGauss__(augmentation: Augmentation):
    """
    GAUSS:{"kernel":5, "sigma":[0.1,2.0]}
    """
    if "kernel" not in augmentation.params:
        logging.getLogger("info").warning("kernel not defined for Gaussian Blur, using default - 5")
        kernel = 5
    else:
        kernel = augmentation.params["kernel"]

    if "sigma" not in augmentation.params:
        logging.getLogger("info").warning("sigma not defined for Gaussian Blur, using default - [0.1, 2.0]")
        sigma = [0.1, 2.0]
    else:
        sigma = augmentation.params["sigma"]

    return GaussianBlur(kernel, sigma=sigma)


def __augToRotation__(augmentation: Augmentation):
    """
    ROTATION:{"degrees":[-3,3]}
    """
    if "degrees" not in augmentation.params:
        logging.getLogger("info").warning("degrees not defined for Rotation, using default - [-3,3]")
        degrees = [-3, 3]
    else:
        degrees = augmentation.params["degrees"]
    return RandomRotation(degrees, expand=True)


def __augToMorph__(augmentation: Augmentation):
    """
    DILATION:{"selem":"square", "shape":3}
    EROSION:{"selem":"square", "shape":3}
    """
    if "selem" not in augmentation.params:
        logging.getLogger("info").warning(
                f"selem (structuring element) not defined for {augmentation.method.name}, using default - 'square'")
        selem = "square"
    else:
        selem = augmentation.params["selem"]
    if "shape" not in augmentation.params:
        logging.getLogger("info").warning(f"selem shape not defined for {augmentation.method.name}, using default: 3")
        shape = 3
    else:
        shape = augmentation.params["shape"]
    return Morph(augmentation.method, selem, shape)


def __augToShear__(augmentation: Augmentation):
    """
    SHEAR:{"shear":[-30,30]}
    """
    if "shear" not in augmentation.params:
        logging.getLogger("info").warning("shear degree not defined for Shear, using default - [-30,30]")
        shear = [-30, 30]
    else:
        shear = augmentation.params["shear"]
    return ImgAugWrapper(ShearX(shear, fit_output=True, backend="skimage", mode="constant"))


def __augToElastic__(augmentation: Augmentation):
    """
    ELASTIC:{"alpha":[0.0,40.0], "sigma":[4.0,8.0]}
    """
    if "alpha" not in augmentation.params:
        logging.getLogger("info").warning("alpha not defined for Elastic Transformation, using default - [0.0, 40.0]")
        alpha = [0.0, 40.0]
    else:
        alpha = augmentation.params["alpha"]
    if "sigma" not in augmentation.params:
        logging.getLogger("info").warning("sigma not defined for Elastic Transformation, using default - [4.0, 8.0]")
        sigma = [4.0, 8.0]
    else:
        sigma = augmentation.params["sigma"]
    return ImgAugWrapper(ElasticTransformation(alpha, sigma))


def __augToDropout__(augmentation: Augmentation):
    """
    DROPOUT:{"rate":[0,0.2]}
    """
    if "rate" not in augmentation.params:
        logging.getLogger("info").warning(f"rate not defined for {augmentation.method.name}, using default: [0,0.2]")
        rate = [0, 0.2]
    else:
        rate = augmentation.params["rate"]
    return ImgAugWrapper(Dropout(rate))


def __augToNoise__(augmentation: Augmentation):
    """
    NOISE:{"noiseType":"GAUSS", "severity":[1,3]}
    """

    noiseDict = {"GAUSS": GaussianNoise, "IMPULSE": ImpulseNoise, "SHOT": ShotNoise, "SPECKLE": SpeckleNoise}

    if "severity" not in augmentation.params:
        logging.getLogger("info").warning(f"severity not defined for {augmentation.method.name}, using default: (1,3)")
        severity = (1, 3)
    else:
        severity = augmentation.params["severity"]

    if "noiseType" not in augmentation.params:
        logging.getLogger("info").warning(f"noiseType not defined for {augmentation.method.name}, using default: gauss")
        noiseType = "GAUSS"
    else:
        noiseType = augmentation.params["noiseType"]
    if noiseType not in noiseDict:
        raise ValueError(f"Unknown noise type: {noiseType}")

    return ImgAugWrapper(noiseDict[noiseType.upper()](severity=severity))


def __augToShift__(augmentation: Augmentation):
    """
    SHIFT : {"horizontal":x, "vertical":y, "fillValue":z}
    """
    if "horizontal" not in augmentation.params:
        logging.getLogger("info").warning(f"horizontal not defined for {augmentation.method.name}, using default: 0")
        horizontal = 0
    else:
        horizontal = augmentation.params["horizontal"]

    if "vertical" not in augmentation.params:
        logging.getLogger("info").warning(f"vertical not defined for {augmentation.method.name}, using default: 0")
        vertical = 0
    else:
        vertical = augmentation.params["vertical"]

    if "fillValue" not in augmentation.params:
        logging.getLogger("info").warning(f"fillValue not defined for {augmentation.method.name}, using default: 0")
        fillValue = 0
    else:
        fillValue = augmentation.params["fillValue"]
    return Shift(horizontalShift=horizontal, verticalShift=vertical, fillValue=fillValue)


def __augToMask__(augmentation: Augmentation):
    """
    MASK_VERTICAL: {"rate":0.1, "maskValue":0}
    MASK_HORIZONTAL: {"rate":0.1, "maskValue":0}
    """
    if "rate" not in augmentation.params:
        logging.getLogger("info").warning(f"rate not defined for {augmentation.method.name}, using default: 0.1")
        rate = 0.1
    else:
        rate = augmentation.params["rate"]
    if "maskValue" not in augmentation.params:
        logging.getLogger("info").warning(f"maskValue not defined for {augmentation.method.name}, using default: 0")
        maskValue = 0
    else:
        maskValue = augmentation.params["maskValue"]
    return MaskLines(method=augmentation.method, rate=rate, maskValue=maskValue)


def __augToDownscale__(augmentation: Augmentation):
    """
    DOWNSCALE: {"scale":[0.95, 1], "targetHeight":64, "padValue":0}
    """
    if "scale" not in augmentation.params:
        logging.getLogger("info").warning(f"scale not defined for {augmentation.method.name}, using default: [0.95,1]")
        scale = [0.95, 1]
    else:
        scale = augmentation.params["scale"]
    if "targetHeight" not in augmentation.params:
        logging.getLogger("info").warning(f"targetHeight not defined for {augmentation.method.name}, using default: 64")
        targetHeight = 64
    else:
        targetHeight = augmentation.params["targetHeight"]
    if "padValue" not in augmentation.params:
        logging.getLogger("info").warning(f"padValue not defined for {augmentation.method.name}, using default: 0")
        padValue = 0
    else:
        padValue = augmentation.params["padValue"]
    return Downscale(scale, targetHeight, padValue)


def __toTransform__(augmentation: Augmentation):
    if augmentation.method == AugmentationMethod.GAUSS:
        return __augToGauss__(augmentation)
    if augmentation.method == AugmentationMethod.ROTATION:
        return __augToRotation__(augmentation)
    if augmentation.method in [AugmentationMethod.DILATION, AugmentationMethod.EROSION]:
        return __augToMorph__(augmentation)
    if augmentation.method == AugmentationMethod.SHEAR:
        return __augToShear__(augmentation)
    if augmentation.method == AugmentationMethod.ELASTIC:
        return __augToElastic__(augmentation)
    if augmentation.method in [AugmentationMethod.MASK_HORIZONTAL, AugmentationMethod.MASK_VERTICAL]:
        return __augToMask__(augmentation)
    if augmentation.method == AugmentationMethod.SHIFT:
        return __augToShift__(augmentation)
    if augmentation.method == AugmentationMethod.DOWNSCALE:
        return __augToDownscale__(augmentation)
    if augmentation.method == AugmentationMethod.DROPOUT:
        return __augToDropout__(augmentation)
    if augmentation.method == AugmentationMethod.NOISE:
        return __augToNoise__(augmentation)
    raise NotImplementedError(f"{augmentation.method.name} not implemented")


def composeTransformations(config: Configuration) -> Union[Compose, List[Compose]]:
    baseT = Compose([Grayscale(num_output_channels=1),
                        ResizeAndPad(height=config.padHeight, width=config.padWidth, padwith=config.padValue),
                        ToTensor()])

    if config.augmentationMode == AugmentationMode.NONE:
        return baseT

    augT = [Grayscale(num_output_channels=1)]
    for aug in config.augmentations:
        if aug.method != AugmentationMethod.NONE:
            augT.append(__wrapIfRandom__(config, __toTransform__(aug)))

    augT.extend([ResizeAndPad(height=config.padHeight, width=config.padWidth, padwith=config.padValue), ToTensor()])
    augT = Compose(augT)

    if config.augmentationMode == AugmentationMode.DUPLICATE:
        return [baseT, augT]
    else:
        return augT
