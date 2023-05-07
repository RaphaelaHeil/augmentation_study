from pathlib import Path

from PIL import Image
from imgaug.augmenters import Dropout
from imgaug.augmenters.geometric import ShearX, ElasticTransformation
from imgaug.augmenters.imgcorruptlike import GaussianNoise
from torchvision.transforms import Compose, Grayscale, GaussianBlur, RandomRotation

from aug.configuration import AugmentationMethod
from aug.utils.transforms import MaskLines, Shift, Downscale, ImgAugWrapper, Morph


def run():
    augmentations = {"blur_lower": GaussianBlur(5, sigma=0.1), "blur_upper": GaussianBlur(5, sigma=2.0),
        "dilation_lower": Morph(AugmentationMethod.DILATION, "square", 1),
        "dilation_upper": Morph(AugmentationMethod.DILATION, "square", 4),
        "disk_dilation_lower": Morph(AugmentationMethod.DILATION, "disk", 1),
        "disk_dilation_upper": Morph(AugmentationMethod.DILATION, "disk", 4),
        "downscale_75": Downscale(scale=0.75, padValue=128), "downscale_95": Downscale(scale=0.95, padValue=128),
        "dropout_upper": ImgAugWrapper(Dropout(p=0.2)),
        "elastic_lower_lower": ImgAugWrapper(ElasticTransformation(alpha=16, sigma=5)),
        "elastic_lower_upper": ImgAugWrapper(ElasticTransformation(alpha=16, sigma=7)),
        "elastic_upper_lower": ImgAugWrapper(ElasticTransformation(alpha=20, sigma=5)),
        "elastic_upper_upper": ImgAugWrapper(ElasticTransformation(alpha=20, sigma=7)),
        "erosion_lower": Morph(AugmentationMethod.EROSION, "square", 1),
        "erosion_upper": Morph(AugmentationMethod.EROSION, "square", 3),
        "disk_erosion_lower": Morph(AugmentationMethod.EROSION, "disk", 1),
        "disk_erosion_upper": Morph(AugmentationMethod.EROSION, "disk", 3),
        "extreme_lower": RandomRotation(degrees=[-10, -10], fill=128, expand=True),
        "extreme_upper": RandomRotation(degrees=[10, 10], fill=128, expand=True),
        "greggm": RandomRotation(degrees=[-2, -2], fill=128, expand=True),
        "greggp": RandomRotation(degrees=[2, 2], fill=128, expand=True),
        "mask_vertical": MaskLines(AugmentationMethod.MASK_VERTICAL, rate=0.1, maskValue=128),
        "mask_vertical04": MaskLines(AugmentationMethod.MASK_VERTICAL, rate=0.4, maskValue=128),
        "negative": RandomRotation(degrees=[-1.5, -1.5], fill=128, expand=True),
        "noise_lower": ImgAugWrapper(GaussianNoise(severity=1)),
        "noise_upper": ImgAugWrapper(GaussianNoise(severity=3)),
        "positive": RandomRotation(degrees=[1.5, 1.5], fill=128, expand=True),
        "flor_lower": RandomRotation(degrees=[-1.5, -1.5], fill=128, expand=True),
        "flor_upper": RandomRotation(degrees=[1.5, 1.5], fill=128, expand=True),
        "tomas_lower": RandomRotation(degrees=[-5, -5], fill=128, expand=True),
        "tomas_upper": RandomRotation(degrees=[5, 5], fill=128, expand=True),
        "shear_-30": ImgAugWrapper(ShearX(shear=-30, cval=128, fit_output=True)),
        "shear_lower": ImgAugWrapper(ShearX(shear=-5, cval=128, fit_output=True)),
        "shear_upper": ImgAugWrapper(ShearX(shear=30, cval=128, fit_output=True)),
        "shift_horizontal": Shift(horizontalShift=15, fillValue=128),
        "shift_vertical_lower": Shift(verticalShift=-3.5, fillValue=128),
        "shift_vertical_upper": Shift(verticalShift=3.5, fillValue=128),
        "shift_horizontal_vertical_lower": Shift(horizontalShift=15, verticalShift=-3.5, fillValue=128),
        "shift_horizontal_vertical_upper": Shift(horizontalShift=15, verticalShift=3.5, fillValue=128)
    }
    outPath = Path("../../tmp/aug_vis")
    outPath.mkdir(exist_ok=True, parents=True)
    lineImage = Image.open("432_44553_0009.png").convert("RGB")
    for name, aug in augmentations.items():
        augImage = Compose([Grayscale(num_output_channels=1), aug])(lineImage)
        augImage.save(outPath / f"{name}.png")


if __name__ == '__main__':
    run()
