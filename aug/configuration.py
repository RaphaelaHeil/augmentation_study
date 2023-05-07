import configparser
import json
import random
from argparse import Namespace
from configparser import SectionProxy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List, Any, Dict

import torch


class AugmentationMode(Enum):
    NONE = auto()
    REPLACE = auto()
    DUPLICATE = auto()
    PARTIAL = auto()

    @staticmethod
    def getByName(name: str) -> "AugmentationMode":
        """

        Args:
            name: string representation that should be converted to a AugmentationMode

        Returns:
            AugmentationMode

        Raises:
            LookupError: if the given name does not correspond to a supported augmentation mode

        """
        if name.upper() in [model.name for model in AugmentationMode]:
            return AugmentationMode[name.upper()]
        else:
            raise LookupError(f"unknown augmentation mode: {name}")


class AugmentationMethod(Enum):
    NONE = auto()
    ROTATION = auto()
    ELASTIC = auto()
    GAUSS = auto()
    SHEAR = auto()
    DILATION = auto()
    EROSION = auto()
    MASK_VERTICAL = auto()
    MASK_HORIZONTAL = auto()
    SHIFT = auto
    DOWNSCALE = auto()
    DROPOUT = auto()
    NOISE = auto()

    @staticmethod
    def getByName(name: str) -> "AugmentationMethod":
        """

        Args:
            name: string representation that should be converted to a AugmentationMethod

        Returns:
            AugmentationMethod

        Raises:
            LookupError: if the given name does not correspond to a supported augmentation method

        """
        if name.upper() in [model.name for model in AugmentationMethod]:
            return AugmentationMethod[name.upper()]
        else:
            raise LookupError(f"unknown augmentation method: {name}")


@dataclass
class Augmentation:
    method: AugmentationMethod = AugmentationMethod.NONE
    params: Dict[str, Any] = field(default_factory=dict)


class Configuration:
    """
    Holds the configuration for the current experiment.
    """

    def __init__(self, parsedConfig: SectionProxy, test: bool = False, fileSection: str = "DEFAULT",
                 filename: Path = None):
        self.parsedConfig = parsedConfig
        self.fileSection = fileSection

        if not test:
            self.outDir = Path(self.parsedConfig.get("out_dir")).resolve() / \
                          f"{fileSection}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{random.randint(0, 100000)}"
            self.parsedConfig["out_dir"] = str(self.outDir)
        else:
            self.outDir = filename

        if not test and not self.outDir.exists():
            self.outDir.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.epochs = self.getSetInt("epochs", 50)
        self.learningRate = self.getSetFloat("learning_rate", 0.001)
        self.earlyStoppingEpochCount = self.getSetInt("early_stopping_epoch_count", -1)

        self.batchSize = self.getSetInt("batch_size", 4)
        self.modelSaveEpoch = self.getSetInt("model_save_epoch", 10)
        self.validationEpoch = self.getSetInt("validation_epoch", 1)
        self.dataDir = Path(self.getSetStr("data_dir")).resolve()
        self.fold = self.getSetInt("fold", 0)

        self.transcriptionLength = self.getSetInt("transcription_length", 274)
        self.padHeight = self.getSetInt('pad_height', 64)
        self.padWidth = self.getSetInt('pad_width', 1362)
        self.padValue = self.getSetInt("pad_value", 0)

        self.testModelFileName = self.getSetStr("test_model_filename", "best_val_loss.pth")

        self.augmentationMode = AugmentationMode.getByName(self.getSetStr("aug_mode", "REPLACE"))
        if self.augmentationMode == AugmentationMode.NONE:
            self.parsedConfig["augmentation"] = "NONE"
        self.augmentations = Configuration.parseAugmentations(self.getSetStr("augmentation", "NONE"))

        if self.augmentationMode == AugmentationMode.PARTIAL:
            self.augmentationRate = self.getSetFloat("aug_rate", 0.5)

        if not test:
            configOut = self.outDir / "config.cfg"
            with configOut.open("w+") as cfile:
                parsedConfig.parser.write(cfile)

    def getSetInt(self, key: str, default: int = None):
        value = self.parsedConfig.getint(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetFloat(self, key: str, default: float = None):
        value = self.parsedConfig.getfloat(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetBoolean(self, key: str, default: bool = None):
        value = self.parsedConfig.getboolean(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetStr(self, key: str, default: str = None):
        value = self.parsedConfig.get(key, default)
        self.parsedConfig[key] = str(value)
        return value

    @staticmethod
    def parseAugmentations(configString: str) -> List[Augmentation]:
        results = []
        for augmentation in configString.split("|"):
            if augmentation.upper() == "NONE":
                results.append(Augmentation(method=AugmentationMethod.getByName("NONE"), params={}))
            else:
                augDetails = augmentation.split(":", 1)
                augmentationMethod = AugmentationMethod.getByName(augDetails[0])
                params = json.loads(augDetails[-1])
                results.append(Augmentation(method=augmentationMethod, params=params))
        return results


def getConfiguration(args: Namespace) -> Configuration:
    """
    Loads the configuration based on the given arguments ``args``.

    Relevant arguments:
        - ``file``: path to config file, default: 'config.cfg'
        - ``section``: config file section to load, default: 'DEFAULT'
        - ``test``: whether to load the config in train or test mode, default: False
    Args:
        args: arguments required to load the configuration

    Returns:
        the parsed configuration

    """
    fileSection = "DEFAULT"
    fileName = "config.cfg"
    test = False
    if "section" in args:
        fileSection = args.section
    if "file" in args:
        fileName = args.file.resolve()
    if "test" in args:
        test = args.test
    configParser = configparser.ConfigParser()
    configParser.read(fileName)

    if test:
        if len(configParser.sections()) > 0:
            parsedConfig = configParser[configParser.sections()[0]]
        else:
            parsedConfig = configParser["DEFAULT"]
    else:
        parsedConfig = configParser[fileSection]
        sections = configParser.sections()
        for s in sections:
            if s != fileSection:
                configParser.remove_section(s)
    if test:
        return Configuration(parsedConfig, fileSection=fileSection, test=test, filename=fileName.parent)
    else:
        return Configuration(parsedConfig, fileSection=fileSection, test=test)
