import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Union, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from aug.configuration import AugmentationMode
from aug.utils.AlphabetEncoder import AlphabetEncoder


class DatasetMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    TEST_LH = 4
    TEST_OOD = 5


PAD_token = 0


class LineDataset(Dataset):

    def __init__(self, rootDir: Path, mode: DatasetMode, imageTransforms: Union[Compose, List[Compose]],
                 textTransforms: Compose, fold: int, augmentationMode: AugmentationMode = AugmentationMode.REPLACE):
        if imageTransforms:
            self.imageTransforms = imageTransforms
        else:
            self.imageTransforms = Compose([ToTensor()])

        self.augmentationMode = augmentationMode

        if textTransforms:
            self.characterTransforms = textTransforms
        else:
            self.characterTransforms = Compose([])

        self.alphabetEncoder = AlphabetEncoder()

        self.imageDir = rootDir
        self.data = []

        if mode in [DatasetMode.TRAIN, DatasetMode.VALIDATION]:
            filename = rootDir / f"clean_fold_{fold}.json"
            with filename.open("r") as inFile:
                foldData = json.load(inFile)
                if mode == DatasetMode.TRAIN:
                    self.data = foldData["train"]
                else:
                    self.data = foldData["val"]
        elif mode == DatasetMode.TEST_LH:
            with (rootDir / "clean_test_lh_lines.json").open("r") as inFile:
                self.data = json.load(inFile)
        elif mode == DatasetMode.TEST_OOD:
            with (rootDir / "clean_test_ood_lines.json").open("r") as inFile:
                self.data = json.load(inFile)
        else:
            with (rootDir / "clean_test_lh_lines.json").open("r") as inFile:
                self.data = json.load(inFile)
            with (rootDir / "clean_test_ood_lines.json").open("r") as inFile:
                self.data.extend(json.load(inFile))

    def __len__(self):
        if self.augmentationMode == AugmentationMode.DUPLICATE:
            return 2 * len(self.data)
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        augIndex = 0
        if self.augmentationMode == AugmentationMode.DUPLICATE and index >= len(self.data):
            index -= len(self.data)
            augIndex = 1
        lineData = self.data[index]
        transcription = lineData["transcription"]

        transcriptionEncoding = self.alphabetEncoder.encode(transcription)
        transcriptionEncoding = torch.tensor(transcriptionEncoding)

        length = transcriptionEncoding.shape[0]

        if self.characterTransforms:
            transcriptionEncoding = self.characterTransforms(transcriptionEncoding)

        lineImage = Image.open(self.imageDir / lineData["filename"]).convert("RGB")

        if self.augmentationMode == AugmentationMode.DUPLICATE:
            lineImage = self.imageTransforms[augIndex](lineImage)
        else:
            lineImage = self.imageTransforms(lineImage)

        return {"image_name": lineData["filename"], "image": lineImage, "transcription_plaintxt": transcription,
            "transcription": transcriptionEncoding, "t_len": length}
