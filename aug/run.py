import json
import logging
import sys
import time
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate

from aug.configuration import getConfiguration, Configuration, AugmentationMode
from aug.dataset import LineDataset, DatasetMode
from aug.models import Gated_CNN_BGRU
from aug.utils.AlphabetEncoder import AlphabetEncoder
from aug.utils.log import initLoggers
from aug.utils.run_utils import composeTextTransformation, composeTransformations, composeEvalTransformations


class EvalMode(Enum):
    NONE = 1
    VALIDATION = 2
    TEST = 3


class Runner:

    def __init__(self, config: Configuration, evalMode: EvalMode = EvalMode.NONE, outFileName: str = "test.json"):
        self.config = config
        self.outFileName = outFileName

        self.alphabetEncoder = AlphabetEncoder()

        self.model = Gated_CNN_BGRU(self.alphabetEncoder.alphabetSize())

        if evalMode != EvalMode.NONE:
            state_dict = torch.load(self.config.outDir / self.config.testModelFileName,
                                    map_location=torch.device(config.device))
            if 'model_state_dict' in state_dict.keys():
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.config.device)

        self.loss = CTCLoss(zero_infinity=True)

        self.optimiser = AdamW(self.model.parameters(), lr=self.config.learningRate)

        imageTransform = composeTransformations(self.config)
        evalTransform = composeEvalTransformations(self.config)
        textTransform = composeTextTransformation(self.config)

        # set number of dataloader workers according to whether debug is active or not:
        numWorkers = 1
        gettrace = getattr(sys, "gettrace", None)
        if gettrace and gettrace():
            numWorkers = 0

        trainDataset = LineDataset(config.dataDir, DatasetMode.TRAIN, imageTransform, textTransform, self.config.fold,
                                   self.config.augmentationMode)
        self.trainDataloader = DataLoader(trainDataset, batch_size=self.config.batchSize, shuffle=True,
                                          num_workers=numWorkers)

        if evalMode == EvalMode.TEST:
            evalDataset = LineDataset(config.dataDir, DatasetMode.TEST, evalTransform, textTransform, self.config.fold,
                                      AugmentationMode.NONE)
        else:
            evalDataset = LineDataset(config.dataDir, DatasetMode.VALIDATION, evalTransform, textTransform,
                                      self.config.fold, AugmentationMode.NONE)
        self.evalDataloader = DataLoader(evalDataset, batch_size=self.config.batchSize, shuffle=False,
                                         num_workers=numWorkers)
        self.infoLogger = logging.getLogger("info")

        if evalMode == EvalMode.NONE:
            self.evalLogger = logging.getLogger("validation")
        elif evalMode == EvalMode.VALIDATION:
            self.evalLogger = logging.getLogger("eval_test")
        else:
            self.evalLogger = logging.getLogger("test")

        self.cerMetric = CharErrorRate()
        self.werMetric = WordErrorRate()

        self.bestValLoss = float("inf")
        self.bestValLossEpoch = 0

    def train(self):
        logger = logging.getLogger("train")
        logger.info("epoch,meanBatchLoss")
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            batchLosses = []
            epochStartTime = time.time()
            datasetsize = len(self.trainDataloader)
            for batchId, data in enumerate(self.trainDataloader):
                lineImage = data["image"].to(self.config.device)
                encodedTranscription = data["transcription"].to(self.config.device)
                plaintextTranscription = data["transcription_plaintxt"]
                predicted = self.model(lineImage)
                predicted = predicted.log_softmax(2)

                input_lengths = torch.full(size=(predicted.shape[1],), fill_value=predicted.shape[0],
                                           dtype=torch.long)

                loss = self.loss(predicted, encodedTranscription, input_lengths, data["t_len"])
                loss.backward()

                if self.config.batchSize == 1:
                    if batchId % 5 == 4 or batchId == datasetsize - 1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimiser.step()
                        self.optimiser.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimiser.step()
                    self.optimiser.zero_grad()

                batchLosses.append(loss.item())
                if epoch == 1 and batchLosses[-1] == 0.0:
                    self.infoLogger.info(f"{batchLosses[-1]}, {plaintextTranscription}")

            meanBatchLoss = np.mean(batchLosses)
            logger.info(f"{epoch},{meanBatchLoss}")
            self.infoLogger.info(
                    f"[{epoch}/{self.config.epochs}] - loss: {meanBatchLoss}, time: {time.time() - epochStartTime}")
            if epoch > 0 and self.config.modelSaveEpoch > 0 and epoch % self.config.modelSaveEpoch == 0:
                torch.save(self.model.state_dict(), self.config.outDir / Path(f'epoch_{epoch}.pth'))
                self.infoLogger.info(f'Epoch {epoch}: model saved')
            if self.config.validationEpoch > 0 and epoch % self.config.validationEpoch == 0:
                valLoss = self.validate()
                if valLoss < self.bestValLoss:
                    self.bestValLoss = valLoss
                    self.bestValLossEpoch = epoch
                    torch.save(self.model.state_dict(), self.config.outDir / Path('best_val_loss.pth'))
                    self.infoLogger.info(f'Epoch {epoch}: val loss model updated')
            if self.config.earlyStoppingEpochCount > 0:
                if epoch - self.bestValLossEpoch >= self.config.earlyStoppingEpochCount:
                    self.infoLogger.info(
                            f'No validation loss improvement in {epoch - self.bestValLossEpoch} epochs, stopping training.')
                    break

        self.infoLogger.info(f"Best Val Loss: {self.bestValLoss} ({self.bestValLossEpoch})")

    def greedyDecode(self, predicted) -> List[str]:
        ll = []
        _, max_index = torch.max(predicted, dim=2)
        for i in range(predicted.shape[1]):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())

            previous = raw_prediction[0]
            output = [previous]
            for char in raw_prediction[1:]:
                if char == output[-1]:
                    continue
                else:
                    output.append(char)

            result = self.alphabetEncoder.decode(output)
            ll.append(result)
        return ll

    def validate(self) -> float:
        batchLosses = []
        self.model.eval()
        for batchId, data in enumerate(self.evalDataloader):
            lineImage = data["image"].to(self.config.device)
            encodedTranscription = data["transcription"].to(self.config.device)

            predicted = self.model(lineImage)
            predicted = predicted.log_softmax(2)

            input_lengths = torch.full(size=(predicted.shape[1],), fill_value=predicted.shape[0], dtype=torch.long)

            loss = CTCLoss(zero_infinity=True)(predicted, encodedTranscription, input_lengths, data["t_len"])
            batchLosses.append(loss.item())

        meanBatchLoss = np.mean(batchLosses)
        self.infoLogger.info(f"{meanBatchLoss}")
        self.evalLogger.info(f"{meanBatchLoss}")
        return meanBatchLoss

    def test(self) -> None:
        self.model.eval()

        predictions = []
        expectations = []
        transliterations = []

        with torch.no_grad():
            for batchId, data in enumerate(self.evalDataloader):
                lineImage = data["image"].to(self.config.device)
                plaintextTranscription = data["transcription_plaintxt"]

                imageName = data["image_name"]

                predicted = self.model(lineImage)

                results = self.greedyDecode(predicted)

                for idx, entry in enumerate(results):
                    predictions.append(entry)
                    expectations.append(plaintextTranscription[idx])
                    transliterations.append({"predicted": entry, "expected": plaintextTranscription[idx],
                                                "image_name": imageName[idx]})

        meanCER = self.cerMetric(predictions, expectations)
        meanWER = self.werMetric(predictions, expectations)
        self.infoLogger.info(f"Mean CER: {meanCER}, Mean WER: {meanWER}")
        logging.getLogger("test").info(f"Mean CER: {meanCER}, Mean WER: {meanWER}")
        with (self.config.outDir / self.outFileName).open("w") as outFile:
            json.dump(transliterations, outFile, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")  # turn off pytorch user warnings for now

    torch.backends.cudnn.benchmark = True
    argParser = ArgumentParser()
    argParser.add_argument("-file", help="path to config-file", default="config.cfg", type=Path)
    argParser.add_argument("-section", help="section of config-file to use", default="DEFAULT")
    argParser.add_argument("-test", action="store_true", help="if set, will load config in test mode")
    args = argParser.parse_args()

    config = getConfiguration(args)

    if args.test:
        initLoggers(config, auxLoggerNames=["test"])
    else:
        initLoggers(config, auxLoggerNames=["train", "validation", "eval_test", "test"])

    if args.test:
        runner = Runner(config, EvalMode.TEST)
        runner.test()
    else:
        runner = Runner(config, EvalMode.NONE)
        logging.getLogger("info").info("Starting training ...")
        runner.train()
        logging.getLogger("info").info("Training complete, evaluating on validation set ...")
        runner = Runner(config, EvalMode.VALIDATION, outFileName="validation_results.json")
        runner.test()
        logging.getLogger("info").info("Training complete, evaluating on test set ...")
        runner = Runner(config, EvalMode.TEST, outFileName="test_results.json")
        runner.test()
