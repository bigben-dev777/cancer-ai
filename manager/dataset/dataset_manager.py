import os
import shutil
from pathlib import Path
from typing import List, Tuple

import bittensor as bt
import pandas as pd
from huggingface_hub import HfApi

from cancer_ai.validator.exceptions import DatasetManagerException
from cancer_ai.validator.manager import SerializableManager
from cancer_ai.validator.utils import log_time, run_command


class LocalDatasetImageCSV:

    def __init__(self, path: str):
        self.temp = ""
        # self.df = pd.read_csv("labels.csv")

    @staticmethod
    def get_all_csv(self, path):
        for dirpath, dirnames, filenames in os.walk(path):
            print(dirpath, dirnames, filenames)

    async def get_training_data(self):
        pass


class CompetitionLocalDatasetManager(SerializableManager):

    def __init__(
        self,
        config,
        competition_id: str,
    ) -> None:
        self.config = config
        self.competition_id = competition_id
        self.data: Tuple[List, List] = ()
        self.handler = None

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        return {}

    async def prepare_dataset(self) -> None:
        bt.logging.info(f"Setting dataset handler {self.competition_id}")
        self.handler = LocalDatasetImageCSV(path="./")
