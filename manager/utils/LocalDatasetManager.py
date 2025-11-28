import os
import shutil
from pathlib import Path
from typing import List, Tuple

import bittensor as bt

from cancer_ai.validator.dataset_handlers.image_csv import DatasetImagesCSV
from cancer_ai.validator.exceptions import DatasetManagerException
from cancer_ai.validator.manager import SerializableManager
from cancer_ai.validator.utils import log_time, run_command


class LocalDatasetManager(SerializableManager):
    def __init__(
        self,
        config,
        competition_id: str,
        dataset_dir: str,
    ) -> None:
        """
        Initializes a new instance of the DatasetManager class.

        Args:
            config: The configuration object.
            competition_id (str): The ID of the competition.
            dataset_hf_id (str): The Hugging Face ID of the dataset.
            file_hf_id (str): The Hugging Face ID of the file.

        Returns:
            None
        """
        self.config = config

        self.competition_id = competition_id
        self.local_extracted_dir = dataset_dir
        self.data: Tuple[List, List] = ()
        self.handler = None

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        return {}

    def set_dataset_handler(self) -> None:
        """Detect dataset type and set handler"""

        # Look for CSV file in the extracted directory or its subdirectories
        # Try common names: labels.csv, test.csv, data.csv, etc.
        csv_names = [
            "labels.csv",
            "test.csv",
            "data.csv",
            "metadata.csv",
            "dataset.csv",
        ]
        labels_csv_path = None
        dataset_root_dir = None

        # Check directly in extracted dir
        for csv_name in csv_names:
            direct_csv_path = Path(self.local_extracted_dir, csv_name)
            print("=" * 60)
            print(direct_csv_path)
            if direct_csv_path.exists():
                labels_csv_path = direct_csv_path
                dataset_root_dir = self.local_extracted_dir
                bt.logging.info(f"Found CSV file: {csv_name}")
                break

        # If not found, check in subdirectories
        if not labels_csv_path:
            for item in os.listdir(self.local_extracted_dir):
                subdir_path = Path(self.local_extracted_dir, item)
                if subdir_path.is_dir() and not item.startswith(
                    "__"
                ):  # Skip __MACOSX etc
                    for csv_name in csv_names:
                        potential_csv = Path(subdir_path, csv_name)
                        if potential_csv.exists():
                            labels_csv_path = potential_csv
                            dataset_root_dir = subdir_path
                            bt.logging.info(
                                f"Found CSV file in subdirectory {subdir_path}: {csv_name}"
                            )
                            break
                    if labels_csv_path:
                        break

        # If still not found, look for any .csv file
        if not labels_csv_path:
            bt.logging.info(
                "Specific CSV names not found, looking for any .csv file..."
            )
            # Check directly in extracted dir
            for item in os.listdir(self.local_extracted_dir):
                if item.endswith(".csv"):
                    labels_csv_path = Path(self.local_extracted_dir, item)
                    dataset_root_dir = self.local_extracted_dir
                    bt.logging.info(f"Found CSV file: {item}")
                    break

            # Check in subdirectories
            if not labels_csv_path:
                for item in os.listdir(self.local_extracted_dir):
                    subdir_path = Path(self.local_extracted_dir, item)
                    if subdir_path.is_dir() and not item.startswith("__"):
                        for subitem in os.listdir(subdir_path):
                            if subitem.endswith(".csv"):
                                labels_csv_path = Path(subdir_path, subitem)
                                dataset_root_dir = subdir_path
                                bt.logging.info(
                                    f"Found CSV file in subdirectory {subdir_path}: {subitem}"
                                )
                                break
                        if labels_csv_path:
                            break

        if labels_csv_path and dataset_root_dir:
            self.handler = DatasetImagesCSV(
                self.config,
                dataset_root_dir,
                labels_csv_path,
            )
        else:
            raise NotImplementedError(
                f"Dataset handler not implemented - no CSV file found in {self.local_extracted_dir}"
            )

    async def prepare_dataset(self) -> None:
        """Download dataset, unzip and set dataset handler"""
        self.set_dataset_handler()
        bt.logging.info(f"Preprocessing dataset '{self.competition_id}'")
        self.data = await self.handler.get_training_data()

    async def get_data(self) -> Tuple[List, List, List]:
        """Get data from dataset handler"""
        if not self.data:
            raise DatasetManagerException(
                f"Dataset '{self.competition_id}' not initalized "
            )
        # Handle backward compatibility - if data has 2 elements, add empty metadata
        if len(self.data) == 2:
            x_data, y_data = self.data
            metadata = [{"age": None, "gender": None} for _ in x_data]
            return x_data, y_data, metadata
        return self.data
