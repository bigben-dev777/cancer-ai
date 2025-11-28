import argparse
import asyncio
import copy
import json
import os
import time
from pathlib import Path

import bittensor as bt
import onnx
from dotenv import load_dotenv

from cancer_ai.base.base_miner import BaseNeuron
from cancer_ai.utils.config import add_miner_args, path_config
from cancer_ai.validator.competition_manager import COMPETITION_HANDLER_MAPPING
from cancer_ai.validator.model_run_manager import ModelRunManager
from cancer_ai.validator.models import ModelInfo
from cancer_ai.validator.utils import get_newest_competition_packages
from manager.utils.LocalDatasetManager import LocalDatasetManager

LICENSE_NOTICE = """
🔒 License Notice:
To share your model for Safe Scan competition, it must be released under the MIT license.

✅ By continuing, you confirm that your model is licensed under the MIT License,
which allows open use, modification, and distribution with attribution.

📤 Make sure your HuggingFace repository has license set to MIT.
"""


class MinerManagerCLI:
    LOG_BASE_DIR = "/home/mateo/cancer-ai/manager/logs"

    def __init__(self, config=None):

        # setting basic Bittensor objects
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = path_config(self)
        self.config.merge(base_config)
        self.config.logging.debug = True
        BaseNeuron.check_config(self.config)
        bt.logging.set_config(config=self.config.logging)

        self.code_zip_path = None

        self.wallet = None
        self.subtensor = None
        self.metagraph = None
        self.hotkey = None
        self.metadata_store = None

        current_date = time.strftime("%Y-%m-%d")
        self.log_path = os.path.join(self.__class__.LOG_BASE_DIR, f"{current_date}.log")

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """Method for injecting miner arguments to the parser."""
        add_miner_args(cls, parser)

    @staticmethod
    def is_onnx_model(model_path: str) -> bool:
        """Checks if model is an ONNX model."""
        if not os.path.exists(model_path):
            bt.logging.error("Model file does not exist")
            return False
        try:
            onnx.checker.check_model(model_path)
        except onnx.checker.ValidationError as e:
            bt.logging.warning(e)
            return False
        return True

    def log_result(self, result, model, dataset) -> None:
        log_folder = os.path.dirname(self.log_path)
        os.makedirs(log_folder, exist_ok=True)

        # Make folder writable
        os.chmod(log_folder, 0o777)

        # Read existing data
        if os.path.exists(self.log_path):
            os.chmod(self.log_path, 0o666)  # ensure file is writable
            with open(self.log_path, "r") as file:
                try:
                    data = json.load(file)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append new result
        result_final = {
            "model": model,
            "dataset": dataset,
            "result": result.model_dump(),
        }
        data.append(result_final)

        # Write back
        with open(self.log_path, "w") as file:
            json.dump(data, file, indent=4)

        print(f"Log appended at {self.log_path}")

    async def evaluate_model(self) -> None:
        bt.logging.info("Evaluate model mode")

        run_manager = ModelRunManager(
            config=self.config, model=ModelInfo(file_path=self.config.model_path)
        )

        # try:
        #     dataset_packages = await get_newest_competition_packages(self.config)
        # except Exception as e:
        #     bt.logging.error(f"Error retrieving competition packages: {e}")
        #     return

        dataset_manager = LocalDatasetManager(
            self.config, self.config.competition_id, self.config.dataset_dir
        )
        await dataset_manager.prepare_dataset()

        X_test, y_test, metadata = await dataset_manager.get_data()
        print(X_test)

        competition_handler = COMPETITION_HANDLER_MAPPING[self.config.competition_id](
            X_test=X_test, y_test=y_test, metadata=metadata, config=self.config
        )

        # Set preprocessing directory and preprocess data once
        competition_handler.set_preprocessed_data_dir(self.config.dataset_dir)
        await competition_handler.preprocess_and_serialize_data(X_test)

        y_test = competition_handler.prepare_y_pred(y_test)

        start_time = time.time()
        # Pass the preprocessed data generator instead of raw paths
        preprocessed_data_gen = competition_handler.get_preprocessed_data_generator()
        y_pred = await run_manager.run(preprocessed_data_gen)
        run_time_s = time.time() - start_time

        # print(y_pred)
        model_result = competition_handler.get_model_result(y_test, y_pred, run_time_s)

        bt.logging.info(
            f"Evalutaion results:\n{model_result.model_dump_json(indent=4)}"
        )

        self.log_result(model_result, self.config.model_path, self.config.dataset_dir)
        # Cleanup preprocessed data
        competition_handler.cleanup_preprocessed_data()

    async def main(self) -> None:
        match self.config.action:
            case "submit":
                return
            case "evaluate":
                await self.evaluate_model()
            case "upload":
                return
            case _:
                bt.logging.error(f"Unrecognized action: {self.config.action}")


if __name__ == "__main__":
    load_dotenv()
    cli_manager = MinerManagerCLI()
    asyncio.run(cli_manager.main())
