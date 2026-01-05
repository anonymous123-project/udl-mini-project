# main.py
import argparse
import importlib

from trainer import MNISTTrainer

import builder
import losses
import models
import acquisitions


def load_config(config_path: str):
    # bcnn_cfg_max_pred_entr.py  -> configs.config
    module_path = config_path.replace(".py", "").replace("/", ".")
    return importlib.import_module(module_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file, e.g. configs/bcnn.py")
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = MNISTTrainer(config)
    trainer()


if __name__ == "__main__":
    main()
