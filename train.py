# python train.py --config configs/test.yaml

import argparse

from train.configuration import load_configuration
from train.utils import get_device, set_seed

# get the arguments from input
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train one ResNet or plain network on CIFAR-10."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment's YAML config like configs/test.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the latest checkpoint instead of starting fresh.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    configuration = load_configuration(arguments.config)

    set_seed(configuration.seed)
    device = get_device()

    print(f"Run:     {configuration.name}")
    print(f"Device:  {device}")
    print(f"Resume:  {arguments.resume}")
    print("Settings:")
    print(configuration.describe())


if __name__ == "__main__":
    main()
