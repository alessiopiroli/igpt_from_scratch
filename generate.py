import argparse

from igpt.generator.generator import Generator
from igpt.utils.misc import load_config


def main(args):
    cfg = load_config(args.cfg)
    inferer = Generator(cfg)
    inferer.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="gpt/config/gpt_config.yaml")
    args = parser.parse_args()
    main(args)
