import argparse

from igpt.trainer.trainer import Trainer
from igpt.utils.misc import load_config


def main(args):
    cfg = load_config(args.cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="igpt/config/igpt_config.yaml", help="cfg path")
    args = parser.parse_args()
    main(args)
