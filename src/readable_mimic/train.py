import argparse
from .config import load_config
from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    trainer = Trainer(config)
    if config.resume:
        trainer.load(config.resume)
    trainer.run()


if __name__ == "__main__":
    main()
