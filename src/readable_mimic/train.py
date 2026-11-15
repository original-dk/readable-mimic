import argparse
from .config import load_config
from .trainer import Trainer


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--resume", help="Checkpoint to resume; overrides config.resume.")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config, resume=arguments.resume)
    trainer = Trainer(config)
    if config.resume:
        trainer.load(config.resume)
    trainer.run()


if __name__ == "__main__":
    main()
