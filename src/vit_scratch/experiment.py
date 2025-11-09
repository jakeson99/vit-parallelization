import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from vit_scratch.train import train_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision Transformer training launcher.")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_app(cfg)


if __name__ == "__main__":
    main()
