"""Dispatch a YAML configuration to a team-owned workflow."""

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """YAML 파일을 읽어 설정 딕셔너리를 반환한다."""
    with Path(config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("설정 파일은 YAML 딕셔너리 형식이어야 합니다.")

    return config


def load_trainer(trainer_path: str) -> Callable[[dict[str, Any]], None]:
    """'training.train_gnn:train' 같은 경로에서 학습 함수를 불러온다."""
    if not isinstance(trainer_path, str) or trainer_path.count(":") != 1:
        raise ValueError(
            "trainer는 '모듈:함수' 형식으로 지정하세요. " "예: training.train_gnn:train"
        )

    # 예: 'training.train_gnn:train' → 모듈 'training.train_gnn', 함수 'train'
    module_name, function_name = trainer_path.split(":")

    if not module_name or not function_name.isidentifier():
        raise ValueError(f"trainer 경로를 확인하세요: {trainer_path!r}")

    trainer_module = importlib.import_module(module_name)
    trainer = getattr(trainer_module, function_name, None)

    if not callable(trainer):
        raise ValueError(f"호출 가능한 학습 함수를 찾을 수 없습니다: {trainer_path!r}")

    return trainer


def run(config_path: str | Path) -> None:
    """설정을 읽고, 지정된 trainer에 설정 전체를 전달해 실행한다."""
    config = load_config(config_path)

    if "trainer" not in config:
        raise ValueError("설정 파일에 trainer 항목을 추가하세요.")

    trainer = load_trainer(config["trainer"])
    trainer(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a team workflow from a YAML config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration"
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
