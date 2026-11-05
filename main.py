"""Common entrypoint for future team training and evaluation workflows."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a team training or evaluation workflow."
    )
    parser.add_argument("--config", help="Path to the team's configuration file")
    parser.parse_args()
    parser.error("No training or evaluation workflow is registered yet.")


if __name__ == "__main__":
    main()
