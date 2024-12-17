"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Continual Learning AI Utilities."""


if __name__ == "__main__":
    main(prog_name="clai-util")  # pragma: no cover
