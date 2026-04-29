import csv
from pathlib import Path
from typing import Annotated

import typer
from CwnGraph import CwnImage
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from dotted_wsd.asbc_eval.asbc_types import AsbcTaggedFile
from dotted_wsd.settings import DATA_DIR
from dotted_wsd.utils import multiprocess_with_progress

SAVE_DIR = (DATA_DIR / "dt_asbc_dataset").resolve()


# Makes any logging calls appear above the progress bar, so a new progress bar isn't printed after each log
console = Console()


logger.remove()
logger.add(
    RichHandler(console=console, show_path=False, markup=True),
    format="<green>{time}</green> <level>{message}</level>",
    enqueue=True,
)


def prepare_wsd_instances_and_save(p: Path) -> None:
    cwn = CwnImage.load("v.2022.08.01")
    save_name = f"{p.stem}-eval-prepared.csv"
    save_path = Path(SAVE_DIR / save_name).resolve()
    res = AsbcTaggedFile.from_file(p)
    gen = res.gather_instances(cwn=cwn)

    first_line = next(gen)
    fieldnames = list(first_line.keys())

    with save_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(first_line)
        for line in gen:
            writer.writerow(line)


def main(
    debug: Annotated[bool, typer.Option(help="Process only the first two files.")] = False,
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    input_dir = (DATA_DIR / "dt-asbc").resolve()
    asbc_files = sorted(input_dir.glob("*.txt"))
    if not asbc_files:
        raise FileNotFoundError(
            f"No *.txt files found under {input_dir}. Place the raw ASBC tagged files there."
        )
    if debug:
        asbc_files = asbc_files[:2]

    multiprocess_with_progress(
        prepare_wsd_instances_and_save,
        [(p,) for p in asbc_files],
        return_results=False,
    )


if __name__ == "__main__":
    typer.run(main)
