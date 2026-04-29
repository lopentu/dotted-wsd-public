from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def multiprocess_with_progress(
    func: Callable,
    argslist: Sequence[Sequence[Any]],
    description: str = "Processing...",
    console: Console | None = None,
    max_workers: int = 32,
    return_results: bool = True,
) -> list[Any] | None:
    """Run `func` over `argslist` in a process pool with a Rich progress bar.

    Each element of `argslist` is a tuple of positional arguments; for a
    single-arg function, pass `[(x,) for x in items]`.
    """
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total = len(argslist)
        res = []
        task_id = progress.add_task(f"[cyan]{description}", total=total)
        cpus = cpu_count()
        cpus = min(cpus if cpus else 16, max_workers)
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [executor.submit(func, *p) for p in argslist]
            for f in as_completed(futures):
                progress.update(task_id, advance=1)
                if return_results:
                    res.append(f.result())

    if return_results:
        return res
    return None
