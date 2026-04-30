"""Multi-case sweep: each row is its own selectivity + tuning pair, all
results land in one JSON file under
vectordb_bench/results/PgSearch/result_<date>_<task_label>_pgsearch.json.

Per-case `rebuild` flag controls whether DROP_OLD+LOAD prepends the
search stage:
  - True  -> drop, reload, build the index, then search
  - False -> reuse whatever is already in Postgres, just search
Set rebuild=True for the first case (so the table exists), and for any
later case where a BUILD-TIME option changes (vector_bit_width is the
only one of those today). Query-time knobs (vector_cluster_probes,
vector_rerank_multiplier) don't need a rebuild -- flip them per case
freely.

Connection details (host/port/user/password/db) are CLI flags so the
script works against any pg_search-running Postgres without editing.
Edit the CASES list below to change the sweep itself.
"""

import os
import time

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.pg_search.config import (
    PgSearchConfig,
    PgSearchIndexConfig,
)
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import (
    CaseConfig,
    CaseType,
    ConcurrencySearchConfig,
    TaskConfig,
    TaskStage,
)

CONCURRENCY = ConcurrencySearchConfig(
    num_concurrency=[1],
    concurrency_duration=30,
    concurrency_timeout=3600,
)

# One row per (selectivity, tuning) combo. Edit freely.
CASES: list[dict] = [
    dict(
        case_id=CaseType.Performance768D1M,           # no filter
        rebuild=True,                                 # builds the index
        vector_cluster_probes=50,
        vector_rerank_multiplier=1.0,
        vector_bit_width=5,
    ),
    dict(
        case_id=CaseType.Performance768D1M1P,         # 1% selectivity
        rebuild=False,                                # reuse index from above
        vector_cluster_probes=50,
        vector_rerank_multiplier=1.5,
        vector_bit_width=5,
    ),
    dict(
        case_id=CaseType.Performance768D1M99P,        # 99% selectivity
        rebuild=False,
        vector_cluster_probes=150,
        vector_rerank_multiplier=2.0,
        vector_bit_width=5,
    ),
]


def stages_for(rebuild: bool) -> list[TaskStage]:
    if rebuild:
        return [TaskStage.DROP_OLD, TaskStage.LOAD, TaskStage.SEARCH_SERIAL]
    return [TaskStage.SEARCH_SERIAL]


def build_tasks(db_config: PgSearchConfig, rebuild: bool) -> list[TaskConfig]:
    tasks: list[TaskConfig] = []
    for case in CASES:
        # `--no-rebuild` (rebuild=False) globally overrides the
        # per-case rebuild flag, forcing search-only on every task.
        # Useful when the data is already loaded and you just want to
        # re-run with different per-case query knobs.
        case_rebuild = case["rebuild"] if rebuild else False
        index_config = PgSearchIndexConfig(
            vector_cluster_probes=case["vector_cluster_probes"],
            vector_rerank_multiplier=case["vector_rerank_multiplier"],
            vector_bit_width=case["vector_bit_width"],
        )
        tasks.append(
            TaskConfig(
                db=DB.PgSearch,
                db_config=db_config,
                db_case_config=index_config,
                case_config=CaseConfig(
                    case_id=case["case_id"],
                    k=100,
                    concurrency_search_config=CONCURRENCY,
                    custom_case={},
                ),
                stages=stages_for(case_rebuild),
                load_concurrency=0,
            )
        )
    return tasks


def warn_redundant_rebuilds() -> None:
    """Rebuilding mid-sweep wipes the table -- only makes sense when a
    build-time option actually changed. Flag a later rebuild that
    repeats the previous task's bit_width."""
    prev_bw = None
    for i, case in enumerate(CASES):
        bw = case["vector_bit_width"]
        if i > 0 and case["rebuild"] and bw == prev_bw:
            print(
                f"  warning: case {i} ({case['case_id'].name}) rebuilds "
                f"with vector_bit_width={bw}, same as previous case "
                f"-- wasted load+build"
            )
        prev_bw = bw


@click.command(context_settings={"show_default": True})
@click.option("--user-name", default="postgres", help="Postgres role")
@click.option(
    "--password",
    default=lambda: os.environ.get("POSTGRES_PASSWORD", ""),
    help="Postgres password (default: $POSTGRES_PASSWORD or empty)",
)
@click.option("--host", default="localhost", help="Postgres host")
@click.option("--port", type=int, default=5432, help="Postgres port")
@click.option("--db-name", default="postgres", help="Postgres database")
@click.option(
    "--task-label",
    default="selectivity_sweep",
    help="Task label -- controls the output file name "
    "(result_<date>_<task_label>_pgsearch.json) and groups all cases "
    "in this run into the same JSON.",
)
@click.option(
    "--rebuild/--no-rebuild",
    default=True,
    help="--rebuild (default) respects each case's per-case `rebuild` "
    "field in CASES. --no-rebuild forces search-only on every case "
    "(skips drop/load/build everywhere) -- requires the index to "
    "already exist. Useful when iterating on query-time knobs.",
)
def main(
    user_name: str,
    password: str,
    host: str,
    port: int,
    db_name: str,
    task_label: str,
    rebuild: bool,
) -> None:
    db_config = PgSearchConfig(
        db_label=task_label,
        user_name=SecretStr(user_name),
        password=SecretStr(password),
        host=host,
        port=port,
        db_name=db_name,
    )
    if rebuild:
        warn_redundant_rebuilds()
    benchmark_runner.run(build_tasks(db_config, rebuild), task_label=task_label)
    while benchmark_runner.has_running():
        time.sleep(1)


# `interface.py` runs the actual sweep in a ProcessPoolExecutor child
# process. On Linux/macOS that child uses the `spawn` start method,
# which re-imports this module from scratch -- so any top-level call
# to `benchmark_runner.run(...)` would recursively submit another
# batch in the child, hit the same code, recurse again, and crash
# with `_check_not_importing_main`. The `__main__` guard blocks that.
if __name__ == "__main__":
    main()
