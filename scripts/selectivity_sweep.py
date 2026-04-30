"""Multi-case sweep: each row is its own selectivity + tuning pair, all
results land in one JSON file under
vectordb_bench/results/PgSearch/result_<date>_<task_label>_pgsearch.json.

Per-case `rebuild` flag controls whether DROP_OLD+LOAD prepends the
search stage:
  - True  -> drop, reload, build the index, then search
  - False -> reuse whatever is already in Postgres, just search
Set rebuild=True on whichever case will actually run first (see note
below) and for any later case where a BUILD-TIME option changes
(vector_bit_width is the only one of those today). Query-time knobs
(vector_cluster_probes, vector_rerank_multiplier) don't need a
rebuild -- flip them per case freely.

** Important: the Assembler reorders tasks. **

vectordb_bench's `Assembler.assemble_all` sorts within a db by
`(dataset_size, 0 if FilterOp.StrEqual else 1)` -- so any
label_percentage cases jump to the FRONT, with no-filter and
filter_rate (int) cases trailing. That means rebuild=True on a
no-filter case is wrong when label cases are in the sweep: the label
cases run first against whatever stale table is already there.

Worse, the rebuilt schema depends on the case that triggers it.
`with_scalar_labels` is derived from `case.filters.type ==
FilterOp.StrEqual`, so a rebuild on a no-filter case creates a
labels-less table -- subsequent label cases fail with `column
"label" does not exist`. A rebuild on any label_percentage case
creates the table WITH a `label VARCHAR(64)` column, and all
downstream cases (label or otherwise) work against it.

Rule of thumb: if any label_percentage case is in CASES, put
rebuild=True on a label_percentage case (not on the no-filter case).
The script asserts this at startup.

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
#
# Each case picks a filter shape. Set at most ONE of:
#   - filter_rate=<float>: synthetic int filter on the primary key
#     (`WHERE id >= filter_rate * dataset_size`). filter_rate is the
#     fraction of docs filtered OUT; 0.5 selects 50% of docs by id.
#     Useful for any rate but doesn't reflect a real categorical
#     filter pattern.
#   - label_percentage=<float>: real string filter on the `labels`
#     column (`WHERE labels = 'label_50p'`). The dataset comes
#     pre-tagged at fixed percentages -- pick one of:
#         0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5
#     Each has a pre-computed ground truth file, so recall is
#     measured against the true top-K among matching docs.
#   - neither set (or both None): the no-filter case.
CASES: list[dict] = [
    # Rebuild on a label-filter case: creates the table WITH a `label`
    # column so every other case (label or no-filter) can query it.
    dict(
        label_percentage=0.5,                         # real 50% label filter
        rebuild=True,
        vector_cluster_probes=50,
        vector_rerank_multiplier=1.0,
        vector_bit_width=5,
    ),
    dict(
        label_percentage=0.01,                        # real 1% label filter
        rebuild=False,
        vector_cluster_probes=50,
        vector_rerank_multiplier=1.0,
        vector_bit_width=5,
    ),
    # No-filter case runs against the same table; the extra `label`
    # column is harmless because the no-filter query never references
    # it.
    dict(
        rebuild=False,
        vector_cluster_probes=50,
        vector_rerank_multiplier=1.0,
        vector_bit_width=5,
    ),
]

# DatasetWithSizeType value for Cohere 1M -- threaded into the
# parameterized cases via custom_case.
_COHERE_MEDIUM_KEY = "Medium Cohere (768dim, 1M)"


def case_for(case: dict) -> tuple[CaseType, dict]:
    """Resolve (case_id, custom_case) from a CASES entry.

    Picks NewIntFilterPerformanceCase for filter_rate, or
    LabelFilterPerformanceCase for label_percentage. Both unset =
    no-filter Performance768D1M.
    """
    fr = case.get("filter_rate")
    lp = case.get("label_percentage")
    if fr is not None and lp is not None:
        raise ValueError(
            f"case has both filter_rate={fr} and label_percentage={lp}; "
            f"set at most one"
        )
    if lp is not None:
        return CaseType.LabelFilterPerformanceCase, {
            "dataset_with_size_type": _COHERE_MEDIUM_KEY,
            "label_percentage": lp,
        }
    if fr is not None:
        return CaseType.NewIntFilterPerformanceCase, {
            "dataset_with_size_type": _COHERE_MEDIUM_KEY,
            "filter_rate": fr,
        }
    return CaseType.Performance768D1M, {}


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
        case_id, custom_case = case_for(case)
        tasks.append(
            TaskConfig(
                db=DB.PgSearch,
                db_config=db_config,
                db_case_config=index_config,
                case_config=CaseConfig(
                    case_id=case_id,
                    k=100,
                    concurrency_search_config=CONCURRENCY,
                    custom_case=custom_case,
                ),
                stages=stages_for(case_rebuild),
                load_concurrency=0,
            )
        )
    return tasks


def _case_label(case: dict) -> str:
    fr = case.get("filter_rate")
    lp = case.get("label_percentage")
    if lp is not None:
        return f"label_percentage={lp}"
    if fr is not None:
        return f"filter_rate={fr}"
    return "no filter"


def warn_redundant_rebuilds() -> None:
    """Rebuilding mid-sweep wipes the table -- only makes sense when a
    build-time option actually changed. Flag a later rebuild that
    repeats the previous task's bit_width."""
    prev_bw = None
    for i, case in enumerate(CASES):
        bw = case["vector_bit_width"]
        if i > 0 and case["rebuild"] and bw == prev_bw:
            print(
                f"  warning: case {i} ({_case_label(case)}) rebuilds with "
                f"vector_bit_width={bw}, same as previous case "
                f"-- wasted load+build"
            )
        prev_bw = bw


def assert_rebuild_schema_matches() -> None:
    """When label_percentage cases are present, the rebuilt table must
    carry a `label` column -- otherwise downstream label queries fail
    with `column "label" does not exist`. The schema is set by
    whichever case actually rebuilds; if that's a no-filter or
    filter_rate case, with_scalar_labels=False and the column is
    omitted. Reject CASES that mix a no-filter rebuild with label
    cases."""
    has_label_case = any(c.get("label_percentage") is not None for c in CASES)
    if not has_label_case:
        return
    for i, case in enumerate(CASES):
        if not case["rebuild"]:
            continue
        if case.get("label_percentage") is not None:
            continue
        raise SystemExit(
            f"case {i} ({_case_label(case)}) has rebuild=True but is not a "
            f"label_percentage case. Because the runner sorts label cases to "
            f"the front, the rebuilt table would lack a `label` column and "
            f"every label_percentage case would fail with `column \"label\" "
            f"does not exist`. Move rebuild=True to a label_percentage case "
            f"in CASES."
        )


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
        assert_rebuild_schema_matches()
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
