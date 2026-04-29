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
"""

import time
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

TASK_LABEL = "selectivity_sweep"

DB_CONFIG = PgSearchConfig(
    db_label=TASK_LABEL,
    user_name=SecretStr("mingying"),
    password=SecretStr(""),
    host="localhost",
    port=28818,
    db_name="pg_search",
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
        vector_cluster_probes=20,
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


tasks: list[TaskConfig] = []
for case in CASES:
    case_id = case["case_id"]
    rebuild = case["rebuild"]
    index_config = PgSearchIndexConfig(
        vector_cluster_probes=case["vector_cluster_probes"],
        vector_rerank_multiplier=case["vector_rerank_multiplier"],
        vector_bit_width=case["vector_bit_width"],
    )
    tasks.append(
        TaskConfig(
            db=DB.PgSearch,
            db_config=DB_CONFIG,
            db_case_config=index_config,
            case_config=CaseConfig(
                case_id=case_id,
                k=100,
                concurrency_search_config=CONCURRENCY,
                custom_case={},
            ),
            stages=stages_for(rebuild),
            load_concurrency=0,
        )
    )

# Sanity check: rebuilding mid-sweep wipes the table, which only makes
# sense when build-time options actually changed. Warn if a later
# rebuild repeats the previous task's bit_width.
prev_bw = None
for i, case in enumerate(CASES):
    bw = case["vector_bit_width"]
    if i > 0 and case["rebuild"] and bw == prev_bw:
        print(
            f"  warning: case {i} ({case['case_id'].name}) rebuilds with "
            f"vector_bit_width={bw}, same as previous case -- wasted load+build"
        )
    prev_bw = bw

benchmark_runner.run(tasks, task_label=TASK_LABEL)
while benchmark_runner.has_running():
    time.sleep(1)
