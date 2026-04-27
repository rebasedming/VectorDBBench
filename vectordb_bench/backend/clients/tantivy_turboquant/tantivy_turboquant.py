"""VectorDBBench client for tantivy turboquant branch.

Wraps the local `tantivy_turboquant_bench` PyO3 module. Each phase
(load / optimize / search) opens its own Tantivy index handle since
the runner spawns a fresh subprocess per phase.

For COSINE metric we normalize on the way in (need_normalize_cosine=True)
and use Inner Product on the index — the turboquant MVP only ships L2 and
InnerProduct distances.
"""

import logging
import os
import shutil
from contextlib import contextmanager

import numpy as np
import tantivy_turboquant_bench as ttb

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import TantivyTurboquantIndexConfig

log = logging.getLogger(__name__)


class TantivyTurboquant(VectorDB):
    thread_safe: bool = False
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter, FilterOp.NumGE]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TantivyTurboquantIndexConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "TantivyTurboquant"
        self.dim = dim
        self.case_config = db_case_config
        self.index_path = db_config["index_path"]
        self.metric_str = db_case_config.parse_metric()
        self.heap_mb = db_case_config.heap_mb

        self._index: ttb.TantivyTurboquantIndex | None = None
        self._writer_open: bool = False
        self._min_id: int | None = None

        if drop_old:
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
                log.info(f"Tantivy turboquant: removed stale index dir {self.index_path}")
            os.makedirs(self.index_path, exist_ok=True)
            # Pre-create the index so the schema lands on disk.
            idx = ttb.TantivyTurboquantIndex.create(self.index_path, self.dim, self.metric_str)
            del idx

    def need_normalize_cosine(self) -> bool:
        # turboquant MVP exposes L2 and InnerProduct only; for cosine
        # we normalize and use IP.
        return self.case_config.metric_type.value == "COSINE"

    def _open_for_write(self) -> None:
        if self._index is None:
            self._index = ttb.TantivyTurboquantIndex.open(
                self.index_path, self.dim, self.metric_str
            )
        if not self._writer_open:
            self._index.writer_init(self.heap_mb)
            self._writer_open = True

    def _open_for_search(self) -> None:
        if self._index is None:
            self._index = ttb.TantivyTurboquantIndex.open(
                self.index_path, self.dim, self.metric_str
            )

    @contextmanager
    def init(self):
        # Caller decides what to do (insert vs search). We open lazily
        # so search subprocesses don't grab the writer lock.
        try:
            yield
        finally:
            if self._index is not None and self._writer_open:
                # MUST commit before this subprocess exits — Tantivy auto-
                # flushes segments to disk under heap pressure but they
                # stay orphaned until commit() updates meta.json. Without
                # this, the optimize subprocess opens an empty index.
                try:
                    self._index.commit()
                except Exception as e:
                    log.warning(f"Tantivy turboquant commit-on-exit failed: {e}")
                self._index.close_writer()
                self._writer_open = False
            self._index = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            self._open_for_write()
            arr = np.ascontiguousarray(embeddings, dtype=np.float32)
            flat = arr.reshape(-1).tolist()
            ids = [int(m) for m in metadata]
            # Don't commit per batch — Tantivy auto-flushes segments under
            # heap pressure. Committing per 100-doc batch creates 10K tiny
            # segments and 10K cluster-plugin serialize calls; commit once
            # at optimize() instead.
            self._index.insert(ids, flat)
            return len(metadata), None
        except Exception as e:
            log.warning(f"Tantivy turboquant insert failed: {e}")
            return 0, e

    def optimize(self, data_size: int | None = None):
        self._open_for_write()
        log.info("Tantivy turboquant: force-merging segments (k-means clustering)")
        self._index.optimize()
        log.info(f"Tantivy turboquant: optimize done, segment_count={self._index.segment_count()}")

    def prepare_filter(self, filters: Filter):
        self._open_for_search()
        if filters.type == FilterOp.NonFilter:
            self._min_id = None
        elif filters.type == FilterOp.NumGE:
            self._min_id = int(filters.int_value)
        else:
            msg = f"Tantivy turboquant does not support filter: {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        return self._index.search(query, k, self._min_id)
