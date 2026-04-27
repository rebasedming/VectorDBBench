"""PDXearch DuckDB extension client.

PDXearch (https://github.com/Noorts/PDXearch) is an experimental DuckDB
extension. The index is not persisted across DuckDB reopens, and concurrent
index access is unsupported. The runner spawns a fresh subprocess per phase
(load / optimize / search), so we use a file-backed DuckDB for the table and
rebuild the index inside any subprocess that needs to query.
"""

import logging
import os
from contextlib import contextmanager

import duckdb
import numpy as np
import pyarrow as pa

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import PDXearchIndexConfig

log = logging.getLogger(__name__)


class PDXearch(VectorDB):
    thread_safe: bool = False
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter, FilterOp.NumGE]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: PDXearchIndexConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "PDXearch"
        self.dim = dim
        self.case_config = db_case_config
        self.extension_path = db_config["extension_path"]
        self.db_path = db_config["db_path"]
        self.table_name = db_config["table_name"]
        self._index_name = f"{self.table_name}_pdx_idx"
        self._array_type = f"FLOAT[{self.dim}]"
        self._distance_fn = self.case_config.parse_distance_fn()

        self._conn: duckdb.DuckDBPyConnection | None = None
        self._where: str = ""
        self._search_sql: str | None = None
        self._index_built_in_process: bool = False

        if drop_old:
            for suffix in ("", ".wal"):
                p = self.db_path + suffix
                if os.path.exists(p):
                    os.remove(p)
                    log.info(f"PDXearch: removed stale db file {p}")
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._open()
            self._conn.execute(
                f"CREATE TABLE {self.table_name} (id BIGINT, embedding {self._array_type})"
            )
            log.info(f"PDXearch: created table {self.table_name} at {self.db_path}")
            self._close()

    def _open(self) -> None:
        if self._conn is not None:
            return
        self._conn = duckdb.connect(self.db_path, config={"allow_unsigned_extensions": "true"})
        self._conn.execute(f"LOAD '{self.extension_path}'")

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def init(self):
        self._open()
        try:
            yield
        finally:
            self._close()

    def _table_has_rows(self) -> bool:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0] > 0

    def _index_exists(self) -> bool:
        rows = self._conn.execute(
            "SELECT 1 FROM duckdb_indexes() WHERE index_name = ?",
            [self._index_name],
        ).fetchall()
        return bool(rows)

    def _repack_table(self) -> None:
        # PDXearch requires every row group except the last to contain exactly
        # 122_880 rows. INSERTs in small batches leave the row groups
        # misaligned. CREATE TABLE AS SELECT packs them tightly.
        tmp = f"{self.table_name}_repack"
        self._conn.execute(f"DROP TABLE IF EXISTS {tmp}")
        self._conn.execute(
            f"CREATE TABLE {tmp} AS SELECT id, embedding FROM {self.table_name} ORDER BY id"
        )
        self._conn.execute(f"DROP TABLE {self.table_name}")
        self._conn.execute(f"ALTER TABLE {tmp} RENAME TO {self.table_name}")
        self._conn.execute("CHECKPOINT")

    def _build_index(self) -> None:
        # PDXearch indexes don't survive a DuckDB reopen — the entry persists
        # in duckdb_indexes() but using it triggers an internal assertion.
        # Always drop any stale entry before rebuilding.
        self._conn.execute(f"DROP INDEX IF EXISTS {self._index_name}")
        log.info("PDXearch: repacking table to align row groups (122_880)")
        self._repack_table()
        opts = self.case_config.index_param()
        with_clause = ", ".join(
            f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}"
            for k, v in opts.items()
        )
        sql = (
            f"CREATE INDEX {self._index_name} ON {self.table_name} "
            f"USING PDXEARCH (embedding) WITH ({with_clause})"
        )
        log.info(f"PDXearch: building index — {sql}")
        self._conn.execute(sql)
        self._conn.execute("SET late_materialization_max_rows = 0")
        n_probe_search = self.case_config.search_param().get("n_probe_search")
        if n_probe_search is not None:
            self._conn.execute(f"SET pdxearch_n_probe = {int(n_probe_search)}")
        self._index_built_in_process = True
        log.info("PDXearch: index built")

    def _to_arrow_batch(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
    ) -> pa.Table:
        arr = np.ascontiguousarray(embeddings, dtype=np.float32)
        flat = pa.array(arr.reshape(-1), type=pa.float32())
        embs = pa.FixedSizeListArray.from_arrays(flat, list_size=self.dim)
        ids = pa.array(metadata, type=pa.int64())
        return pa.table({"id": ids, "embedding": embs})

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            batch = self._to_arrow_batch(embeddings, metadata)
            self._conn.register("_pdx_batch", batch)
            self._conn.execute(
                f"INSERT INTO {self.table_name} "
                f"SELECT id, embedding::{self._array_type} FROM _pdx_batch"
            )
            self._conn.unregister("_pdx_batch")
            return len(metadata), None
        except Exception as e:
            log.warning(f"PDXearch insert failed: {e}")
            return 0, e

    def optimize(self, data_size: int | None = None):
        if self._index_built_in_process:
            return
        self._build_index()

    def prepare_filter(self, filters: Filter):
        if not self._index_built_in_process and self._table_has_rows():
            log.info("PDXearch: rebuilding index in this subprocess (not persisted)")
            self._build_index()

        if filters.type == FilterOp.NonFilter:
            self._where = ""
        elif filters.type == FilterOp.NumGE:
            self._where = f"WHERE id >= {filters.int_value}"
        else:
            msg = f"PDXearch does not support filter: {filters}"
            raise ValueError(msg)
        self._search_sql = (
            f"SELECT id FROM {self.table_name} {self._where} "
            f"ORDER BY {self._distance_fn}(embedding, ?::{self._array_type}) LIMIT ?"
        )

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        rows = self._conn.execute(self._search_sql, [query, k]).fetchall()
        return [row[0] for row in rows]
