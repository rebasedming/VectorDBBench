"""VectorDBBench client for ParadeDB pg_search BM25 index with vector ORDER BY support."""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import MetricType, VectorDB
from .config import PgSearchConfigDict, PgSearchIndexConfig


# pg_search metric strings consumed by `pdb.vector('metric=...')` typmod.
_METRIC_TO_PDB: dict[MetricType, str] = {
    MetricType.L2: "l2",
    MetricType.COSINE: "cosine",
    MetricType.IP: "ip",
}

# pgvector distance operators corresponding to each metric.
_METRIC_TO_OP: dict[MetricType, str] = {
    MetricType.L2: "<->",
    MetricType.COSINE: "<=>",
    MetricType.IP: "<#>",
}

log = logging.getLogger(__name__)


class PgSearch(VectorDB):
    """ParadeDB pg_search BM25 index over (id, embedding, label).

    Search shape:
        SELECT id FROM <table>
        WHERE id @@@ paradedb.all() [AND <filter>]
        ORDER BY embedding <op> %s::vector
        LIMIT %s

    `<op>` is `<->` (L2), `<=>` (cosine), or `<#>` (negative inner
    product), selected at runtime from the case's `metric_type` —
    see `_generate_search_query`.

    `id @@@ paradedb.all()` is the canonical "match-all" predicate that
    triggers the BM25 custom scan. Scalar filters (`label = X`,
    `label >= N`) are pushed down to the BM25 index by
    `pg_search/src/scan/filter_pushdown.rs` when the column is in the
    index's targetlist.
    """

    thread_safe: bool = False
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    conn: psycopg.Connection[Any] | None = None
    cursor: psycopg.Cursor[Any] | None = None

    _search: sql.Composed

    def __init__(
        self,
        dim: int,
        db_config: PgSearchConfigDict,
        db_case_config: PgSearchIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "PgSearch"
        self.case_config = db_case_config
        self.table_name = db_config["table_name"]
        self.connect_config = db_config["connect_config"]
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels

        self._index_name = "pg_search_index"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._scalar_label_field = "label"
        self.where_clause: str = ""

        self.conn, self.cursor = self._create_connection(**self.connect_config)

        # Required extensions: vector (for the column type + <-> operator)
        # and pg_search (for the BM25 index access method).
        log.info(f"{self.name} ensuring extensions")
        self.cursor.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector;"))
        self.cursor.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS pg_search;"))
        self.conn.commit()

        if drop_old:
            self._drop_index()
            self._drop_table()
            self._create_table(dim)

        # Index is always built after the bulk load (post-insert path).
        # CREATE INDEX on a populated table is much faster for BM25 than
        # incremental insert during loading.
        self.case_config.create_index_before_load = False
        self.case_config.create_index_after_load = True

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None
        assert cursor is not None

        return conn, cursor

    def _generate_search_query(self) -> sql.Composed:
        """Build the per-query SQL with the metric-appropriate operator."""
        metric = self.case_config.metric_type
        op = _METRIC_TO_OP.get(metric, "<->") if metric else "<->"
        return sql.SQL(
            """
            SELECT {primary_field}
            FROM public.{table_name}
            {where_clause}
            ORDER BY {vector_field} {op} %s::vector
            LIMIT %s
            """,
        ).format(
            primary_field=sql.Identifier(self._primary_field),
            table_name=sql.Identifier(self.table_name),
            vector_field=sql.Identifier(self._vector_field),
            op=sql.SQL(op),
            where_clause=sql.SQL(self.where_clause),
        )

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn, self.cursor = self._create_connection(**self.connect_config)

        # Apply per-session GUCs from case_config.session_param().
        # Mirrors the pgvector client. Currently this disables parallel
        # scan so per-query QPS reflects single-thread cost.
        session_options: Sequence[dict[str, Any]] = self.case_config.session_param()[
            "session_options"
        ]
        for setting in session_options:
            self.cursor.execute(
                sql.SQL("SET {setting_name} = {val};").format(
                    setting_name=sql.Identifier(setting["parameter"]["setting_name"]),
                    val=sql.Identifier(str(setting["parameter"]["val"])),
                ),
            )
        self.conn.commit()

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        log.info(f"{self.name} drop table {self.table_name}")
        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{tbl}").format(
                tbl=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

    def _drop_index(self):
        log.info(f"{self.name} drop index {self._index_name}")
        self.cursor.execute(
            sql.SQL("DROP INDEX IF EXISTS {idx}").format(
                idx=sql.Identifier(self._index_name),
            ),
        )
        self.conn.commit()

    def _create_table(self, dim: int):
        log.info(f"{self.name} create table {self.table_name} (with_labels={self.with_scalar_labels})")
        if self.with_scalar_labels:
            self.cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS public.{tbl} "
                    "({pk} BIGINT PRIMARY KEY, {emb} vector({dim}), {lbl} VARCHAR(64));",
                ).format(
                    tbl=sql.Identifier(self.table_name),
                    pk=sql.Identifier(self._primary_field),
                    emb=sql.Identifier(self._vector_field),
                    lbl=sql.Identifier(self._scalar_label_field),
                    dim=sql.Literal(dim),
                ),
            )
        else:
            self.cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS public.{tbl} "
                    "({pk} BIGINT PRIMARY KEY, {emb} vector({dim}));",
                ).format(
                    tbl=sql.Identifier(self.table_name),
                    pk=sql.Identifier(self._primary_field),
                    emb=sql.Identifier(self._vector_field),
                    dim=sql.Literal(dim),
                ),
            )
        self.conn.commit()

    def _embedding_index_expr(self) -> sql.Composable:
        """Indexed expression for the embedding column.

        When the case's metric_type is set, wraps the column in a
        `pdb.vector('metric=...')` cast so the underlying RaBitQ +
        clustering index gets built for that metric instead of pg_search's
        default L2.
        """
        metric = self.case_config.metric_type
        if metric is None or metric not in _METRIC_TO_PDB:
            return sql.Identifier(self._vector_field)
        metric_str = _METRIC_TO_PDB[metric]
        return sql.SQL("({col}::pdb.vector('metric={m}'))").format(
            col=sql.Identifier(self._vector_field),
            m=sql.SQL(metric_str),
        )

    def _create_index(self):
        log.info(
            f"{self.name} create BM25 index {self._index_name} "
            f"(metric={self.case_config.metric_type})"
        )
        emb_expr = self._embedding_index_expr()
        if self.with_scalar_labels:
            ddl = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON public.{tbl} "
                "USING bm25 ({pk}, {emb_expr}, {lbl}) "
                "WITH (key_field='{pk_str}');",
            ).format(
                idx=sql.Identifier(self._index_name),
                tbl=sql.Identifier(self.table_name),
                pk=sql.Identifier(self._primary_field),
                emb_expr=emb_expr,
                lbl=sql.Identifier(self._scalar_label_field),
                pk_str=sql.SQL(self._primary_field),
            )
        else:
            ddl = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON public.{tbl} "
                "USING bm25 ({pk}, {emb_expr}) "
                "WITH (key_field='{pk_str}');",
            ).format(
                idx=sql.Identifier(self._index_name),
                tbl=sql.Identifier(self.table_name),
                pk=sql.Identifier(self._primary_field),
                emb_expr=emb_expr,
                pk_str=sql.SQL(self._primary_field),
            )
        log.debug(ddl.as_string(self.cursor))
        self.cursor.execute(ddl)
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        log.info(f"{self.name} post-insert optimize: build BM25 index")
        self._drop_index()
        self._create_index()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            if self.with_scalar_labels:
                with self.cursor.copy(
                    sql.SQL("COPY public.{tbl} ({pk}, {emb}, {lbl}) FROM STDIN (FORMAT BINARY)").format(
                        tbl=sql.Identifier(self.table_name),
                        pk=sql.Identifier(self._primary_field),
                        emb=sql.Identifier(self._vector_field),
                        lbl=sql.Identifier(self._scalar_label_field),
                    ),
                ) as copy:
                    copy.set_types(["bigint", "vector", "varchar"])
                    for i, row in enumerate(metadata_arr):
                        copy.write_row((row, embeddings_arr[i], labels_data[i]))
            else:
                with self.cursor.copy(
                    sql.SQL("COPY public.{tbl} ({pk}, {emb}) FROM STDIN (FORMAT BINARY)").format(
                        tbl=sql.Identifier(self.table_name),
                        pk=sql.Identifier(self._primary_field),
                        emb=sql.Identifier(self._vector_field),
                    ),
                ) as copy:
                    copy.set_types(["bigint", "vector"])
                    for i, row in enumerate(metadata_arr):
                        copy.write_row((row, embeddings_arr[i]))

            self.conn.commit()
            return len(metadata), None
        except Exception as e:  # noqa: BLE001
            log.warning(f"{self.name} insert error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        # `id @@@ paradedb.all()` is required to engage the pg_search
        # custom scan; scalar filters are AND'd on top and pushed down
        # by filter_pushdown.rs when the column is in the index.
        base = f"WHERE {self._primary_field} @@@ paradedb.all()"
        if filters.type == FilterOp.NonFilter:
            self.where_clause = base
        elif filters.type == FilterOp.NumGE:
            # IntFilter targets the primary key column (`id`), not the label.
            self.where_clause = f"{base} AND {self._primary_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = (
                f"{base} AND {self._scalar_label_field} = '{filters.label_value}'"
            )
        else:
            raise ValueError(f"Filter type not supported by PgSearch: {filters}")

        self._search = self._generate_search_query()

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None
        assert self.cursor is not None

        q = np.asarray(query)
        result = self.cursor.execute(
            self._search,
            (q, k),
            prepare=True,
            binary=True,
        )
        return [int(row[0]) for row in result.fetchall()]
