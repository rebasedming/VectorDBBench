from typing import Any, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class PgSearchConfigDict(TypedDict):
    """Connection kwargs for psycopg + table name."""

    connect_config: dict[str, Any]
    table_name: str


class PgSearchConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr
    host: str = "localhost"
    port: int = 5432
    db_name: str = "vectordb"
    table_name: str = "vdbbench_pg_search"

    def to_dict(self) -> PgSearchConfigDict:
        user_str = (
            self.user_name.get_secret_value()
            if isinstance(self.user_name, SecretStr)
            else self.user_name
        )
        pwd_str = self.password.get_secret_value()
        return {
            "connect_config": {
                "host": self.host,
                "port": self.port,
                "dbname": self.db_name,
                "user": user_str,
                "password": pwd_str,
            },
            "table_name": self.table_name,
        }


class PgSearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

    # Per-segment cluster probe count for vector ORDER BY. Higher =
    # better recall, linearly more scoring work. Default in pg_search
    # is 50.
    vector_cluster_probes: int = 50

    # Over-fetch factor for exact-distance rerank. >1.0 engages a
    # heap-side rerank pass that reloads full-precision vectors for
    # `ceil(k * multiplier)` approximate candidates and re-sorts by
    # exact distance. 1.0 (default) disables rerank.
    vector_rerank_multiplier: float = 1.0

    # TurboQuant total bits per coordinate for the IVF/cluster path.
    # Allowed: 4 (3-bit codebook + 1-bit sign, default) or 5 (4-bit
    # codebook + 1-bit sign — same SIMD kernel cost, doubles the
    # stage-1 codebook size, reduces quantization mis-ranking).
    vector_bit_width: int = 4

    def index_param(self) -> dict[str, Any]:
        return {}

    def search_param(self) -> dict[str, Any]:
        return {}

    def session_param(self) -> dict[str, Any]:
        return {
            "session_options": [
                {
                    "parameter": {
                        "setting_name": "max_parallel_workers_per_gather",
                        "val": "0",
                    },
                },
                {
                    "parameter": {
                        "setting_name": "paradedb.vector_cluster_probes",
                        "val": str(self.vector_cluster_probes),
                    },
                },
                {
                    "parameter": {
                        "setting_name": "paradedb.vector_rerank_multiplier",
                        "val": str(self.vector_rerank_multiplier),
                    },
                },
            ],
        }
