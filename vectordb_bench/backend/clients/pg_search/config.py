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
    """pg_search BM25 indexes have no per-query knobs at the SQL level —
    `max_probe`, `distance_ratio`, etc. are tantivy-side defaults baked
    into the build. So this is intentionally minimal.
    """

    metric_type: MetricType | None = None
    create_index_before_load: bool = False
    create_index_after_load: bool = True

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
            ],
        }
