from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, MetricType


class PDXearchConfig(DBConfig):
    extension_path: str
    db_path: str = "/tmp/pdxearch_bench.duckdb"
    table_name: str = "pdxearch_bench"

    def to_dict(self) -> dict:
        return {
            "extension_path": self.extension_path,
            "db_path": self.db_path,
            "table_name": self.table_name,
        }


class PDXearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.COSINE
    quantization: str = "u8"
    n_probe: int = 24
    n_probe_search: int | None = None
    seed: int | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.L2:
            return "l2sq"
        msg = f"PDXearch does not support metric: {self.metric_type}"
        raise ValueError(msg)

    def parse_distance_fn(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "array_cosine_distance"
        if self.metric_type == MetricType.L2:
            return "array_distance"
        msg = f"PDXearch does not support metric: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        opts = {
            "metric": self.parse_metric(),
            "quantization": self.quantization,
            "n_probe": self.n_probe,
        }
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts

    def search_param(self) -> dict:
        return {"n_probe_search": self.n_probe_search}
