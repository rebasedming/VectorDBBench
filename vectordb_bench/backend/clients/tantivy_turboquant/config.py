from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig, MetricType


class TantivyTurboquantConfig(DBConfig):
    index_path: str = "/tmp/tantivy_turboquant_bench"

    def to_dict(self) -> dict:
        return {"index_path": self.index_path}


class TantivyTurboquantIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType = MetricType.COSINE
    heap_mb: int = 200

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.IP:
            return "ip"
        if self.metric_type == MetricType.L2:
            return "l2"
        msg = f"Tantivy turboquant does not support metric: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {"metric": self.parse_metric(), "heap_mb": self.heap_mb}

    def search_param(self) -> dict:
        return {}
