import logging

import numpy as np
from pandas import DataFrame

from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.backend.filter import Filter, FilterOp

log = logging.getLogger(__name__)


def get_data(data_df: DataFrame, normalize: bool) -> tuple[list[list[float]], list[str]]:
    all_metadata = data_df["id"].tolist()
    emb_np = np.stack(data_df["emb"])
    if normalize:
        log.debug("normalize the 100k train data")
        all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
    else:
        all_embeddings = emb_np.tolist()
    return all_embeddings, all_metadata


def get_data_with_labels(
    data_df: DataFrame,
    normalize: bool,
    dataset: DatasetManager,
    filters: Filter,
) -> tuple[list[list[float]], list[str], list[str] | None]:
    """Same as get_data, plus a labels_data list when the active filter is a
    label-equality predicate.

    Mirrors the join performed in ConcurrentInsertRunner._next_batch:
    when scalar_labels live in a separate parquet, look up labels by id;
    otherwise, read the column directly off the train batch.
    """
    all_embeddings, all_metadata = get_data(data_df, normalize)
    labels_data: list[str] | None = None
    if filters.type == FilterOp.StrEqual:
        if dataset.data.scalar_labels_file_separated:
            labels_data = dataset.scalar_labels[filters.label_field][all_metadata].to_list()
        else:
            labels_data = data_df[filters.label_field].tolist()
    return all_embeddings, all_metadata, labels_data
