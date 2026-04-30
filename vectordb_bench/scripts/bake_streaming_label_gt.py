"""Bake per-stage ground truth for the streaming + label-filtered benchmark.

For each stage s in (0, 1), the GT is computed against the *active set at
that stage*: the first int(s * N) vectors in the streaming insertion order,
intersected with rows whose label matches the chosen label percentage.

Insertion order = the order DataSetIterator yields rows = the order of
train_files concatenated, in file order. The streaming runner consumes the
dataset in that exact same order, so this prefix is the set of vectors
actually present in the index at stage s.

Output naming follows the existing static GT convention (see filter.py:93):
  neighbors_{label_field}_{label_value}_stage_{int(s*100)}.parquet

Schema matches existing GT files: one column "neighbors_id" of list[int].

Usage:
    python -m vectordb_bench.scripts.bake_streaming_label_gt \\
        --dataset cohere --size 1000000 \\
        --label-percentage 0.05 \\
        --stages 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \\
        --k 100

By default reads from the dataset's standard local data dir
(config.DATASET_LOCAL_DIR/<name>/<dir_name>/) and writes outputs to the
same directory. Run the regular dataset download path first to populate it.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import numpy as np
import polars as pl

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.data_source import DatasetSource
from vectordb_bench.backend.dataset import BaseDataset, Dataset
from vectordb_bench.backend.filter import LabelFilter

log = logging.getLogger(__name__)

DATASETS = {
    "cohere": Dataset.COHERE,
    "openai": Dataset.OPENAI,
    "bioasq": Dataset.BIOASQ,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--label-percentage", type=float, required=True)
    p.add_argument(
        "--stages",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated insertion-progress fractions in (0, 1).",
    )
    p.add_argument("--k", type=int, default=100)
    p.add_argument(
        "--tile-size",
        type=int,
        default=500,
        help="Test queries per tile (controls peak memory).",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override input dir (default: dataset's standard local path).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output dir (default: same as --data-dir).",
    )
    p.add_argument(
        "--source",
        type=str,
        choices=["S3", "AliyunOSS"],
        default="S3",
        help="Remote source for dataset download.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume train/test/labels are already on disk; skip remote download.",
    )
    return p.parse_args()


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def stack_embeddings(series: pl.Series) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(np.float32)


def compute_topk_block(
    scores: np.ndarray,
    active_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    """For each row in `scores`, return the top-k active_ids by descending score.

    scores: (T, M) inner products against the M active vectors.
    active_ids: (M,) original IDs corresponding to score columns.
    Returns: (T, k) int64. If M < k, rows are padded with -1.
    """
    n_queries, n_candidates = scores.shape
    if n_candidates == 0:
        return np.full((n_queries, k), -1, dtype=np.int64)

    if n_candidates <= k:
        order = np.argsort(-scores, axis=1)
        out = active_ids[order]
        if order.shape[1] < k:
            pad = np.full((n_queries, k - order.shape[1]), -1, dtype=np.int64)
            out = np.concatenate([out, pad], axis=1)
        return out.astype(np.int64)

    # argpartition for top-k positions (unsorted within), then sort by score.
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row_idx = np.arange(n_queries)[:, None]
    part_scores = scores[row_idx, part]
    order_within = np.argsort(-part_scores, axis=1)
    sorted_local = part[row_idx, order_within]
    return active_ids[sorted_local].astype(np.int64)


def _load_inputs(base: BaseDataset, data_dir: pathlib.Path, label_field: str):
    """Load train / test / labels from disk, normalize for cosine.

    Returns (train_vecs, test_vecs, insertion_ids, labels_arr, n_total, n_queries) or
    None on validation failure.
    """
    log.info(f"loading train: {base.train_files} from {data_dir}")
    train_dfs = [pl.read_parquet(data_dir / f) for f in base.train_files]
    train_df = pl.concat(train_dfs)
    n_total = len(train_df)
    log.info(f"train: {n_total} rows")
    if n_total != base.size:
        log.warning(f"train row count ({n_total}) != dataset.size ({base.size})")

    test_path = data_dir / base.test_file
    log.info(f"loading test: {test_path}")
    test_df = pl.read_parquet(test_path)
    test_vecs = stack_embeddings(test_df[base.test_vector_field])
    n_queries = len(test_vecs)
    log.info(f"test: {n_queries} queries, dim={test_vecs.shape[1]}")

    labels_path = data_dir / base.scalar_labels_file
    log.info(f"loading labels: {labels_path}")
    labels_arr = pl.read_parquet(labels_path)[label_field].to_numpy()
    if len(labels_arr) != n_total:
        log.error(f"scalar_labels size ({len(labels_arr)}) != train size ({n_total})")
        return None

    insertion_ids = train_df[base.train_id_field].to_numpy().astype(np.int64)
    train_vecs = stack_embeddings(train_df[base.train_vector_field])

    if base.metric_type == MetricType.COSINE:
        log.info("normalizing for cosine -> inner product")
        train_vecs = normalize(train_vecs).astype(np.float32)
        test_vecs = normalize(test_vecs).astype(np.float32)
    elif base.metric_type != MetricType.IP:
        log.error(f"metric {base.metric_type} not supported by this script (need cosine or IP)")
        return None

    return train_vecs, test_vecs, insertion_ids, labels_arr, n_total, n_queries


def _build_stage_specs(
    stages: list[float],
    insertion_ids: np.ndarray,
    labels_arr: np.ndarray,
    label_value: str,
    n_total: int,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    stage_specs = []
    for s in stages:
        prefix = round(s * n_total)
        prefix_ids = insertion_ids[:prefix]
        match_mask = labels_arr[prefix_ids] == label_value
        active_local = np.where(match_mask)[0]
        active_ids = prefix_ids[match_mask]
        log.info(
            f"stage {s:.2f}: prefix_rows={prefix}, "
            f"label_matches={len(active_ids)} "
            f"({100 * len(active_ids) / max(prefix, 1):.2f}% of prefix)",
        )
        stage_specs.append((s, active_local, active_ids))
    return stage_specs


def _compute_per_stage_topk(
    test_vecs: np.ndarray,
    train_vecs: np.ndarray,
    stage_specs: list[tuple[float, np.ndarray, np.ndarray]],
    k: int,
    tile: int,
) -> dict[float, np.ndarray]:
    n_queries = test_vecs.shape[0]
    out = {s: np.empty((n_queries, k), dtype=np.int64) for s, _, _ in stage_specs}
    t_start = time.perf_counter()
    for q_start in range(0, n_queries, tile):
        q_end = min(q_start + tile, n_queries)
        q_block = test_vecs[q_start:q_end]
        for s, active_local, active_ids in stage_specs:
            if len(active_ids) == 0:
                out[s][q_start:q_end] = -1
                continue
            scores = q_block @ train_vecs[active_local].T
            out[s][q_start:q_end] = compute_topk_block(scores, active_ids, k)
        log.info(f"tile [{q_start}:{q_end}]/{n_queries} done")
    log.info(f"all tiles done in {time.perf_counter() - t_start:.1f}s")
    return out


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[bake] %(message)s")

    stages = [float(s) for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if not 0.0 < s < 1.0:
            log.error(f"stage {s} must be in (0, 1)")
            return 1

    mgr = DATASETS[args.dataset].manager(args.size)
    base = mgr.data
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else pathlib.Path(mgr.data_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (base.with_scalar_labels and base.scalar_labels_file_separated):
        log.error(f"dataset {base.name} has no separate scalar_labels file; not supported")
        return 1

    label_filter = LabelFilter(label_percentage=args.label_percentage)
    label_field, label_value = label_filter.label_field, label_filter.label_value
    log.info(f'label predicate: {label_field} == "{label_value}"')

    # Fetch train / test / scalar_labels (and the static GT for that label,
    # which the runtime needs anyway) from S3 if not already present.
    # When --data-dir is set, the user is pointing at a self-managed dir
    # and is responsible for populating it.
    if not args.skip_download and not args.data_dir:
        log.info(f"ensuring dataset is downloaded to {data_dir} (source={args.source})")
        mgr.prepare(source=DatasetSource[args.source], filters=label_filter)

    loaded = _load_inputs(base, data_dir, label_field)
    if loaded is None:
        return 1
    train_vecs, test_vecs, insertion_ids, labels_arr, n_total, _ = loaded

    stage_specs = _build_stage_specs(stages, insertion_ids, labels_arr, label_value, n_total)
    per_stage_topk = _compute_per_stage_topk(test_vecs, train_vecs, stage_specs, args.k, args.tile_size)

    for s, _, _ in stage_specs:
        out_rows = [[int(x) for x in r if x != -1] for r in per_stage_topk[s]]
        s_int = round(s * 100)
        out_path = out_dir / f"neighbors_{label_field}_{label_value}_stage_{s_int}.parquet"
        pl.DataFrame({base.gt_neighbors_field: out_rows}).write_parquet(out_path)
        log.info(f"wrote {out_path}")

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
