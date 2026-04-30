import concurrent
import concurrent.futures
import logging
import math
import multiprocessing as mp
import time
from collections.abc import Iterable

import numpy as np

from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.backend.filter import Filter, FilterOp, StreamingLabelFilter, non_filter
from vectordb_bench.backend.utils import time_it
from vectordb_bench.metric import Metric

from .mp_runner import MultiProcessingSearchRunner
from .rate_runner import RatedMultiThreadingInsertRunner
from .serial_runner import SerialSearchRunner
from .util import get_data, get_data_with_labels

log = logging.getLogger(__name__)


class ReadWriteRunner(MultiProcessingSearchRunner, RatedMultiThreadingInsertRunner):
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        insert_rate: int = 1000,
        normalize: bool = False,
        k: int = 100,
        filters: Filter = non_filter,
        concurrencies: Iterable[int] = (1, 15, 50),
        search_stages: Iterable[float] = (
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),  # search from insert portion, 0.0 means search from the start
        optimize_after_write: bool = True,
        read_dur_after_write: int = 300,  # seconds, search duration when insertion is done
        timeout: float | None = None,
        label_percentage: float | None = None,
        skip_search_concurrent: bool = False,
        seed_fraction: float | None = None,
    ):
        self.insert_rate = insert_rate
        self.data_volume = dataset.data.size
        self.ds_manager = dataset
        self.k = k
        self.label_percentage = label_percentage
        self.skip_search_concurrent = skip_search_concurrent
        self.seed_fraction = seed_fraction

        for stage in search_stages:
            assert 0.0 <= stage < 1.0, "each search stage should be in [0.0, 1.0)"
        self.search_stages = sorted(search_stages)
        self.optimize_after_write = optimize_after_write
        self.read_dur_after_write = read_dur_after_write

        conc_desc = "skipped" if skip_search_concurrent else f"{list(concurrencies)}"
        log.info(
            f"Init runner, concurencys={conc_desc}, search_stages={self.search_stages}, "
            f"stage_search_dur={read_dur_after_write}, label_percentage={label_percentage}",
        )

        if normalize:
            test_emb = np.array(dataset.test_data)
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
            test_emb = test_emb.tolist()
        else:
            test_emb = dataset.test_data
        self.test_emb = test_emb

        MultiProcessingSearchRunner.__init__(
            self,
            db=db,
            test_data=test_emb,
            k=k,
            filters=filters,
            concurrencies=concurrencies,
        )
        RatedMultiThreadingInsertRunner.__init__(
            self,
            rate=insert_rate,
            db=db,
            dataset_iter=iter(dataset),
            normalize=normalize,
            filters=filters,
        )
        # Per-stage runner is rebuilt at each stage so it picks up that
        # stage's GT and (when filtered) its StreamingLabelFilter. The
        # initial value here is just a placeholder for the post-insert /
        # post-optimize searches at stage 1.0, which use the static GT.
        self.serial_search_runner = SerialSearchRunner(
            db=db,
            test_data=test_emb,
            ground_truth=dataset.gt_data,
            k=k,
            filters=filters,
        )
        # Remember the case's static filter so we can restore self.filters
        # after the streaming loop mutates it per-stage.
        self.case_filter = filters

    def _stage_filter_and_gt(self, stage: float) -> tuple[Filter, list[list[int]] | None]:
        """Pick the filter and GT to use for measurement at insertion progress `stage`.

        For stage < 1.0 with a label predicate, we use a StreamingLabelFilter
        (which selects the per-stage GT file) and the loaded per-stage GT.
        For stage >= 1.0, we fall back to the static (full-dataset) GT and
        the case's static filter, which is what already lives on `self`.
        """
        if stage >= 1.0 or self.label_percentage is None:
            return self.filters, self.ds_manager.gt_data
        s_int = round(stage * 100)
        gt = self.ds_manager.gt_data_by_stage.get(s_int)
        if gt is None:
            log.warning(
                f"per-stage GT missing for stage={stage} (key={s_int}); "
                f"falling back to full-dataset GT — recall will be biased."
            )
            return self.filters, self.ds_manager.gt_data
        stage_filter = StreamingLabelFilter(
            label_percentage=self.label_percentage,
            stage=stage,
        )
        return stage_filter, gt

    def _make_stage_serial_runner(self, stage: float) -> tuple[SerialSearchRunner, Filter]:
        stage_filter, gt = self._stage_filter_and_gt(stage)
        runner = SerialSearchRunner(
            db=self.db,
            test_data=self.test_emb,
            ground_truth=gt,
            k=self.k,
            filters=stage_filter,
        )
        return runner, stage_filter

    @time_it
    def run_seed_load(self) -> int:
        """Bulk-insert the first `seed_fraction * N` rows synchronously, then
        build the index.

        For ANN structures like IVF that require data to compute centroids,
        this gives the index a meaningful starting state before streaming
        updates start landing on it. The dataset iterator on `self.dataset`
        is consumed in place, so subsequent `run_with_rate` continues from
        the row right after the seed.
        """
        target = round((self.seed_fraction or 0) * self.data_volume)
        if target <= 0:
            return 0
        log.info(f"Seed load - target {target} rows ({self.seed_fraction:.0%} of dataset)")
        total = 0
        use_labels = self.filters.type == FilterOp.StrEqual
        with self.db.init():
            for data in self.dataset:
                if use_labels:
                    emb, ids, labels_data = get_data_with_labels(
                        data, self.normalize, self.dataset._ds, self.filters,
                    )
                else:
                    emb, ids = get_data(data, self.normalize)
                    labels_data = None
                inserted, err = self.db.insert_embeddings(emb, ids, labels_data=labels_data)
                if err is not None:
                    raise err
                total += inserted
                if total >= target:
                    break
            log.info(f"Seed load - inserted {total} rows; now building index")
            self.db.optimize(data_size=total)
            log.info("Seed load - optimize done")
        return total

    @time_it
    def run_optimize(self):
        """Optimize needs to run in differenct process for pymilvus schema recursion problem"""
        with self.db.init():
            log.info("Search after write - Optimize start")
            self.db.optimize(data_size=self.data_volume)
            log.info("Search after write - Optimize finished")

    def run_search(self, perc: int):
        log.info("Search after write - Serial search start")
        test_time = round(time.perf_counter(), 4)
        res, ssearch_dur = self.serial_search_runner.run()
        recall, ndcg, p99_latency, p95_latency, _avg_latency = res
        log.info(
            f"Search after write - Serial search - recall={recall}, ndcg={ndcg}, "
            f"p99={p99_latency}, p95={p95_latency}, dur={ssearch_dur:.4f}",
        )
        max_qps, conc_failed_rate = 0, 0
        conc_num_list, conc_qps_list = [], []
        conc_latency_p99_list, conc_latency_p95_list, conc_latency_avg_list = [], [], []
        if self.skip_search_concurrent:
            log.info("Search after write - Conc search skipped (skip_search_concurrent=True)")
        else:
            log.info(
                f"Search after wirte - Conc search start, dur for each conc={self.read_dur_after_write}",
            )
            result = self.run_by_dur(self.read_dur_after_write)
            max_qps = result[0]
            conc_failed_rate = result[1]
            conc_num_list = result[2]
            conc_qps_list = result[3]
            conc_latency_p99_list = result[4]
            conc_latency_p95_list = result[5]
            conc_latency_avg_list = result[6]
            log.info(f"Search after wirte - Conc search finished, max_qps={max_qps}")

        return [
            (
                perc,
                test_time,
                max_qps,
                recall,
                ndcg,
                p99_latency,
                p95_latency,
                conc_failed_rate,
                conc_num_list,
                conc_qps_list,
                conc_latency_p99_list,
                conc_latency_p95_list,
                conc_latency_avg_list,
            )
        ]

    def run_read_write(self) -> Metric:  # noqa: PLR0915
        """
        Test search performance with a fixed insert rate.
        - Insert requests are sent to VectorDB at a fixed rate within a dedicated insert process pool.
          - if the database cannot promptly process these requests, the process pool will accumulate insert tasks.
        - Search Tests are categorized into three types:
          - streaming_search: Initiates a new search test upon receiving a signal that the inserted data has
          reached the search_stage.
          - streaming_end_search: initiates a new search test after all data has been inserted.
          - optimized_search (optional): After the streaming_end_search, optimizes and initiates a search test.
        """
        m = Metric()
        if self.seed_fraction:
            seed_count, seed_dur = self.run_seed_load()
            log.info(f"Seed load complete: {seed_count} rows, optimize+load in {seed_dur:.1f}s")
            # The streaming-search loop's batch math is "seconds-of-inserts
            # streamed by run_with_rate." Seed rows didn't go through
            # run_with_rate, so subtract them when computing how many
            # streaming batches we need to wait for at each stage.
            self.seed_batches_offset = math.ceil(seed_count / self.insert_rate)
        else:
            self.seed_batches_offset = 0
        with mp.Manager() as mp_manager:
            q = mp_manager.Queue()
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=2) as executor:
                insert_future = executor.submit(self.run_with_rate, q)
                streaming_search_future = executor.submit(self.run_search_by_sig, q)

                try:
                    start_time = time.perf_counter()
                    _, m.insert_duration = insert_future.result()
                    streaming_search_res = streaming_search_future.result()
                    if streaming_search_res is None:
                        streaming_search_res = []

                    streaming_end_search_future = executor.submit(self.run_search, 100)
                    streaming_end_search_res = streaming_end_search_future.result()

                    # Wait for read_write_futures finishing and do optimize and search.
                    # When seeded, the index was already built up front and we
                    # explicitly do NOT rebuild it — the whole point is to
                    # measure how the original index degrades under updates.
                    if self.optimize_after_write and not self.seed_fraction:
                        op_future = executor.submit(self.run_optimize)
                        _, m.optimize_duration = op_future.result()
                        log.info(f"Optimize cost {m.optimize_duration}s")
                        optimized_search_future = executor.submit(self.run_search, 110)
                        optimized_search_res = optimized_search_future.result()
                    else:
                        if self.seed_fraction:
                            log.info("Skip post-insert optimize (seeded run preserves the original index)")
                        else:
                            log.info("Skip optimization and search")
                        optimized_search_res = []

                    r = [*streaming_search_res, *streaming_end_search_res, *optimized_search_res]
                    m.st_search_stage_list = [d[0] for d in r]
                    m.st_search_time_list = [round(d[1] - start_time, 4) for d in r]
                    m.st_max_qps_list_list = [d[2] for d in r]
                    m.st_recall_list = [d[3] for d in r]
                    m.st_ndcg_list = [d[4] for d in r]
                    m.st_serial_latency_p99_list = [d[5] for d in r]
                    m.st_serial_latency_p95_list = [d[6] for d in r]
                    m.st_conc_failed_rate_list = [d[7] for d in r]

                    # Extract concurrent latency data
                    m.st_conc_num_list_list = [d[8] for d in r]
                    m.st_conc_qps_list_list = [d[9] for d in r]
                    m.st_conc_latency_p99_list_list = [d[10] for d in r]
                    m.st_conc_latency_p95_list_list = [d[11] for d in r]
                    m.st_conc_latency_avg_list_list = [d[12] for d in r]

                except Exception as e:
                    log.warning(f"Read and write error: {e}")
                    executor.shutdown(wait=True, cancel_futures=True)
                    # raise e
        m.st_ideal_insert_duration = math.ceil(self.data_volume / self.insert_rate)
        log.info(f"Concurrent read write all done, results: {m}")
        return m

    def get_each_conc_search_dur(self, ssearch_dur: float, cur_stage: float, next_stage: float) -> float:
        # Search duration for non-last search stage is carefully calculated.
        # If duration for each concurrency is less than 30s, runner will raise error.
        total_dur_between_stages = self.data_volume * (next_stage - cur_stage) // self.insert_rate
        csearch_dur = total_dur_between_stages - ssearch_dur

        # Try to leave room for init process executors
        if csearch_dur > 60:
            csearch_dur -= 30
        elif csearch_dur > 30:
            csearch_dur -= 15
        else:
            csearch_dur /= 2

        each_conc_search_dur = round(csearch_dur / len(self.concurrencies), 4)
        if each_conc_search_dur < 30:
            warning_msg = (
                f"Results might be inaccurate, duration[{csearch_dur:.4f}] left for conc-search is too short, "
                f"total available dur={total_dur_between_stages}, serial_search_cost={ssearch_dur}, "
                f"each_conc_search_dur={each_conc_search_dur}."
            )
            log.warning(warning_msg)
        return each_conc_search_dur

    def run_search_by_sig(self, q: mp.Queue):  # noqa: PLR0915
        """
        Args:
            q: multiprocessing queue
                (None) means abnormal exit
                (False) means updating progress
                (True) means normal exit
        """
        result, start_batch = [], 0
        total_batch = math.ceil(self.data_volume / self.insert_rate)
        recall, ndcg, p99_latency, p95_latency = None, None, None, None

        def wait_next_target(start: int, target_batch: int) -> bool:
            """Return False when receive True or None"""
            while start < target_batch:
                sig = q.get(block=True)

                if sig is None or sig is True:
                    return False
                start += 1
            return True

        seed_offset = getattr(self, "seed_batches_offset", 0)
        for idx, stage in enumerate(self.search_stages):
            # Seed rows didn't flow through run_with_rate, so subtract them
            # from the target. A stage at or below the seed becomes
            # target=0 and fires immediately — that's the user's "load the
            # first stage, build index, measure" point.
            target_batch = max(0, int(total_batch * stage) - seed_offset)
            perc = int(stage * 100)

            got = wait_next_target(start_batch, target_batch)
            if got is False:
                log.warning(f"Abnormal exit, target_batch={target_batch}, start_batch={start_batch}")
                return None

            log.info(f"Insert {perc}% done, total batch={total_batch}")
            test_time = round(time.perf_counter(), 4)
            max_qps, recall, ndcg, p99_latency, p95_latency, conc_failed_rate = 0, 0, 0, 0, 0, 0
            conc_num_list, conc_qps_list = [], []
            conc_latency_p99_list, conc_latency_p95_list, conc_latency_avg_list = [], [], []
            try:
                # Build a per-stage serial search runner whose ground truth
                # is computed against the active set at this stage (i.e.,
                # what's actually in the index). For the filtered case this
                # also swaps the search-time filter to a StreamingLabelFilter
                # so the concurrent QPS measurement queries with the same
                # predicate.
                stage_serial_runner, stage_filter = self._make_stage_serial_runner(stage)
                self.filters = stage_filter
                log.info(
                    f"[{target_batch}/{total_batch}] Serial search - {perc}% start "
                    f"(filter={stage_filter})",
                )
                res, ssearch_dur = stage_serial_runner.run()
                ssearch_dur = round(ssearch_dur, 4)
                recall, ndcg, p99_latency, p95_latency, _avg_latency = res
                log.info(
                    f"[{target_batch}/{total_batch}] Serial search - {perc}% done, "
                    f"recall={recall}, ndcg={ndcg}, p99={p99_latency}, p95={p95_latency}, dur={ssearch_dur}"
                )

                if self.skip_search_concurrent:
                    log.info(f"[{target_batch}/{total_batch}] Concurrent search skipped at {perc}%")
                else:
                    each_conc_search_dur = self.get_each_conc_search_dur(
                        ssearch_dur,
                        cur_stage=stage,
                        next_stage=self.search_stages[idx + 1] if idx < len(self.search_stages) - 1 else 1.0,
                    )
                    if each_conc_search_dur > 10:
                        log.info(
                            f"[{target_batch}/{total_batch}] Concurrent search - {perc}% start, "
                            f"dur={each_conc_search_dur:.4f}"
                        )
                        conc_result = self.run_by_dur(each_conc_search_dur)
                        max_qps = conc_result[0]
                        conc_failed_rate = conc_result[1]
                        conc_num_list = conc_result[2]
                        conc_qps_list = conc_result[3]
                        conc_latency_p99_list = conc_result[4]
                        conc_latency_p95_list = conc_result[5]
                        conc_latency_avg_list = conc_result[6]
                    else:
                        log.warning(
                            f"Skip concurrent tests, each_conc_search_dur={each_conc_search_dur} less than 10s.",
                        )
            except Exception as e:
                log.warning(f"Streaming Search Failed at stage={stage}. Exception: {e}")
            result.append(
                (
                    perc,
                    test_time,
                    max_qps,
                    recall,
                    ndcg,
                    p99_latency,
                    p95_latency,
                    conc_failed_rate,
                    conc_num_list,
                    conc_qps_list,
                    conc_latency_p99_list,
                    conc_latency_p95_list,
                    conc_latency_avg_list,
                )
            )
            start_batch = target_batch

        # Restore the case's static filter so the post-insert / post-optimize
        # searches in run_search() use it (the streaming loop above mutates
        # self.filters per stage for the concurrent QPS measurement).
        self.filters = self.case_filter

        # Drain the queue
        while q.empty() is False:
            q.get(block=True)
        return result
