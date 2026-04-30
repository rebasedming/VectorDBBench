import concurrent
import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor

from pandas import DataFrame

from vectordb_bench import config
from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DataSetIterator
from vectordb_bench.backend.filter import Filter, FilterOp, non_filter
from vectordb_bench.backend.utils import time_it

from .util import get_data, get_data_with_labels

log = logging.getLogger(__name__)


class RatedMultiThreadingInsertRunner:
    def __init__(
        self,
        rate: int,  # numRows per second
        db: api.VectorDB,
        dataset_iter: DataSetIterator,
        normalize: bool = False,
        timeout: float | None = None,
        filters: Filter = non_filter,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset_iter
        self.db = db
        self.normalize = normalize
        self.insert_rate = rate
        self.batch_rate = rate // config.NUM_PER_BATCH
        self.filters = filters

        self.executing_futures = []
        self.sig_idx = 0

    def send_insert_task(
        self,
        db: api.VectorDB,
        emb: list[list[float]],
        metadata: list[str],
        labels_data: list[str] | None = None,
    ):
        def _insert_embeddings(retry_idx: int = 0):
            _, error = db.insert_embeddings(emb, metadata, labels_data=labels_data)
            if error is not None:
                log.warning(f"Insert Failed, try_idx={retry_idx}, Exception: {error}")
                retry_idx += 1
                if retry_idx <= config.MAX_INSERT_RETRY:
                    time.sleep(retry_idx)
                    _insert_embeddings(retry_idx=retry_idx)
                else:
                    msg = f"Insert failed and retried more than {config.MAX_INSERT_RETRY} times"
                    raise RuntimeError(msg) from None

        # Connection / cursor opened by run_with_rate's `with self.db.init()`
        # is shared across worker threads. Non-thread-safe DBs (PgVector,
        # Doris, ...) clamp max_workers to 1 in run_with_rate, so this is
        # the sole worker — no concurrent access to the shared connection.
        # See ConcurrentInsertRunner._get_thread_db for the same pattern.
        _insert_embeddings(retry_idx=0)

    def _prepare_batch(self, data: DataFrame) -> tuple[list[list[float]], list[str], list[str] | None]:
        """Pull (emb, metadata, labels_data) for one parquet batch, joining
        labels per-id when the active filter is a label predicate."""
        if self.filters.type == FilterOp.StrEqual:
            return get_data_with_labels(data, self.normalize, self.dataset._ds, self.filters)
        emb, metadata = get_data(data, self.normalize)
        return emb, metadata, None

    @time_it
    def run_with_rate(self, q: mp.Queue):  # noqa: PLR0915
        # Non-thread-safe clients (PgVector, Doris, ...) must run a single
        # worker — the open connection is shared via self.db. Thread-safe
        # clients can fan out across cores for higher insert throughput.
        max_workers = mp.cpu_count() if getattr(self.db, "thread_safe", True) else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            @time_it
            def submit_by_rate() -> bool:
                rate = self.batch_rate
                for data in self.dataset:
                    emb, metadata, labels_data = self._prepare_batch(data)
                    self.executing_futures.append(
                        executor.submit(
                            self.send_insert_task, self.db, emb, metadata, labels_data,
                        ),
                    )
                    rate -= 1

                    if rate == 0:
                        return False
                return rate == self.batch_rate

            def check_and_send_signal(wait_interval: float, finished: bool = False):
                try:
                    done, not_done = concurrent.futures.wait(
                        self.executing_futures,
                        timeout=wait_interval,
                        return_when=concurrent.futures.FIRST_EXCEPTION,
                    )
                    _ = [fut.result() for fut in done]
                    if len(not_done) > 0:
                        self.executing_futures = list(not_done)
                    else:
                        self.executing_futures = []

                    self.sig_idx += len(done)
                    while self.sig_idx >= self.batch_rate:
                        self.sig_idx -= self.batch_rate
                        if self.sig_idx < self.batch_rate and len(not_done) == 0 and finished:
                            q.put(True, block=True)
                        else:
                            q.put(False, block=False)

                except Exception as e:
                    log.warning(f"task error, terminating, err={e}")
                    q.put(None, block=True)
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise e from None

            time_per_batch = config.TIME_PER_BATCH
            with self.db.init():
                start_time = time.perf_counter()
                round_idx = 0

                while True:
                    if len(self.executing_futures) > 200:
                        log.warning("Skip data insertion this round. There are 200+ unfinished insertion tasks.")
                    else:
                        finished, elapsed_time = submit_by_rate()
                        if finished is True:
                            log.info(
                                f"End of dataset, left unfinished={len(self.executing_futures)}, num_round={round_idx}"
                            )
                            break
                        if elapsed_time >= 1.5:
                            log.warning(
                                f"Submit insert tasks took {elapsed_time}s, expected 1s, "
                                f"indicating potential resource limitations on the client machine.",
                            )

                    check_and_send_signal(wait_interval=0.001, finished=False)
                    dur = time.perf_counter() - start_time - round_idx * time_per_batch
                    if dur < time_per_batch:
                        time.sleep(time_per_batch - dur)
                    round_idx += 1

                # wait for all tasks in executing_futures to complete
                while len(self.executing_futures) > 0:
                    check_and_send_signal(wait_interval=1, finished=True)
                    round_idx += 1

                log.info(f"Finish all streaming insertion, num_round={round_idx}")
