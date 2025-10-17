import os
import time
import logging
from typing import Union, Dict, Tuple, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Thread
from tqdm import tqdm

import httpx
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseKimina
from .models import CheckRequest, CheckResponse, Infotree, ReplResponse, Snippet
from .utils import build_log, find_code_column, find_id_column

logger = logging.getLogger("kimina-client")


class KiminaClient(BaseKimina):
    def __init__(
        self,
        api_url: Union[str, None] = None,
        api_key: Union[str, None] = None,
        headers: Union[Dict[str, str], None] = None,
        http_timeout: int = 600,
        n_retries: int = 3,
    ):
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            headers=headers,
            http_timeout=http_timeout,
            n_retries=n_retries,
        )

    def check(
        self,
        snips: Union[str, List[str], Snippet, List[Snippet]],
        timeout: int = 600,
        debug: bool = False,
        reuse: bool = True,
        infotree: Union[Infotree, None] = None,
        batch_size: int = 8,
        max_workers: int = 5,
        show_progress: bool = True,
        cache_file: str = None,
        cache_interval_length: int = 128,
        cache_interval_seconds: int = 300,
    ) -> CheckResponse:
        if isinstance(snips, str):
            snips = [snips]
        elif isinstance(snips, Snippet):
            snips = [snips]
        snippets = [Snippet.from_snip(snip) for snip in snips]

        results = []
        if cache_file is not None and os.path.exists(cache_file):
            with open(cache_file, "r", encoding='utf-8') as f:
                cached_response = CheckResponse.model_validate_json(f.read())
            cached_results = cached_response.results
            cached_id_to_result = {result.id:result for result in cached_results}
            logger.info(f"read {len(cached_results)} results from {cache_file}")
            noncached_snippets = []
            for snippet in snippets:
                if snippet.id in cached_id_to_result:
                    results.append(cached_id_to_result[snippet.id])
                else:
                    noncached_snippets.append(snippet)
            snippets = noncached_snippets
            print(f"reuse / left = {len(results)} / {len(snippets)}")

        batches = [
            snippets[i : i + batch_size] for i in range(0, len(snippets), batch_size)
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.api_check, batch, timeout, debug, reuse, infotree, True
                ): batch
                for batch in batches
            }
            iterator = (
                tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Batches",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]",
                )
                if show_progress
                else as_completed(futures)
            )
            last_save_time = time.time()
            last_save_length = len(results)
            for future in iterator:
                results.extend(future.result().results)
                if (cache_file is not None
                        and len(results) // cache_interval_length > last_save_length // cache_interval_length
                        and time.time() > last_save_time + cache_interval_seconds):
                    with open(cache_file, "w", encoding='utf-8') as f:
                        f.write(CheckResponse(results=results).model_dump_json(indent=4))
                    last_save_time = time.time()
                    last_save_length = len(results)
        return CheckResponse(results=results)

    def api_check(
        self,
        snippets: List[Snippet],
        timeout: int = 30,
        debug: bool = False,
        reuse: bool = True,
        infotree: Union[Infotree, None] = None,
        safe: bool = False,
    ) -> CheckResponse:
        """
        Makes a POST request to /api/check with provided arguments.

        Returns a `CheckResponse`.
        """
        try:
            url = self.build_url("/api/check")

            payload = CheckRequest(
                snippets=snippets,
                timeout=timeout,
                debug=debug,
                reuse=reuse,
                infotree=infotree,
            ).model_dump()

            resp = self._query(url, payload)
            return self.handle(resp, CheckResponse)
        except Exception as e:
            if safe:
                return CheckResponse(
                    results=[
                        ReplResponse(id=snip.id, error=str(e)) for snip in snippets
                    ],
                )
            raise e

    def _query(
        self, url: str, payload: Union[Dict[str, Any], None] = None, method: str = "POST"
    ) -> Any:
        """
        Sends a `method` request to `url` with `payload` as body/params.
        A new `httpx.Client` is created for each request for thread-safety.
        Use AsyncClient for more efficient concurrent requests (TCP connection reuse/pooling).
        """

        @retry(
            stop=stop_after_attempt(self.n_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.ERROR),
        )
        def run_method() -> Any:
            try:
                with httpx.Client(
                    headers=self.headers,
                    timeout=httpx.Timeout(self.http_timeout, read=self.http_timeout),
                ) as client:
                    if method.upper() == "POST":
                        response = client.post(url, json=payload)
                    elif method.upper() == "GET":
                        response = client.get(url, params=payload)
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    response.raise_for_status()  # Ensure 2xx, otherwise retry
            except httpx.HTTPError as e:
                logger.error(f"Error posting to {url}: {e}")
                raise e

            try:
                return response.json()  # Ensure JSON, otherwise retry
            except ValueError:
                logger.error(f"Server returned non-JSON: {response.text}")
                raise ValueError("Invalid response from server: not a valid JSON")

        try:
            return run_method()
        except RetryError:
            raise RuntimeError(f"Request failed after {self.n_retries} retries")

    def health(self) -> Any:
        """
        Checks server's healthy.
        """
        url = self.build_url("/health")
        resp = self._query(url, method="GET")
        return resp  # TODO: create status object to cast automatically

    def test(self) -> None:
        """
        Tests with `#check Nat`.
        """
        logger.info("Testing with `#check Nat`...")
        response = self.check("#check Nat", show_progress=False).results[0].response
        assert response is not None, "Response should not be None"
        assert response.get("messages", None) == [
            {
                "severity": "info",
                "pos": {"line": 1, "column": 0},
                "endPos": {"line": 1, "column": 6},
                "data": "Nat : Type",
            }
        ]
        logger.info("Test passed!")

    def run_benchmark(
        self,
        dataset_name: str = "Goedel-LM/Lean-workbook-proofs",
        split: str = "train",
        n: int = 100,
        batch_size: int = 8,
        max_workers: int = 5,
        timeout: int = 60,
        reuse: bool = True,
        show_progress: bool = True,
    ) -> None:
        """
        Runs benchmark on Hugging Face dataset.
        Displays results in the console.
        """
        # TODO: add option output dir with file hierarchy based on metadata like uuid in the run_benchmark method
        # TODO: add count heartbeats option

        if n <= 0:
            logger.error("Please specify n > 0")
            return

        if batch_size <= 0:
            logger.warning("Cannot use batch size = %d, defaulting to 8", batch_size)
            batch_size = 8

        logger.info(build_log(dataset_name, n, batch_size))

        try:
            from datasets import load_dataset, load_dataset_builder  # type: ignore
        except Exception as e:
            raise ImportError(
                "The 'datasets' library is required for run_benchmark.\n"
                "Install it with 'pip install datasets'."
            ) from e

        builder = load_dataset_builder(dataset_name)
        if not builder.info.features:
            logger.error("Dataset has no features, cannot run benchmark")
            return

        columns: List[str] = list(builder.info.features)

        id_column_name: Union[str, Tuple[str, str]] = "id"
        code_column_name: str = "code"
        if dataset_name == "Goedel-LM/Lean-workbook-proofs":
            id_column_name = "problem_id"
            code_column_name = "full_proof"
        elif dataset_name == "AI-MO/math-test-inference-results":
            id_column_name = ("uuid", "proof_id")
            code_column_name = "proof"
        else:
            id_column_name = find_id_column(columns)
            code_column_name = find_code_column(columns)

        dataset = load_dataset(dataset_name, split=split + f"[:{n}]")

        def get_id(sample: Any, id_column_name: Union[str, Tuple[str, str]]) -> str:
            if isinstance(id_column_name, tuple):
                a, b = id_column_name
                return str(sample[a]) + "_" + str(sample[b])
            return str(sample[id_column_name])

        snips = [
            Snippet(
                id=str(get_id(sample, id_column_name)),
                code=sample[code_column_name],  # type: ignore
            )
            for sample in dataset  # type: ignore
        ]

        start_time = time.time()
        check_response = self.check(
            snips=snips,
            timeout=timeout,
            reuse=reuse,
            batch_size=batch_size,
            max_workers=max_workers,
            show_progress=show_progress,
        )
        elapsed_time = time.time() - start_time

        check_response.analyze(elapsed_time)


class KiminaSandboxClient:
    def __init__(self, heartbeat_record_path, http_timeout=600):
        self.heartbeat_record_path = heartbeat_record_path
        self.http_timeout = http_timeout

    def get_active_ips(self, heartbeat_timeout=60):
        active_ips = []
        now = time.time()
        for filename in os.listdir(self.heartbeat_record_path):
            if filename.endswith('.heartbeat'):
                ip = filename[:-len('.heartbeat')]
                file_path = os.path.join(self.heartbeat_record_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        heartbeat_time = float(f.read().strip())
                    if now - heartbeat_time <= heartbeat_timeout:
                        active_ips.append(ip)
                except Exception as e:
                    print(f"fail to read {file_path}: {e}")
        return active_ips

    def check(
        self,
        snips: List,
        sandbox_batch_size: int = 4,
        sandbox_show_progress: bool = True,
        max_workers: int = 32,
        batch_size: int = 1,
        **kwargs,
    ) -> CheckResponse:
        active_ips = self.get_active_ips()
        if not active_ips:
            raise RuntimeError("No active servers available for check.")
        _n = "\n"
        logger.info(f"active_ips: {_n.join(active_ips)}")
        client_pool = [KiminaClient(api_url=f"http://{ip}:8888", http_timeout=self.http_timeout)
                       for ip in active_ips]
        n_clients = len(client_pool)
        batch_queue = Queue()
        result_queue = Queue()
        batches = [snips[i: i + sandbox_batch_size] for i in range(0, len(snips), sandbox_batch_size)]
        for batch in batches:
            batch_queue.put(batch)
        total_batches = len(batches)

        def worker(client: KiminaClient):
            logger.info(f"worker for {client.api_url} starts")
            while True:
                try:
                    batch_snips = batch_queue.get_nowait()
                except Exception:
                    break
                try:
                    response = client.check(
                        batch_snips,
                        timeout=self.http_timeout,
                        show_progress=False,
                        max_workers=max_workers,
                        batch_size=batch_size,
                        **{k: v for k, v in kwargs.items() if k not in ["timeout", "show_progress"]},
                    )
                    result_queue.put(response.results)
                except Exception as e:
                    logger.error(f"Error in worker: {e}")
                finally:
                    batch_queue.task_done()
            logger.info(f"worker for {client.api_url} exists")

        # 启动 worker 线程
        threads = []
        for client in client_pool:
            t = Thread(target=worker, args=(client,), daemon=True)
            t.start()
            threads.append(t)

        # 主线程定时用qsize刷新进度条
        if sandbox_show_progress:
            pbar = tqdm(total=total_batches, desc="KiminaSandboxClient")
        last_completed = 0
        while True:
            completed_batches = result_queue.qsize()
            if completed_batches > last_completed:
                if sandbox_show_progress:
                    pbar.update(completed_batches - last_completed)
                last_completed = completed_batches
            if completed_batches >= total_batches:
                break
            time.sleep(0.1)
        if sandbox_show_progress:
            pbar.close()
        # 等待所有线程结束
        for t in threads:
            t.join()

        # 汇总所有结果
        results = []
        while not result_queue.empty():
            results.extend(result_queue.get())

        return CheckResponse(results=results)