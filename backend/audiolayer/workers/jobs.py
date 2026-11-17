"""In-process background job manager with progress, persistence and cancellation.

Jobs run in a thread pool (default 1 worker, because separation saturates the CPU/GPU).
State is persisted to ``<workspace>/jobs/<id>.json`` so status survives restarts; jobs that
were running when the process died are marked failed on startup. The interface is small
on purpose so it can be swapped for Celery/RQ/Arq in a multi-node deployment.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from ..core.errors import AudioLayerError, Cancelled, NotFound
from ..core.schemas import Job
from ..core.store import ProjectStore

log = logging.getLogger("audiolayer.jobs")


class JobContext:
    def __init__(self, manager: JobManager, job: Job):
        self._m = manager
        self.job = job
        self._cancel = threading.Event()
        self._last_save = 0.0
        self._range = (0.0, 1.0)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("Job was cancelled")

    def stage(self, name: str, lo: float, hi: float, message: str = "") -> None:
        self._range = (lo, hi)
        self.job.stage = name
        self.job.message = message or name
        self.progress(0.0, force=True)

    def progress(self, frac: float, *, force: bool = False) -> None:
        lo, hi = self._range
        self.job.progress = round(min(1.0, lo + (hi - lo) * max(0.0, min(1.0, frac))), 4)
        now = time.time()
        if force or now - self._last_save > 0.5:
            self._last_save = now
            self._m.store.save_job(self.job)

    def sub(self, lo: float, hi: float) -> Callable[[float], None]:
        """Progress callback mapping [0,1] into [lo,hi] of the current stage."""
        a, b = self._range

        def cb(f: float) -> None:
            self.job.progress = round(min(1.0, a + (b - a) * (lo + (hi - lo) * max(0.0, min(1.0, f)))), 4)
            if time.time() - self._last_save > 0.5:
                self._last_save = time.time()
                self._m.store.save_job(self.job)

        return cb


JobFn = Callable[[JobContext], dict[str, Any]]


class JobManager:
    def __init__(self, store: ProjectStore, workers: int = 1):
        self.store = store
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="audiolayer-job")
        self._ctx: dict[str, JobContext] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()
        self._recover()

    def _recover(self) -> None:
        jobs_dir = self.store.settings.jobs_dir
        if not jobs_dir.exists():
            return
        for f in jobs_dir.glob("*.json"):
            try:
                job = Job.model_validate_json(f.read_text())
            except Exception:  # noqa: S112 - skip unreadable job files
                continue
            if job.status in ("queued", "running"):
                job.status = "failed"
                job.error = "Server restarted while the job was running; start it again."
                job.finished_at = time.time()
                self.store.save_job(job)

    def submit(self, project_id: str, kind: str, fn: JobFn, *, owner: str = "local",
               on_done: Callable[[Job], None] | None = None) -> Job:
        job = Job(project_id=project_id, kind=kind, owner=owner)
        ctx = JobContext(self, job)
        self.store.save_job(job)
        with self._lock:
            self._ctx[job.id] = ctx
            self._futures[job.id] = self._pool.submit(self._run, ctx, fn, on_done)
        return job

    def _run(self, ctx: JobContext, fn: JobFn, on_done: Callable[[Job], None] | None) -> None:
        job = ctx.job
        if ctx.cancelled:
            job.status = "cancelled"
        else:
            job.status = "running"
            job.started_at = time.time()
            self.store.save_job(job)
            try:
                job.result = fn(ctx) or {}
                job.status = "succeeded"
                job.progress = 1.0
                job.stage = "done"
                job.message = "Completed"
            except Cancelled:
                job.status = "cancelled"
                job.message = "Cancelled by user"
            except AudioLayerError as e:
                job.status = "failed"
                job.error = e.message + (f" — {e.hint}" if e.hint else "")
                log.warning("job %s failed: %s", job.id, e.message)
            except Exception as e:  # noqa: BLE001 - surface any failure to the user
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                log.error("job %s crashed\n%s", job.id, traceback.format_exc())
        job.finished_at = time.time()
        self.store.save_job(job)
        if on_done:
            try:
                on_done(job)
            except Exception:  # noqa: BLE001
                log.exception("on_done hook failed")
        with self._lock:
            self._ctx.pop(job.id, None)

    def get(self, job_id: str) -> Job:
        with self._lock:
            ctx = self._ctx.get(job_id)
        if ctx:
            return ctx.job.model_copy(deep=True)
        return self.store.load_job(job_id)

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            ctx = self._ctx.get(job_id)
        if not ctx:
            job = self.store.load_job(job_id)
            if job.status in ("queued", "running"):
                raise NotFound("Job is not running in this process")
            return job
        ctx._cancel.set()
        return ctx.job.model_copy(deep=True)

    def active_for(self, project_id: str) -> list[Job]:
        with self._lock:
            return [c.job.model_copy(deep=True) for c in self._ctx.values() if c.job.project_id == project_id]

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        with self._lock:
            fut = self._futures.get(job_id)
        if fut:
            fut.result(timeout=timeout)
        return self.get(job_id)

    def shutdown(self) -> None:
        for ctx in list(self._ctx.values()):
            ctx._cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
