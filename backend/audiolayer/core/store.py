"""File-based project store inside the configured workspace.

Layout::

    <workspace>/projects/<project_id>/
        project.json          # Project model
        original/<file>       # untouched upload (never modified)
        audio/master.wav      # extracted working audio (48 kHz float)
        streams/*.wav         # separated streams (voice, music, fx, noise)
        stems/*.flac          # category stems rendered for preview
        renders/              # preview mixes
        exports/              # export files and ZIP packages
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..config import Settings
from .errors import Forbidden, NotFound
from .schemas import Job, Project

_SAFE = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def _check_id(value: str) -> str:
    if not value or len(value) > 64 or not set(value.lower()) <= _SAFE:
        raise NotFound(f"Unknown id: {value!r}")
    return value


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}{threading.get_ident()}")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


class ProjectStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._locks: dict[str, threading.RLock] = {}
        self._glock = threading.Lock()

    # ---- paths -------------------------------------------------------------
    def project_dir(self, project_id: str) -> Path:
        return self.settings.projects_dir / _check_id(project_id)

    def path(self, project_id: str, *parts: str) -> Path:
        base = self.project_dir(project_id).resolve()
        p = base.joinpath(*parts).resolve()
        if base != p and base not in p.parents:
            raise Forbidden("Path escapes project directory")
        return p

    def lock(self, project_id: str) -> threading.RLock:
        with self._glock:
            return self._locks.setdefault(project_id, threading.RLock())

    # ---- projects ----------------------------------------------------------
    def create_dirs(self, project_id: str) -> Path:
        d = self.project_dir(project_id)
        for sub in ("original", "audio", "streams", "stems", "renders", "exports", "analysis"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    def save(self, project: Project) -> None:
        with self.lock(project.id):
            project.updated_at = time.time()
            _atomic_write(self.path(project.id, "project.json"), project.model_dump_json(indent=2))

    def load(self, project_id: str, *, owner: str | None = None) -> Project:
        p = self.project_dir(project_id) / "project.json"
        if not p.exists():
            raise NotFound(f"Project {project_id} not found")
        project = Project.model_validate_json(p.read_text(encoding="utf-8"))
        if owner is not None and project.owner != owner:
            # Do not reveal existence of other users' projects.
            raise NotFound(f"Project {project_id} not found")
        return project

    @contextmanager
    def edit(self, project_id: str, *, owner: str | None = None) -> Iterator[Project]:
        with self.lock(project_id):
            project = self.load(project_id, owner=owner)
            yield project
            self.save(project)

    def list(self, *, owner: str | None = None) -> list[Project]:
        out: list[Project] = []
        if not self.settings.projects_dir.exists():
            return out
        for d in sorted(self.settings.projects_dir.iterdir()):
            f = d / "project.json"
            if f.exists():
                try:
                    pr = Project.model_validate_json(f.read_text(encoding="utf-8"))
                except Exception:  # noqa: S112 - skip unreadable/partial project files
                    continue
                if owner is None or pr.owner == owner:
                    out.append(pr)
        out.sort(key=lambda p: p.created_at, reverse=True)
        return out

    def delete(self, project_id: str, *, owner: str | None = None) -> None:
        self.load(project_id, owner=owner)
        shutil.rmtree(self.project_dir(project_id), ignore_errors=True)

    # ---- jobs --------------------------------------------------------------
    def save_job(self, job: Job) -> None:
        _atomic_write(self.settings.jobs_dir / f"{_check_id(job.id)}.json", job.model_dump_json(indent=2))

    def load_job(self, job_id: str) -> Job:
        f = self.settings.jobs_dir / f"{_check_id(job_id)}.json"
        if not f.exists():
            raise NotFound(f"Job {job_id} not found")
        return Job.model_validate_json(f.read_text(encoding="utf-8"))

    def write_json(self, project_id: str, rel: str, data: object) -> Path:
        p = self.path(project_id, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, json.dumps(data, indent=2))
        return p
