"""REST API (incl. auth/ownership) and MCP tools (in-memory MCP client)."""

from __future__ import annotations

import time
from dataclasses import replace

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp import Client

from audiolayer.api.app import create_app
from audiolayer.core.components import Components
from audiolayer.core.service import AudioLayerService
from audiolayer.mcp_server.server import build_server

from conftest import SCENE_SCHEDULE, OracleSeparator, ScheduleClassifier, wait_job

REQUIRED_TOOLS = {
    "analyze_media", "get_analysis", "list_audio_segments", "update_segment_category", "separate_audio_category",
    "remove_audio_category", "export_audio_stems", "export_cleaned_audio", "get_processing_status", "cancel_processing",
}


def _poll(client: TestClient, job_id: str, headers=None) -> dict:
    for _ in range(600):
        j = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise TimeoutError


def test_rest_flow(service, scene):
    app = create_app(service.settings, service)
    with TestClient(app) as c:
        assert c.get("/api/health").json() == {"ok": True}
        r = c.post("/api/projects", files={"file": ("bad.txt", b"hello")})
        assert r.status_code == 415 and r.json()["error"]["code"] == "invalid_media"
        r = c.post("/api/projects", files={"file": ("broken.mp4", b"\x00" * 300)})
        assert r.status_code == 415
        with open(scene["mix"], "rb") as f:
            r = c.post("/api/projects", files={"file": ("scene.wav", f, "audio/wav")}, data={"analyze": "true"})
        assert r.status_code == 200, r.text
        pid = r.json()["project"]["id"]
        assert _poll(c, r.json()["job"]["id"])["status"] == "succeeded"
        pr = c.get(f"/api/projects/{pid}").json()
        assert pr["status"] == "analyzed" and pr["stems_available"]
        peaks = c.get(f"/api/projects/{pid}/peaks").json()
        assert abs(peaks["duration"] - 10.0) < 0.05 and len(peaks["max"]) > 100
        seg = next(s for s in pr["segments"] if s["category"] == "dialogue")
        r = c.patch(f"/api/projects/{pid}/segments/{seg['id']}", json={"category": "sfx"})
        assert r.json()["segment"]["category"] == "sfx"
        _poll(c, r.json()["render_job"]["id"])
        r = c.post(f"/api/projects/{pid}/mix", json={"removed": ["bgm", "noise"]})
        assert r.json()["removed"] == ["bgm", "noise"]
        assert c.post(f"/api/projects/{pid}/undo").status_code == 200
        assert c.get(f"/api/projects/{pid}").json()["mix"]["removed"] == []
        assert c.post(f"/api/projects/{pid}/redo").status_code == 200
        j = _poll(c, c.post(f"/api/projects/{pid}/preview", json={}).json()["id"])
        assert j["status"] == "succeeded"
        r = c.get(f"/api/projects/{pid}/files/{j['result']['file']}", headers={"Range": "bytes=0-99"})
        assert r.status_code == 206 and len(r.content) == 100
        j = _poll(c, c.post(f"/api/projects/{pid}/export", json={"items": ["mix", "metadata"], "format": "flac"}).json()["id"])
        assert j["status"] == "succeeded"
        assert c.get(f"/api/projects/{pid}/files/{j['result']['zip']}").status_code == 200
        # path traversal & non-downloadable files are refused
        assert c.get(f"/api/projects/{pid}/files/project.json").status_code == 403
        assert c.get(f"/api/projects/{pid}/files/exports/../project.json").status_code in (403, 404)
        assert c.get("/api/projects/../../etc").status_code == 404


def test_auth_and_ownership(settings, scene):
    s = replace(settings, auth_tokens={"tok-a": "alice", "tok-b": "bob"})
    svc = AudioLayerService(s, Components(classifier=ScheduleClassifier(SCENE_SCHEDULE),
                                          separator=OracleSeparator(scene["voice"]), asr=None))
    try:
        with TestClient(create_app(s, svc)) as c:
            assert c.get("/api/projects").status_code == 401
            assert c.get("/api/projects", headers={"Authorization": "Bearer nope"}).status_code == 401
            A = {"Authorization": "Bearer tok-a"}
            B = {"Authorization": "Bearer tok-b"}
            with open(scene["mix"], "rb") as f:
                pid = c.post("/api/projects", files={"file": ("s.wav", f)}, headers=A).json()["project"]["id"]
            assert c.get(f"/api/projects/{pid}", headers=A).status_code == 200
            assert c.get(f"/api/projects/{pid}", headers=B).status_code == 404  # no existence leak
            assert c.get("/api/projects", headers=B).json() == []
            assert c.get(f"/api/projects/{pid}/media?token=tok-a").status_code == 200
    finally:
        svc.jobs.shutdown()


async def _mcp_flow(service, scene_path) -> None:
    server = build_server(service, remote=False)
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert REQUIRED_TOOLS <= set(tools)
        for name in REQUIRED_TOOLS:
            assert tools[name].description and tools[name].output_schema
        bad = await c.call_tool("analyze_media", {"file_path": "/etc/passwd"})
        assert bad.is_error and "forbidden" in bad.content[0].text
        bad = await c.call_tool("analyze_media", {})
        assert bad.is_error
        r = await c.call_tool("analyze_media", {"file_path": str(scene_path), "transcribe": False})
        assert not r.is_error, r.content
        pid, job = r.structured_content["project_id"], r.structured_content["job_id"]
        for _ in range(400):
            st = (await c.call_tool("get_processing_status", {"job_id": job})).structured_content
            if st["status"] not in ("queued", "running"):
                break
            await anyio.sleep(0.05)
        assert st["status"] == "succeeded"
        a = (await c.call_tool("get_analysis", {"project_id": pid})).structured_content
        assert a["segment_counts"]["dialogue"] == 1 and a["analysis"]["separated"]
        segs = (await c.call_tool("list_audio_segments", {"project_id": pid, "category": "dialogue"})).structured_content
        assert segs["total"] == 1
        sid = segs["segments"][0]["id"]
        up = await c.call_tool("update_segment_category", {"project_id": pid, "segment_id": sid, "category": "sfx"})
        assert up.structured_content["segment"]["category"] == "sfx"
        rem = await c.call_tool("remove_audio_category", {"project_id": pid, "categories": ["bgm", "music"]})
        assert rem.structured_content["removed_categories"] == ["bgm", "music"]
        ex = await c.call_tool("export_cleaned_audio", {"project_id": pid, "format": "wav"})
        job = ex.structured_content["job_id"]
        for _ in range(400):
            st = (await c.call_tool("get_processing_status", {"job_id": job})).structured_content
            if st["status"] not in ("queued", "running"):
                break
            await anyio.sleep(0.05)
        assert st["status"] == "succeeded"
        from pathlib import Path

        assert all(Path(f["path"]).exists() for f in st["files"])
        sep = await c.call_tool("separate_audio_category", {"project_id": pid, "category": "sfx"})
        assert sep.structured_content["kind"] == "export"
        cancel = await c.call_tool("cancel_processing", {"job_id": "job_doesnotexist"})
        assert cancel.is_error


def test_mcp_tools(service, scene):
    anyio.run(_mcp_flow, service, scene["mix"])


def test_mcp_remote_requires_token(settings, scene):
    import os

    s = replace(settings, auth_tokens={"tok-a": "alice"}, public_base_url="https://example.test")
    svc = AudioLayerService(s, Components(classifier=ScheduleClassifier(SCENE_SCHEDULE),
                                          separator=OracleSeparator(scene["voice"]), asr=None))
    os.environ["AUDIOLAYER_ENABLE_REMOTE_MCP"] = "true"
    try:
        with TestClient(create_app(s, svc), base_url="https://example.test") as c:
            init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
            hdr = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
            assert c.post("/mcp/", json=init, headers=hdr).status_code == 401
            r = c.post("/mcp/", json=init, headers={**hdr, "Authorization": "Bearer tok-a"})
            assert r.status_code == 200, r.text
            r = c.post("/mcp/", headers={**hdr, "Authorization": "Bearer tok-a"},
                       json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                             "params": {"name": "create_upload_url", "arguments": {}}})
            assert r.status_code == 200, r.text
            url = r.json()["result"]["structuredContent"]["upload_url"]
            assert url.startswith("https://example.test/api/uploads/")
            tok = url.rsplit("/", 1)[1]
            with open(scene["mix"], "rb") as f:
                up = c.post(f"/api/uploads/{tok}", files={"file": ("s.wav", f)})
            assert up.status_code == 200
            pid = up.json()["project_id"]
            assert svc.get(pid, "alice").owner == "alice"
            with open(scene["mix"], "rb") as f:  # single use
                assert c.post(f"/api/uploads/{tok}", files={"file": ("s.wav", f)}).status_code == 404
            r = c.post("/mcp/", headers={**hdr, "Authorization": "Bearer tok-a"},
                       json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                             "params": {"name": "analyze_media", "arguments": {"file_path": "/etc/passwd"}}})
            assert r.json()["result"]["isError"] is True
    finally:
        os.environ.pop("AUDIOLAYER_ENABLE_REMOTE_MCP", None)
        svc.jobs.shutdown()
    _ = wait_job
    _ = pytest
