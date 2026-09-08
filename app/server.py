"""
server.py - local web app for making 8D audio.

Binds to 127.0.0.1 only. Files never leave the machine.
    ~/8d/.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import shutil
import secrets
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import engine

HERE = Path(__file__).parent
# Optional shared-secret gate. Unset (the local default) means no gate.
# Any exposed deployment MUST set this, or it is an open compute endpoint.
ACCESS_TOKEN = os.environ.get("EIGHTD_TOKEN", "").strip()
MAX_UPLOAD_MB = int(os.environ.get("EIGHTD_MAX_UPLOAD_MB", "0") or 0)
JOBS = HERE / "jobs"
JOBS.mkdir(exist_ok=True)

app = FastAPI(title="8D Audio")


@app.middleware("http")
async def gate(request, call_next):
    if ACCESS_TOKEN:
        supplied = (request.headers.get("x-access-token")
                    or request.query_params.get("t") or "")
        if not secrets.compare_digest(supplied, ACCESS_TOKEN):
            return JSONResponse({"detail": "Not authorized"}, status_code=401)
    return await call_next(request)

_state: dict[str, dict] = {}
_lock = threading.Lock()


def _set(jid: str, **kw):
    with _lock:
        _state.setdefault(jid, {}).update(kw)


def _get(jid: str) -> dict:
    with _lock:
        return dict(_state.get(jid, {}))


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex[:12]
    d = JOBS / uid
    d.mkdir(parents=True, exist_ok=True)
    dest = d / ("source" + Path(file.filename or "input").suffix.lower())

    written = 0
    cap = MAX_UPLOAD_MB * 1024 * 1024
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 22):
            written += len(chunk)
            if cap and written > cap:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(
                    413, f"File larger than the {MAX_UPLOAD_MB} MB limit set on this server.")
            f.write(chunk)

    try:
        info = engine.probe(str(dest))
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Could not read that file: {e}")

    if not info["audio"]:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "That file has no audio track.")

    # Browser-playable copy of the original, so A/B works even for .mov
    preview = None
    try:
        pv = d / "original.mp3"
        import subprocess as _sp
        _sp.run(["ffmpeg", "-v", "error", "-y", "-i", str(dest),
                 "-map", "0:a:0", "-vn", "-b:a", "192k", str(pv)],
                check=True, capture_output=True, timeout=300)
        if pv.exists():
            preview = pv.name
    except Exception:
        pass

    _set(uid, path=str(dest), info=info,
         stem=Path(file.filename or "audio").stem, stage="ready")
    return {"upload_id": uid, "name": file.filename,
            "source_name": dest.name, "preview": preview, **info}


def _run(jid: str, mode: str, period: float, bass: float,
         reverb: float, width: float, mux: bool):
    st = _get(jid)
    path, info, stem = st["path"], st["info"], st["stem"]

    def prog(p, msg):
        _set(jid, stage="working", progress=round(float(p), 4), message=msg)

    note = None
    try:
        prog(0.03, "Decoding audio")
        if mode == "ambisonic":
            try:
                b, fs = engine.read_ambisonic(path, info["ambisonic_slot"])
            except Exception as e:
                # Some files carry an APAC track that CoreAudio will not decode,
                # notably anything remuxed by ffmpeg, which drops the packet
                # metadata the decoder needs. Fall back rather than fail.
                note = ("Spatial track present but could not be decoded, so the "
                        "standard audio was used instead. This usually means the "
                        "file was re-encoded or remuxed after recording.")
                _set(jid, note=note)
                mode = "stereo"
            else:
                out = engine.process_ambisonic(b, fs, period=period,
                                               width=width, progress=prog)
        if mode == "stereo":
            a, fs = engine.read_stereo(path)
            out = engine.process_stereo(a, fs, period=period, bass=bass,
                                        reverb=reverb, progress=prog)

        prog(0.94, "Writing files")
        files = engine.write_outputs(
            out, fs, Path(path).parent, stem,
            source_video=path if (mux and info["has_video"]) else None,
            progress=prog)

        _set(jid, stage="done", progress=1.0, message="Done",
             outputs=[f.name for f in files],
             duration=len(out) / fs, mode=mode, note=note)
    except Exception as e:
        _set(jid, stage="error", message=str(e),
             detail=traceback.format_exc()[-1500:])


@app.post("/process")
def process(upload_id: str = Form(...), mode: str = Form("auto"),
            period: float = Form(12.0), bass: float = Form(120.0),
            reverb: float = Form(0.25), width: float = Form(1.0),
            mux: bool = Form(False)):
    st = _get(upload_id)
    if not st.get("path"):
        raise HTTPException(404, "Upload not found. Drop the file again.")

    if mode == "auto":
        mode = "ambisonic" if st["info"]["has_ambisonic"] else "stereo"
    if mode == "ambisonic" and not st["info"]["has_ambisonic"]:
        raise HTTPException(400, "This file has no spatial audio track.")

    period = max(2.0, min(60.0, period))
    bass = max(20.0, min(500.0, bass))
    reverb = max(0.0, min(1.0, reverb))
    width = max(0.0, min(1.5, width))

    _set(upload_id, stage="working", progress=0.0, message="Starting",
         outputs=None)
    threading.Thread(target=_run, args=(upload_id, mode, period, bass,
                                        reverb, width, mux),
                     daemon=True).start()
    return {"job_id": upload_id, "mode": mode}


@app.get("/status/{jid}")
def status(jid: str):
    st = _get(jid)
    if not st:
        raise HTTPException(404, "No such job")
    return {k: st.get(k) for k in
            ("stage", "progress", "message", "outputs", "duration", "mode",
             "detail", "note")}


@app.get("/file/{jid}/{name}")
def get_file(jid: str, name: str):
    # Resolve straight off disk. In-memory job state is lost on restart, but the
    # rendered files are not, so downloads keep working across restarts.
    if "/" in jid or "\\" in jid or ".." in jid:
        raise HTTPException(400, "Bad job id")
    jobdir = (JOBS / jid).resolve()
    if not jobdir.is_dir() or not jobdir.is_relative_to(JOBS.resolve()):
        raise HTTPException(404, "No such job")
    p = (jobdir / name).resolve()
    if not p.is_relative_to(jobdir) or not p.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(p, filename=name)


@app.delete("/job/{jid}")
def delete_job(jid: str):
    shutil.rmtree(JOBS / jid, ignore_errors=True)
    with _lock:
        _state.pop(jid, None)
    return {"ok": True}
