"""
Real-Time Speech Transcription Server — Whisper Edition
=========================================================
FastAPI WebSocket backend with:
  - Local Whisper model support (tiny / base / small / medium / large-v2 / large-v3)
  - Hot-swappable model via REST API (no server restart needed)
  - Per-session 3-second rolling audio buffer for real streaming feel
  - Token aggregation with interim → final lifecycle
  - Exact response schema compliance
"""

import asyncio
import json
import time
import uuid
import threading
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import deque
import pathlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

# ─────────────────────────────────────────────────────────────────────────────
# WHISPER MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_MODELS = {
    "tiny":     {"size": "~75 MB",  "speed": "~32x", "quality": "★☆☆☆☆"},
    "base":     {"size": "~142 MB", "speed": "~16x", "quality": "★★☆☆☆"},
    "small":    {"size": "~466 MB", "speed": "~6x",  "quality": "★★★☆☆"},
    "medium":   {"size": "~1.5 GB", "speed": "~2x",  "quality": "★★★★☆"},
    "large-v2": {"size": "~3.1 GB", "speed": "~1x",  "quality": "★★★★★"},
    "large-v3": {"size": "~3.1 GB", "speed": "~1x",  "quality": "★★★★★"},
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    text: str
    start_ms: int
    end_ms: int
    confidence: float
    is_final: bool
    speaker: str
    language: str

@dataclass
class TranscriptionResult:
    tokens: list
    audio_offset_ms: int
    final_audio_proc_ms: int
    total_audio_proc_ms: int

    def to_json(self) -> str:
        return json.dumps([{
            "tokens": [asdict(t) for t in self.tokens],
            "audio_offset_ms": self.audio_offset_ms,
            "final_audio_proc_ms": self.final_audio_proc_ms,
            "total_audio_proc_ms": self.total_audio_proc_ms,
        }])


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER PROVIDER
# Uses faster-whisper for low-latency streaming transcription.
# Audio is accumulated in a rolling buffer; transcription runs every
# CHUNK_TRIGGER_MS of new audio, returning tokens with timestamps.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16000
CHUNK_TRIGGER_MS   = 3000   # Run Whisper every 3 s of buffered audio
MAX_BUFFER_MS      = 30000  # Keep max 30 s of rolling context
OVERLAP_MS         = 1000   # Overlap between windows to avoid cut-off words

class WhisperProvider:
    def __init__(self):
        self._model      = None
        self._model_name = None
        self._lock       = threading.Lock()
        self._loading    = False
        self._load_error = None

        # Per-session state
        self._sessions: dict[str, dict] = {}

    # ── Model Management ────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "loaded_model": self._model_name,
            "loading":      self._loading,
            "error":        self._load_error,
            "models":       WHISPER_MODELS,
        }

    def load_model(self, model_name: str) -> dict:
        """Load a Whisper model in a background thread. Non-blocking."""
        if model_name not in WHISPER_MODELS:
            return {"ok": False, "error": f"Unknown model '{model_name}'"}
        if self._loading:
            return {"ok": False, "error": "Another model is already loading"}

        self._loading    = True
        self._load_error = None

        def _do_load():
            try:
                from faster_whisper import WhisperModel
                # Use GPU if available, fall back to CPU
                try:
                    import ctranslate2
                    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                except ImportError:
                    device = "cpu"

                compute = "float16" if device == "cuda" else "int8"
                print(f"[Whisper] Loading '{model_name}' on {device} ({compute})…")
                new_model = WhisperModel(model_name, device=device, compute_type=compute)
                with self._lock:
                    self._model      = new_model
                    self._model_name = model_name
                print(f"[Whisper] Model '{model_name}' ready.")
            except Exception as e:
                self._load_error = str(e)
                print(f"[Whisper] Load error: {e}")
            finally:
                self._loading = False

        t = threading.Thread(target=_do_load, daemon=True)
        t.start()
        return {"ok": True, "message": f"Loading '{model_name}'…"}

    # ── Session helpers ──────────────────────────────────────────────────

    def _get_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "pcm_f32":       np.array([], dtype=np.float32),
                "audio_offset":  0,       # ms offset of buffer[0] in the stream
                "last_run_ms":   0,       # audio_cursor_ms when last transcription ran
                "pending_tokens": deque(),
                "final_cursor":  0,
                "token_seq":     0,
            }
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ── Core processing ──────────────────────────────────────────────────

    async def process_chunk(
        self,
        pcm_int16: bytes,
        session_id: str,
        audio_cursor_ms: int,
        task: str = "transcribe",
        beam_size: int = 5,
    ) -> Optional[TranscriptionResult]:
        state = self._get_session(session_id)

        # Convert Int16 → Float32 normalized [-1, 1]
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        state["pcm_f32"] = np.concatenate([state["pcm_f32"], samples])

        # Cap rolling buffer to MAX_BUFFER_MS
        max_samples = int(SAMPLE_RATE * MAX_BUFFER_MS / 1000)
        if len(state["pcm_f32"]) > max_samples:
            drop = len(state["pcm_f32"]) - max_samples
            state["pcm_f32"] = state["pcm_f32"][drop:]
            state["audio_offset"] += int(drop / SAMPLE_RATE * 1000)

        # Only run Whisper every CHUNK_TRIGGER_MS of new audio
        since_last = audio_cursor_ms - state["last_run_ms"]
        if since_last < CHUNK_TRIGGER_MS:
            return None

        with self._lock:
            model = self._model

        if model is None:
            return None  # Model not loaded yet

        state["last_run_ms"] = audio_cursor_ms

        # Run Whisper in a thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_whisper, state["pcm_f32"].copy(), state, audio_cursor_ms, task, beam_size
        )
        return result

    def _run_whisper(
        self,
        audio: np.ndarray,
        state: dict,
        audio_cursor_ms: int,
        task: str = "transcribe",
        beam_size: int = 5,
    ) -> Optional[TranscriptionResult]:
        try:
            with self._lock:
                model = self._model
            if model is None:
                return None

            segments, info = model.transcribe(
                audio,
                language=None,               # auto-detect
                task=task,
                word_timestamps=True,        # essential for token-level output
                vad_filter=True,             # skip silence
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=beam_size,
                without_timestamps=False,
            )

            offset_ms = state["audio_offset"]
            tokens    = []

            for seg in segments:
                lang = info.language if info else "en"
                if seg.words:
                    for word in seg.words:
                        # Mark as final if the word is >1 s before current cursor
                        word_end_ms = int(word.end * 1000) + offset_ms
                        is_final    = (audio_cursor_ms - word_end_ms) > 1000
                        tok = Token(
                            text        = word.word,  # faster-whisper preserves leading spaces
                            start_ms    = int(word.start * 1000) + offset_ms,
                            end_ms      = word_end_ms,
                            confidence  = round(float(word.probability), 3),
                            is_final    = is_final,
                            speaker     = "1",
                            language    = lang,
                        )
                        tokens.append(tok)
                else:
                    # No word timestamps — emit segment as single token
                    tok = Token(
                        text        = seg.text,
                        start_ms    = int(seg.start * 1000) + offset_ms,
                        end_ms      = int(seg.end   * 1000) + offset_ms,
                        confidence  = round(float(getattr(seg, "avg_logprob", -0.3) + 1), 3),
                        is_final    = True,
                        speaker     = "1",
                        language    = lang,
                    )
                    tokens.append(tok)

            if not tokens:
                return None

            final_cursor = max(
                (t.end_ms for t in tokens if t.is_final),
                default=state["final_cursor"]
            )
            state["final_cursor"] = max(state["final_cursor"], final_cursor)

            return TranscriptionResult(
                tokens              = tokens,
                audio_offset_ms     = offset_ms,
                final_audio_proc_ms = state["final_cursor"],
                total_audio_proc_ms = audio_cursor_ms,
            )

        except Exception as e:
            print(f"[Whisper] Transcription error: {e}")
            return None

    async def flush(
        self,
        session_id: str,
        audio_cursor_ms: int,
        task: str = "transcribe",
        beam_size: int = 5,
    ) -> Optional[TranscriptionResult]:
        """Transcribe any remaining audio and mark all tokens final."""
        state = self._get_session(session_id)
        if len(state["pcm_f32"]) < SAMPLE_RATE * 0.5:
            return None

        with self._lock:
            model = self._model
        if model is None:
            return None

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_whisper, state["pcm_f32"].copy(), state, audio_cursor_ms, task, beam_size
        )
        if result:
            for t in result.tokens:
                t.is_final = True
            result.final_audio_proc_ms = audio_cursor_ms
        return result


# ─────────────────────────────────────────────────────────────────────────────
# SESSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id:       str
    websocket:        WebSocket
    audio_cursor_ms:  int   = 0
    chunk_duration_ms: int  = 60
    created_at:       float = field(default_factory=time.time)

class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self, ws: WebSocket) -> Session:
        sid     = str(uuid.uuid4())
        session = Session(session_id=sid, websocket=ws)
        self._sessions[sid] = session
        return session

    def remove(self, sid: str) -> None:
        self._sessions.pop(sid, None)

    def get(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app             = FastAPI(title="VoxStream — Whisper Transcription")
session_manager = SessionManager()
provider        = WhisperProvider()

BASE_DIR = pathlib.Path(__file__).parent
with open(BASE_DIR / "index.html", "r", encoding="utf-8") as f:
    HTML_CONTENT = f.read()


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML_CONTENT)


@app.get("/api/models")
async def get_models():
    """Return model list + current load status."""
    return JSONResponse(provider.get_status())


@app.post("/api/models/{model_name}/load")
async def load_model(model_name: str):
    """Trigger background load of a Whisper model."""
    result = provider.load_model(model_name)
    return JSONResponse(result)


@app.websocket("/ws/transcribe")
async def transcribe_ws(
    websocket: WebSocket,
    translate: bool = False,
    beam_size: int = 5
):
    await websocket.accept()
    session = session_manager.create(websocket)
    sid     = session.session_id

    await websocket.send_text(json.dumps({
        "type":       "session_init",
        "session_id": sid,
    }))

    task = "translate" if translate else "transcribe"

    try:
        while True:
            data = await websocket.receive_bytes()
            session.audio_cursor_ms += session.chunk_duration_ms

            result = await provider.process_chunk(
                pcm_int16        = data,
                session_id       = sid,
                audio_cursor_ms  = session.audio_cursor_ms,
                task             = task,
                beam_size        = beam_size,
            )
            if result:
                await websocket.send_text(result.to_json())

    except WebSocketDisconnect:
        flush = await provider.flush(sid, session.audio_cursor_ms, task=task, beam_size=beam_size)
        if flush:
            try:
                await websocket.send_text(flush.to_json())
            except Exception:
                pass
    finally:
        provider.reset(sid)
        session_manager.remove(sid)
