"""
Real-Time Speech Transcription Server — Whisper Edition (Ultra Fast & Simple)
=============================================================================
FastAPI WebSocket backend optimized for low-latency greedy transcription:
  - Local Whisper model support (tiny / base / small / medium / large-v2 / large-v3)
  - Hot-swappable model via REST API
  - Zero server-side VAD latency (fully handled by browser RMS gates)
  - Monotonic chronological timestamps via offset tracking
  - Greedy single-pass transcription
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
# REPETITION DETECTOR
# Catch and remove hallucinated repeating words.
# ─────────────────────────────────────────────────────────────────────────────

def detect_repetition(words: list[dict], max_ngram: int = 5) -> list[dict]:
    if len(words) < 4:
        return words

    texts = [w["text"].strip().lower() for w in words]
    keep = [True] * len(words)

    for n in range(2, min(max_ngram + 1, len(texts) // 2 + 1)):
        for i in range(len(texts) - 2 * n + 1):
            ngram1 = tuple(texts[i:i + n])
            ngram2 = tuple(texts[i + n:i + 2 * n])
            if ngram1 == ngram2:
                for j in range(i + n, i + 2 * n):
                    keep[j] = False

    return [w for w, k in zip(words, keep) if k]


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER PROVIDER
# Handles model loading and single-pass streaming transcription.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16000
MAX_BUFFER_MS = 30000

class WhisperProvider:
    def __init__(self):
        self._model      = None
        self._model_name = None
        self._lock       = threading.Lock()
        self._loading    = False
        self._load_error = None
        self._sessions: dict[str, dict] = {}

    def get_status(self) -> dict:
        return {
            "loaded_model": self._model_name,
            "loading":      self._loading,
            "error":        self._load_error,
            "models":       WHISPER_MODELS,
        }

    def load_model(self, model_name: str) -> dict:
        if model_name not in WHISPER_MODELS:
            return {"ok": False, "error": f"Unknown model '{model_name}'"}
        if self._loading:
            return {"ok": False, "error": "Another model is already loading"}

        self._loading    = True
        self._load_error = None

        def _do_load():
            try:
                from faster_whisper import WhisperModel
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

    def _get_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "pcm_f32":                np.array([], dtype=np.float32),
                "session_offset_ms":      0,
                "transcribe_in_progress": False,
                "websocket":              None,
                "send_lock":              None,
            }
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def process_chunk(
        self,
        pcm_int16: bytes,
        session_id: str,
        audio_cursor_ms: int,
        task: str = "transcribe",
        beam_size: int = 5,
        websocket: WebSocket = None,
        language: str = None,
    ) -> None:
        state = self._get_session(session_id)
        if "websocket" not in state or state["websocket"] is None:
            state["websocket"] = websocket
            state["send_lock"] = asyncio.Lock()

        # Convert Int16 → Float32
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        state["pcm_f32"] = np.concatenate([state["pcm_f32"], samples])

        # Limit rolling buffer to MAX_BUFFER_MS
        max_samples = int(SAMPLE_RATE * MAX_BUFFER_MS / 1000)
        if len(state["pcm_f32"]) > max_samples:
            drop = len(state["pcm_f32"]) - max_samples
            state["pcm_f32"] = state["pcm_f32"][drop:]
            state["session_offset_ms"] += int(drop / SAMPLE_RATE * 1000)

        # Trigger greedy transcription task
        if not state["transcribe_in_progress"]:
            state["transcribe_in_progress"] = True
            asyncio.create_task(self._run_transcription_task(session_id, audio_cursor_ms, task, beam_size, language))

    async def _run_transcription_task(self, session_id: str, audio_cursor_ms: int, task: str, beam_size: int, language: str = None):
        samples_processed = 0
        try:
            state = self._sessions.get(session_id)
            if not state:
                return

            audio = state["pcm_f32"]
            samples_processed = len(audio)
            if samples_processed < SAMPLE_RATE * 0.1:
                return

            with self._lock:
                model = self._model
            if model is None:
                return

            offset_ms = state["session_offset_ms"]

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._run_whisper_single_pass, audio.copy(), offset_ms, task, beam_size, False, language
            )

            if result and state["websocket"]:
                async with state["send_lock"]:
                    await state["websocket"].send_text(result.to_json())

        except Exception as e:
            print(f"[Whisper] Transcription task error: {e}")
        finally:
            state = self._sessions.get(session_id)
            if state:
                state["transcribe_in_progress"] = False
                # If more audio has arrived since we started, trigger another task!
                if len(state["pcm_f32"]) > samples_processed:
                    state["transcribe_in_progress"] = True
                    asyncio.create_task(self._run_transcription_task(session_id, audio_cursor_ms, task, beam_size, language))

    async def flush_session(self, session_id: str, task: str = "transcribe", beam_size: int = 5, language: str = None) -> None:
        state = self._sessions.get(session_id)
        if not state:
            return

        audio = state["pcm_f32"]
        if len(audio) < SAMPLE_RATE * 0.1:
            return

        with self._lock:
            model = self._model
        if model is None:
            return

        # Run final transcription with is_final = True
        offset_ms = state["session_offset_ms"]
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_whisper_single_pass, audio.copy(), offset_ms, task, beam_size, True, language
        )

        if result and state["websocket"]:
            async with state["send_lock"]:
                await state["websocket"].send_text(result.to_json())

        # Reset active sentence buffer & update session offset
        state["session_offset_ms"] += int(len(audio) / SAMPLE_RATE * 1000)
        state["pcm_f32"] = np.array([], dtype=np.float32)

    def _run_whisper_single_pass(
        self,
        audio: np.ndarray,
        offset_ms: int,
        task: str,
        beam_size: int,
        is_final: bool,
        language: str = None,
    ) -> Optional[TranscriptionResult]:
        try:
            with self._lock:
                model = self._model
            if model is None:
                return None

            transcribe_kwargs = dict(
                language=language,
                task=task,
                word_timestamps=True,
                vad_filter=False,
                beam_size=beam_size,
                without_timestamps=False,
                temperature=0.0,
                no_speech_threshold=None,
                log_prob_threshold=None,
                compression_ratio_threshold=None,
                condition_on_previous_text=True,
            )

            segments, info = model.transcribe(audio, **transcribe_kwargs)

            tokens = []
            for seg in segments:
                lang = info.language if info else "en"
                
                if hasattr(seg, 'avg_logprob') and seg.avg_logprob < -1.0:
                    continue
                if hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.6:
                    continue

                if seg.words:
                    for word in seg.words:
                        tokens.append(Token(
                            text       = word.word,
                            start_ms   = int(word.start * 1000) + offset_ms,
                            end_ms     = int(word.end * 1000) + offset_ms,
                            confidence = round(float(word.probability), 3),
                            is_final   = is_final,
                            speaker    = "1",
                            language   = lang,
                        ))
                else:
                    conf = round(float(getattr(seg, "avg_logprob", -0.3) + 1), 3)
                    tokens.append(Token(
                        text       = seg.text,
                        start_ms   = int(seg.start * 1000) + offset_ms,
                        end_ms     = int(seg.end * 1000) + offset_ms,
                        confidence = conf,
                        is_final   = is_final,
                        speaker    = "1",
                        language   = lang,
                    ))

            if not tokens:
                return None

            raw_words = [
                {
                    "text": t.text,
                    "start_ms": t.start_ms,
                    "end_ms": t.end_ms,
                    "confidence": t.confidence,
                    "language": t.language,
                }
                for t in tokens
            ]
            raw_words = detect_repetition(raw_words)
            
            tokens = [
                Token(
                    text=w["text"],
                    start_ms=w["start_ms"],
                    end_ms=w["end_ms"],
                    confidence=w["confidence"],
                    is_final=is_final,
                    speaker="1",
                    language=w["language"],
                )
                for w in raw_words
            ]

            final_audio_proc_ms = max((t.end_ms for t in tokens), default=offset_ms) if is_final else offset_ms

            return TranscriptionResult(
                tokens              = tokens,
                audio_offset_ms     = offset_ms,
                final_audio_proc_ms = final_audio_proc_ms,
                total_audio_proc_ms = offset_ms + int(len(audio) / SAMPLE_RATE * 1000),
            )

        except Exception as e:
            print(f"[Whisper] Single-pass error: {e}")
            return None

    async def flush(
        self,
        session_id: str,
        audio_cursor_ms: int,
        task: str = "transcribe",
        beam_size: int = 5,
        language: str = None,
    ) -> None:
        await self.flush_session(session_id, task, beam_size, language)


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
    return JSONResponse(provider.get_status())


@app.post("/api/models/{model_name}/load")
async def load_model(model_name: str):
    result = provider.load_model(model_name)
    return JSONResponse(result)


@app.websocket("/ws/transcribe")
async def transcribe_ws(
    websocket: WebSocket,
    translate: bool = False,
    beam_size: int = 5,
    language: str = None
):
    await websocket.accept()
    session = session_manager.create(websocket)
    sid     = session.session_id

    await websocket.send_text(json.dumps({
        "type":       "session_init",
        "session_id": sid,
    }))

    # Sanitize language parameter
    if language == "auto" or not language:
        language = None

    task = "translate" if translate else "transcribe"

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                data = message["bytes"]
                session.audio_cursor_ms += session.chunk_duration_ms
                await provider.process_chunk(
                    pcm_int16        = data,
                    session_id       = sid,
                    audio_cursor_ms  = session.audio_cursor_ms,
                    task             = task,
                    beam_size        = beam_size,
                    websocket        = websocket,
                    language         = language,
                )
            elif "text" in message:
                text_data = message["text"]
                try:
                    msg = json.loads(text_data)
                    if msg.get("type") == "flush":
                        await provider.flush_session(sid, task=task, beam_size=beam_size, language=language)
                except Exception:
                    pass

    except WebSocketDisconnect:
        await provider.flush(sid, session.audio_cursor_ms, task=task, beam_size=beam_size, language=language)
    finally:
        provider.reset(sid)
        session_manager.remove(sid)
