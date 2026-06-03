"""
Real-Time Speech Transcription Server — Whisper Edition
=========================================================
FastAPI WebSocket backend with:
  - Local Whisper model support (tiny / base / small / medium / large-v2 / large-v3)
  - Hot-swappable model via REST API (no server restart needed)
  - Per-session rolling audio buffer for real streaming feel
  - Voice Activity Detection (client-side RMS + server-side Silero VAD)
  - Word consensus finalization for improved accuracy
  - Prompt conditioning for context-aware transcription
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
import re

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
# VOICE ACTIVITY DETECTION (Silero VAD)
# Server-side VAD as a secondary filter. The client already does RMS-based
# silence gating, but this catches subtler cases and prevents Whisper from
# hallucinating on near-silence audio.
# ─────────────────────────────────────────────────────────────────────────────

class SileroVAD:
    """Lightweight wrapper around Silero VAD model for speech detection."""

    def __init__(self):
        self._model = None
        self._utils = None
        self._available = False
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            self._available = True
            print("[VAD] Silero VAD loaded successfully")
        except Exception as e:
            print(f"[VAD] Silero VAD not available ({e}), using energy-based fallback")
            self._available = False

    def contains_speech(self, audio: np.ndarray, sample_rate: int = 16000,
                        threshold: float = 0.35) -> bool:
        """
        Returns True if the audio chunk contains speech.
        Uses Silero VAD if available, otherwise falls back to energy-based detection.
        """
        if len(audio) < 512:
            return False

        # Energy-based pre-filter: skip obviously silent audio
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.005:
            return False

        if self._available:
            try:
                import torch
                with self._lock:
                    # Silero VAD expects 16kHz, mono, float32
                    tensor = torch.from_numpy(audio).float()
                    # Process in 512-sample windows (Silero's native chunk size)
                    window_size = 512
                    max_prob = 0.0
                    for i in range(0, len(tensor) - window_size + 1, window_size):
                        chunk = tensor[i:i + window_size]
                        prob = self._model(chunk, sample_rate).item()
                        max_prob = max(max_prob, prob)
                        if max_prob >= threshold:
                            return True
                    return max_prob >= threshold
            except Exception:
                pass

        # Fallback: energy-based detection
        return rms > 0.01


# ─────────────────────────────────────────────────────────────────────────────
# WORD CONSENSUS TRACKER
# Tracks words across multiple transcription passes. A word becomes "final"
# only after it appears consistently (same text at approximately the same
# timestamp) across CONSENSUS_COUNT consecutive passes.
# ─────────────────────────────────────────────────────────────────────────────

CONSENSUS_COUNT = 2       # Number of passes a word must survive to be finalized
TIME_TOLERANCE_MS = 500   # Max timestamp drift to consider words as "the same"
MIN_CONFIDENCE = 0.25     # Words below this confidence are dropped entirely

class WordConsensusTracker:
    """
    Stabilizes transcription output by requiring words to appear consistently
    across multiple Whisper passes before being marked as final.
    """

    def __init__(self):
        # Each entry: {normalized_key: {"text": str, "start_ms": int, "end_ms": int,
        #              "confidence": float, "count": int, "language": str}}
        self._candidates: dict[str, dict] = {}
        self._finalized: list[dict] = []  # Chronologically ordered final words
        self._finalized_end_ms: int = 0   # End time of last finalized word

    def _normalize_key(self, text: str, start_ms: int) -> str:
        """Create a fuzzy key for matching words across passes."""
        clean = text.strip().lower()
        # Bucket timestamps into TIME_TOLERANCE_MS windows
        bucket = start_ms // TIME_TOLERANCE_MS
        return f"{clean}@{bucket}"

    def _alt_keys(self, text: str, start_ms: int) -> list[str]:
        """Generate alternative keys to handle timestamp jitter."""
        clean = text.strip().lower()
        bucket = start_ms // TIME_TOLERANCE_MS
        return [
            f"{clean}@{bucket - 1}",
            f"{clean}@{bucket}",
            f"{clean}@{bucket + 1}",
        ]

    def update(self, words: list[dict]) -> list[dict]:
        """
        Process a new batch of words from Whisper. Returns the stabilized word
        list with accurate is_final flags.
        """
        new_candidates = {}

        for w in words:
            text = w["text"]
            start_ms = w["start_ms"]
            end_ms = w["end_ms"]
            confidence = w["confidence"]
            language = w.get("language", "en")

            # Skip low-confidence noise
            if confidence < MIN_CONFIDENCE:
                continue

            # Skip words that are already in finalized region
            if end_ms <= self._finalized_end_ms:
                continue

            primary_key = self._normalize_key(text, start_ms)

            # Check if this word existed in previous candidates
            prev_count = 0
            matched_key = None
            for alt_key in self._alt_keys(text, start_ms):
                if alt_key in self._candidates:
                    prev = self._candidates[alt_key]
                    prev_count = max(prev_count, prev["count"])
                    matched_key = alt_key

            new_count = prev_count + 1

            new_candidates[primary_key] = {
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "confidence": confidence,
                "count": new_count,
                "language": language,
            }

        self._candidates = new_candidates

        # Build output: finalized words + current candidates
        result = []

        # Add previously finalized words
        for fw in self._finalized:
            result.append({
                "text": fw["text"],
                "start_ms": fw["start_ms"],
                "end_ms": fw["end_ms"],
                "confidence": fw["confidence"],
                "is_final": True,
                "language": fw["language"],
            })

        # Process current candidates
        newly_finalized = []
        interim_words = []

        for key, cand in sorted(self._candidates.items(), key=lambda x: x[1]["start_ms"]):
            if cand["count"] >= CONSENSUS_COUNT:
                newly_finalized.append(cand)
            else:
                interim_words.append(cand)

        # Finalize words that have reached consensus
        for nf in newly_finalized:
            if nf["end_ms"] > self._finalized_end_ms:
                self._finalized.append(nf)
                self._finalized_end_ms = nf["end_ms"]
                result.append({
                    "text": nf["text"],
                    "start_ms": nf["start_ms"],
                    "end_ms": nf["end_ms"],
                    "confidence": nf["confidence"],
                    "is_final": True,
                    "language": nf["language"],
                })

        # Add interim (not yet confirmed) words
        for iw in interim_words:
            result.append({
                "text": iw["text"],
                "start_ms": iw["start_ms"],
                "end_ms": iw["end_ms"],
                "confidence": iw["confidence"],
                "is_final": False,
                "language": iw["language"],
            })

        return result

    def get_prompt_context(self, max_words: int = 20) -> str:
        """
        Returns the last N finalized words as a prompt string.
        Used to condition Whisper for better continuity.
        """
        if not self._finalized:
            return ""
        recent = self._finalized[-max_words:]
        return "".join(w["text"] for w in recent).strip()

    def force_finalize_all(self) -> list[dict]:
        """Mark all remaining candidates as final (used on flush)."""
        result = []
        for fw in self._finalized:
            result.append({
                "text": fw["text"],
                "start_ms": fw["start_ms"],
                "end_ms": fw["end_ms"],
                "confidence": fw["confidence"],
                "is_final": True,
                "language": fw["language"],
            })

        for key, cand in sorted(self._candidates.items(), key=lambda x: x[1]["start_ms"]):
            if cand["confidence"] >= MIN_CONFIDENCE and cand["end_ms"] > self._finalized_end_ms:
                result.append({
                    "text": cand["text"],
                    "start_ms": cand["start_ms"],
                    "end_ms": cand["end_ms"],
                    "confidence": cand["confidence"],
                    "is_final": True,
                    "language": cand["language"],
                })

        self._candidates.clear()
        return result

    def get_finalized_copy(self) -> list[dict]:
        """Return a copy of the finalized words list."""
        return [dict(fw) for fw in self._finalized]

    def reset(self):
        self._candidates.clear()
        self._finalized.clear()
        self._finalized_end_ms = 0


# ─────────────────────────────────────────────────────────────────────────────
# REPETITION DETECTOR
# Whisper sometimes hallucinates repeating phrases. This catches and removes
# obvious repetition patterns.
# ─────────────────────────────────────────────────────────────────────────────

def detect_repetition(words: list[dict], max_ngram: int = 5) -> list[dict]:
    """Remove repeated n-gram sequences from word list."""
    if len(words) < 4:
        return words

    texts = [w["text"].strip().lower() for w in words]
    keep = [True] * len(words)

    for n in range(2, min(max_ngram + 1, len(texts) // 2 + 1)):
        for i in range(len(texts) - 2 * n + 1):
            ngram1 = tuple(texts[i:i + n])
            ngram2 = tuple(texts[i + n:i + 2 * n])
            if ngram1 == ngram2:
                # Mark the repeated copy for removal
                for j in range(i + n, i + 2 * n):
                    keep[j] = False

    return [w for w, k in zip(words, keep) if k]


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER PROVIDER
# Uses faster-whisper for low-latency streaming transcription.
# Audio is accumulated in a rolling buffer; transcription runs every
# CHUNK_TRIGGER_MS of new audio, returning tokens with timestamps.
# Now with:
#   - Silero VAD for server-side silence detection
#   - Word consensus for finalization accuracy
#   - Prompt conditioning for contextual continuity
#   - Repetition detection
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE         = 16000
FAST_TRIGGER_MS     = 500    # Run fast Whisper pass every 500ms
ACCURATE_TRIGGER_MS = 3000   # Run accurate Whisper pass every 3s
MAX_BUFFER_MS       = 30000  # Keep max 30 s of rolling context
OVERLAP_MS          = 1000   # Overlap between windows to avoid cut-off words

class WhisperProvider:
    def __init__(self):
        self._model      = None
        self._model_name = None
        self._lock       = threading.Lock()
        self._loading    = False
        self._load_error = None
        self._vad        = SileroVAD()

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
                "pcm_f32":        np.array([], dtype=np.float32),
                "audio_offset":   0,       # ms offset of buffer[0] in the stream
                "last_fast_ms":   0,       # audio_cursor_ms when last fast pass ran
                "last_accurate_ms": 0,     # audio_cursor_ms when last accurate pass ran
                "fast_in_progress": False,
                "accurate_in_progress": False,
                "websocket":      None,
                "send_lock":      None,
                "pending_tokens": deque(),
                "final_cursor":   0,
                "token_seq":      0,
                "consensus":      WordConsensusTracker(),
                "silence_frames": 0,       # Consecutive silent frames counter
                "speech_active":  False,    # Whether speech is currently detected
            }
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["consensus"].reset()
        self._sessions.pop(session_id, None)

    # ── Core processing ──────────────────────────────────────────────────

    async def process_chunk(
        self,
        pcm_int16: bytes,
        session_id: str,
        audio_cursor_ms: int,
        task: str = "transcribe",
        beam_size: int = 5,
        websocket: WebSocket = None,
    ) -> None:
        state = self._get_session(session_id)
        if "websocket" not in state or state["websocket"] is None:
            state["websocket"] = websocket
            state["send_lock"] = asyncio.Lock()

        # Convert Int16 → Float32 normalized [-1, 1]
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0

        # Server-side VAD: check if chunk contains speech
        has_speech = self._vad.contains_speech(samples)

        if has_speech:
            state["silence_frames"] = 0
            state["speech_active"] = True
        else:
            state["silence_frames"] += 1
            # After 15 consecutive silent frames (~900ms), stop treating as speech
            if state["silence_frames"] > 15:
                state["speech_active"] = False

        # Always accumulate audio (needed for context)
        state["pcm_f32"] = np.concatenate([state["pcm_f32"], samples])

        # Cap rolling buffer to MAX_BUFFER_MS
        max_samples = int(SAMPLE_RATE * MAX_BUFFER_MS / 1000)
        if len(state["pcm_f32"]) > max_samples:
            drop = len(state["pcm_f32"]) - max_samples
            state["pcm_f32"] = state["pcm_f32"][drop:]
            state["audio_offset"] += int(drop / SAMPLE_RATE * 1000)

        # Skip Whisper if we haven't detected speech recently
        if not state["speech_active"]:
            state["last_fast_ms"] = audio_cursor_ms
            state["last_accurate_ms"] = audio_cursor_ms
            return

        # Check if fast pass should run
        since_last_fast = audio_cursor_ms - state["last_fast_ms"]
        if since_last_fast >= FAST_TRIGGER_MS and not state["fast_in_progress"]:
            state["last_fast_ms"] = audio_cursor_ms
            state["fast_in_progress"] = True
            asyncio.create_task(self._run_fast_pass_task(session_id, audio_cursor_ms, task))

        # Check if accurate pass should run
        since_last_accurate = audio_cursor_ms - state["last_accurate_ms"]
        if since_last_accurate >= ACCURATE_TRIGGER_MS and not state["accurate_in_progress"]:
            state["last_accurate_ms"] = audio_cursor_ms
            state["accurate_in_progress"] = True
            asyncio.create_task(self._run_accurate_pass_task(session_id, audio_cursor_ms, task, beam_size))

    async def _run_fast_pass_task(self, session_id: str, audio_cursor_ms: int, task: str):
        try:
            state = self._sessions.get(session_id)
            if not state or not state["speech_active"]:
                return

            offset_ms = state["audio_offset"]
            final_cursor = state["final_cursor"]
            
            # Start from final_cursor - OVERLAP_MS to keep context
            start_ms = max(offset_ms, final_cursor - OVERLAP_MS)
            start_sample = int((start_ms - offset_ms) * SAMPLE_RATE / 1000)
            
            audio = state["pcm_f32"]
            if start_sample >= len(audio):
                return
                
            audio_slice = audio[start_sample:].copy()
            if len(audio_slice) < SAMPLE_RATE * 0.1:  # at least 100ms
                return

            with self._lock:
                model = self._model
            if model is None:
                return

            prompt_context = state["consensus"].get_prompt_context()
            finalized_words = state["consensus"].get_finalized_copy()

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._run_whisper_fast, audio_slice, start_ms, task, prompt_context, finalized_words, final_cursor
            )

            if result and state["websocket"]:
                async with state["send_lock"]:
                    await state["websocket"].send_text(result.to_json())
                    
        except Exception as e:
            print(f"[Whisper] Fast pass error: {e}")
        finally:
            state = self._sessions.get(session_id)
            if state:
                state["fast_in_progress"] = False

    def _run_whisper_fast(
        self,
        audio: np.ndarray,
        start_ms: int,
        task: str,
        prompt_context: str,
        finalized_words: list[dict],
        final_cursor: int,
    ) -> Optional[TranscriptionResult]:
        try:
            with self._lock:
                model = self._model
            if model is None:
                return None

            transcribe_kwargs = dict(
                language=None,
                task=task,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 200,
                    "threshold": 0.35,
                },
                beam_size=1,  # Greedy decoding for maximum speed
                without_timestamps=False,
                no_speech_threshold=0.5,
                log_prob_threshold=-0.8,
                condition_on_previous_text=True,
                compression_ratio_threshold=2.2,
            )

            if prompt_context:
                transcribe_kwargs["initial_prompt"] = prompt_context

            segments, info = model.transcribe(audio, **transcribe_kwargs)

            raw_words = []
            for seg in segments:
                lang = info.language if info else "en"
                
                if hasattr(seg, 'avg_logprob') and seg.avg_logprob < -1.0:
                    continue
                if hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.6:
                    continue

                if seg.words:
                    for word in seg.words:
                        word_start_ms = int(word.start * 1000) + start_ms
                        word_end_ms = int(word.end * 1000) + start_ms
                        raw_words.append({
                            "text": word.word,
                            "start_ms": word_start_ms,
                            "end_ms": word_end_ms,
                            "confidence": round(float(word.probability), 3),
                            "language": lang,
                        })
                else:
                    seg_start_ms = int(seg.start * 1000) + start_ms
                    seg_end_ms = int(seg.end * 1000) + start_ms
                    conf = round(float(getattr(seg, "avg_logprob", -0.3) + 1), 3)
                    if conf > MIN_CONFIDENCE:
                        raw_words.append({
                            "text": seg.text,
                            "start_ms": seg_start_ms,
                            "end_ms": seg_end_ms,
                            "confidence": conf,
                            "language": lang,
                        })

            if not raw_words:
                raw_words = []

            raw_words = detect_repetition(raw_words)

            tokens = []
            for fw in finalized_words:
                tokens.append(Token(
                    text=fw["text"],
                    start_ms=fw["start_ms"],
                    end_ms=fw["end_ms"],
                    confidence=fw["confidence"],
                    is_final=True,
                    speaker="1",
                    language=fw["language"]
                ))
                
            for w in raw_words:
                if w["end_ms"] > final_cursor:
                    tokens.append(Token(
                        text=w["text"],
                        start_ms=w["start_ms"],
                        end_ms=w["end_ms"],
                        confidence=w["confidence"],
                        is_final=False,
                        speaker="1",
                        language=w["language"]
                    ))

            return TranscriptionResult(
                tokens              = tokens,
                audio_offset_ms     = start_ms,
                final_audio_proc_ms = final_cursor,
                total_audio_proc_ms = start_ms + int(len(audio) / SAMPLE_RATE * 1000),
            )

        except Exception as e:
            print(f"[Whisper] Fast backend error: {e}")
            return None

    async def _run_accurate_pass_task(self, session_id: str, audio_cursor_ms: int, task: str, beam_size: int):
        try:
            state = self._sessions.get(session_id)
            if not state or not state["speech_active"]:
                return

            offset_ms = state["audio_offset"]
            final_cursor = state["final_cursor"]
            
            start_ms = max(offset_ms, final_cursor - OVERLAP_MS)
            start_sample = int((start_ms - offset_ms) * SAMPLE_RATE / 1000)
            
            audio = state["pcm_f32"]
            if start_sample >= len(audio):
                return
                
            audio_slice = audio[start_sample:].copy()
            if len(audio_slice) < SAMPLE_RATE * 0.5:  # at least 500ms for accurate pass
                return

            with self._lock:
                model = self._model
            if model is None:
                return

            prompt_context = state["consensus"].get_prompt_context()

            loop = asyncio.get_event_loop()
            raw_words = await loop.run_in_executor(
                None, self._run_whisper_accurate_backend, audio_slice, start_ms, task, beam_size, prompt_context
            )

            if raw_words is not None:
                consensus = state["consensus"]
                stabilized_words = consensus.update(raw_words)
                
                tokens = []
                for w in stabilized_words:
                    tokens.append(Token(
                        text       = w["text"],
                        start_ms   = w["start_ms"],
                        end_ms     = w["end_ms"],
                        confidence = w["confidence"],
                        is_final   = w["is_final"],
                        speaker    = "1",
                        language   = w["language"],
                    ))

                new_final_cursor = max(
                    (t.end_ms for t in tokens if t.is_final),
                    default=state["final_cursor"]
                )
                state["final_cursor"] = max(state["final_cursor"], new_final_cursor)

                result = TranscriptionResult(
                    tokens              = tokens,
                    audio_offset_ms     = offset_ms,
                    final_audio_proc_ms = state["final_cursor"],
                    total_audio_proc_ms = audio_cursor_ms,
                )

                if state["websocket"]:
                    async with state["send_lock"]:
                        await state["websocket"].send_text(result.to_json())

        except Exception as e:
            print(f"[Whisper] Accurate pass error: {e}")
        finally:
            state = self._sessions.get(session_id)
            if state:
                state["accurate_in_progress"] = False

    def _run_whisper_accurate_backend(
        self,
        audio: np.ndarray,
        start_ms: int,
        task: str,
        beam_size: int,
        prompt_context: str = "",
    ) -> Optional[list[dict]]:
        try:
            with self._lock:
                model = self._model
            if model is None:
                return None

            transcribe_kwargs = dict(
                language=None,
                task=task,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 200,
                    "threshold": 0.35,
                },
                beam_size=beam_size,
                without_timestamps=False,
                no_speech_threshold=0.5,
                log_prob_threshold=-0.8,
                condition_on_previous_text=True,
                compression_ratio_threshold=2.2,
            )

            if prompt_context:
                transcribe_kwargs["initial_prompt"] = prompt_context

            segments, info = model.transcribe(audio, **transcribe_kwargs)

            raw_words = []
            for seg in segments:
                lang = info.language if info else "en"

                if hasattr(seg, 'avg_logprob') and seg.avg_logprob < -1.0:
                    continue
                if hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.6:
                    continue

                if seg.words:
                    for word in seg.words:
                        word_start_ms = int(word.start * 1000) + start_ms
                        word_end_ms = int(word.end * 1000) + start_ms
                        raw_words.append({
                            "text": word.word,
                            "start_ms": word_start_ms,
                            "end_ms": word_end_ms,
                            "confidence": round(float(word.probability), 3),
                            "language": lang,
                        })
                else:
                    seg_start_ms = int(seg.start * 1000) + start_ms
                    seg_end_ms = int(seg.end * 1000) + start_ms
                    conf = round(float(getattr(seg, "avg_logprob", -0.3) + 1), 3)
                    if conf > MIN_CONFIDENCE:
                        raw_words.append({
                            "text": seg.text,
                            "start_ms": seg_start_ms,
                            "end_ms": seg_end_ms,
                            "confidence": conf,
                            "language": lang,
                        })

            if not raw_words:
                return []

            return detect_repetition(raw_words)

        except Exception as e:
            print(f"[Whisper] Accurate backend error: {e}")
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
            consensus = state["consensus"]
            final_words = consensus.force_finalize_all()
            if final_words:
                tokens = [
                    Token(
                        text=w["text"], start_ms=w["start_ms"], end_ms=w["end_ms"],
                        confidence=w["confidence"], is_final=True, speaker="1",
                        language=w["language"],
                    )
                    for w in final_words
                ]
                return TranscriptionResult(
                    tokens=tokens,
                    audio_offset_ms=state["audio_offset"],
                    final_audio_proc_ms=audio_cursor_ms,
                    total_audio_proc_ms=audio_cursor_ms,
                )
            return None

        offset_ms = state["audio_offset"]
        final_cursor = state["final_cursor"]
        start_ms = max(offset_ms, final_cursor - OVERLAP_MS)
        start_sample = int((start_ms - offset_ms) * SAMPLE_RATE / 1000)

        audio_slice = state["pcm_f32"][start_sample:].copy()
        prompt_context = state["consensus"].get_prompt_context()

        loop   = asyncio.get_event_loop()
        raw_words = await loop.run_in_executor(
            None, self._run_whisper_accurate_backend, audio_slice, start_ms, task, beam_size, prompt_context
        )

        if raw_words is not None:
            state["consensus"].update(raw_words)

        consensus = state["consensus"]
        final_words = consensus.force_finalize_all()

        if final_words:
            tokens = [
                Token(
                    text=w["text"], start_ms=w["start_ms"], end_ms=w["end_ms"],
                    confidence=w["confidence"], is_final=True, speaker="1",
                    language=w["language"],
                )
                for w in final_words
            ]
            return TranscriptionResult(
                tokens=tokens,
                audio_offset_ms=state["audio_offset"],
                final_audio_proc_ms=audio_cursor_ms,
                total_audio_proc_ms=audio_cursor_ms,
            )

        return None


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

            await provider.process_chunk(
                pcm_int16        = data,
                session_id       = sid,
                audio_cursor_ms  = session.audio_cursor_ms,
                task             = task,
                beam_size        = beam_size,
                websocket        = websocket,
            )

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
