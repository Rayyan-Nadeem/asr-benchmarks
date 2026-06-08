"""Per-connection session state machine — wires the WebSocket to the engine + diarizer."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from server.diarizer_registry import load_diarizer
from server.engine_registry import load_engine
from server.engines._base import StreamFinal, StreamPartial, StreamWord
from server.punctuator_registry import load_punctuator
from server.protocol import (
    AddPartialTranscript,
    AddTranscript,
    AudioAdded,
    AudioFormat,
    EndOfStream,
    EndOfTranscript,
    Error,
    Info,
    RecognitionStarted,
    StartRecognition,
    Word,
    decode,
    encode,
    validate_audio_format,
    validate_transcription_config,
)

# Terminating punctuation marks that flip the result-level `is_eos` flag.
_EOS_MARKS = {".", "?", "!"}


log = logging.getLogger(__name__)


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.id = str(uuid.uuid4())
        self.seq_no = 0
        self.engine = load_engine()
        self.diarizer = load_diarizer()
        self.punctuator = load_punctuator()
        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.audio_so_far = bytearray()  # for the diarizer's overlay pass
        self.transcription_config: dict[str, Any] = {}
        # Arrival-order remap so diarizer-native labels (pyannote "SPEAKER_00",
        # sortformer "speaker_0", etc.) surface to the client as SM-style
        # "S1"/"S2"/"S3" — what DepoDash's middleware has been parsing for years.
        self._speaker_remap: dict[str, str] = {}

    async def run(self) -> None:
        try:
            start = await self._await_start_recognition()
            if start is None:
                return
            await self._send(RecognitionStarted(id=self.id))
            await self._send(Info(
                type="recognition_quality",
                reason=f"Running recognition using {self.engine.name}",
                quality="broadcast",
            ))
            await asyncio.gather(
                self._receive_loop(),
                self._engine_loop(),
            )
        except WebSocketDisconnect:
            log.info("session %s: client disconnected", self.id)
        except Exception as e:  # noqa: BLE001
            log.exception("session %s: unhandled error", self.id)
            await self._send_safe(Error(type="job_error", reason=str(e)))

    async def _await_start_recognition(self) -> StartRecognition | None:
        raw = await self.ws.receive_text()
        msg = decode(raw)
        if msg.get("message") != "StartRecognition":
            await self._send(Error(
                type="invalid_message",
                reason=f"expected StartRecognition first; got {msg.get('message')!r}",
            ))
            return None
        start = StartRecognition.from_json(msg)
        if err := validate_audio_format(start.audio_format):
            await self._send(err)
            return None
        if err := validate_transcription_config(start.transcription_config):
            await self._send(err)
            return None
        self.transcription_config = start.transcription_config
        return start

    async def _receive_loop(self) -> None:
        """Pump binary frames into the engine queue; handle EndOfStream."""
        while True:
            try:
                msg = await self.ws.receive()
            except WebSocketDisconnect:
                await self.audio_queue.put(None)
                return

            if msg.get("type") == "websocket.disconnect":
                await self.audio_queue.put(None)
                return

            if "bytes" in msg and msg["bytes"] is not None:
                self.seq_no += 1
                self.audio_so_far.extend(msg["bytes"])
                await self.audio_queue.put(msg["bytes"])
                await self._send(AudioAdded(seq_no=self.seq_no))
            elif "text" in msg and msg["text"] is not None:
                control = decode(msg["text"])
                if control.get("message") == "EndOfStream":
                    await self.audio_queue.put(None)
                    return

    async def _audio_iter(self):
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                return
            yield chunk

    async def _engine_loop(self) -> None:
        """Forward engine events to the client as protocol frames."""
        async for event in self.engine.stream(self._audio_iter(), self.transcription_config):
            if isinstance(event, StreamPartial):
                await self._send(AddPartialTranscript(
                    transcript=event.transcript,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    results=[],
                ))
            elif isinstance(event, StreamFinal):
                labeled_words = await self.diarizer.label(event.words, bytes(self.audio_so_far))
                # Punctuation post-process — passthrough by default (raw
                # model output), distilbert when the user toggles it on.
                punctuated_words = await self.punctuator.punctuate(labeled_words)
                # Rebuild the human-readable transcript from the
                # punctuated words so the AddTranscript.transcript field
                # reflects the punctuator's output, not the engine's raw.
                if punctuated_words is not labeled_words:
                    parts: list[str] = []
                    for w in punctuated_words:
                        if w.is_punctuation:
                            parts.append(w.content)
                        else:
                            parts.append((" " if parts else "") + w.content)
                    transcript = "".join(parts).strip()
                else:
                    transcript = event.transcript
                await self._send(AddTranscript(
                    transcript=transcript,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    results=[self._word_to_protocol(w) for w in punctuated_words],
                ))
        await self._send(EndOfTranscript())

    async def _send(self, msg: Any) -> None:
        await self.ws.send_text(encode(msg))

    async def _send_safe(self, msg: Any) -> None:
        try:
            await self._send(msg)
        except Exception:  # noqa: BLE001
            pass

    def _remap_speaker(self, raw: str | None) -> str | None:
        if not raw:
            return None
        existing = self._speaker_remap.get(raw)
        if existing is not None:
            return existing
        new = f"S{len(self._speaker_remap) + 1}"
        self._speaker_remap[raw] = new
        return new

    def _word_to_protocol(self, w: StreamWord) -> Word:
        alt: dict[str, Any] = {
            "confidence": w.confidence if w.confidence is not None else 1.0,
            "content": w.content,
            "language": "en",
        }
        speaker = self._remap_speaker(w.speaker)
        if speaker is not None:
            alt["speaker"] = speaker

        if w.is_punctuation:
            return Word(
                type="punctuation",
                start_time=w.start_time,
                end_time=w.end_time,
                alternatives=[alt],
                attaches_to="previous",
                is_eos=w.content in _EOS_MARKS,
            )
        return Word(
            type="word",
            start_time=w.start_time,
            end_time=w.end_time,
            alternatives=[alt],
        )
