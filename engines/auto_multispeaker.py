"""Auto-multispeaker meta-engine — routes per-session to the right concrete engine.

Decision: multitalker-parakeet for sessions with ≤4 expected speakers (its
hardware-enforced ceiling), nemotron-native + sortformer-streaming v2.1 for
>4. This is the production architecture from `docs/ARCHITECTURE.md`.

Both engines speak the same Speechmatics-shim WebSocket protocol from the
caller's perspective; the routing happens entirely inside the server.

Cold-start strategy: lazy-load each underlying engine on first use. Loading
both upfront would double VRAM (~5 GB multitalker + ~3 GB nemotron-native)
and most sessions only need one. The trade-off is a one-time ~3–8 s cold
start per engine, paid on the first session that needs it.

On the fallback (>4 spks) path, nemotron-native emits words with
`speaker=None` and relies on a downstream diarizer to label them. This
meta-engine runs sortformer-streaming v2.1 inline so the labels arrive
attached to the StreamFinal events. The session then sees per-word speaker
labels regardless of which underlying engine fired — the user-selected
diarizer should stay `passthrough` because labeling is handled internally.
"""
from __future__ import annotations

from typing import Any, AsyncIterable

from server.engines._base import StreamEvent, StreamFinal, StreamingEngine

# 4 = the architectural ceiling of multitalker-parakeet 0.6B v1.
# instance_manager.max_num_of_spks is hardcoded to 4 in the bundled
# transcription config + the speaker-kernel encoder is sized for 4 channels.
# Past this we route to the pipeline fallback.
_MULTITALKER_SPK_CEILING = 4


def _resolve_max_speakers(transcription_config: dict[str, Any]) -> int:
    """Extract max_speakers from the Speechmatics transcription_config.

    The Speechmatics-canonical path is
    `transcription_config.speaker_diarization_config.max_speakers`. Some
    clients put it bare at `transcription_config.max_speakers` — read
    both, prefer the canonical nested one.

    Speechmatics' real default is 50, but for our routing the safer
    assumption when unset is the multitalker ceiling — better speaker
    accuracy at the cost of breaking on >4-spk sessions where DepoDash
    forgot to set the field. Calibrate based on field experience.
    """
    sm_diar = transcription_config.get("speaker_diarization_config") or {}
    legacy_diar = transcription_config.get("diarization_config") or {}
    raw = (
        sm_diar.get("max_speakers")
        or legacy_diar.get("max_speakers")
        or transcription_config.get("max_speakers")
    )
    try:
        return int(raw) if raw is not None else _MULTITALKER_SPK_CEILING
    except (TypeError, ValueError):
        return _MULTITALKER_SPK_CEILING


class AutoMultispeakerEngine:
    """Meta-engine that delegates `stream()` to multitalker or nemotron-native
    based on `transcription_config.max_speakers`."""

    name = "auto-multispeaker"

    def __init__(self) -> None:
        self._multitalker: StreamingEngine | None = None
        self._fallback: StreamingEngine | None = None
        self._fallback_diarizer = None

    def _load_multitalker(self) -> StreamingEngine:
        if self._multitalker is None:
            from engines.multitalker_parakeet import MultitalkerParakeet
            self._multitalker = MultitalkerParakeet()
        return self._multitalker

    def _load_fallback(self) -> StreamingEngine:
        if self._fallback is None:
            from engines.nemotron_native_streaming import NemotronNativeStreaming
            self._fallback = NemotronNativeStreaming()
        return self._fallback

    def _load_fallback_diarizer(self):
        if self._fallback_diarizer is None:
            from diarizers.sortformer_streaming import SortformerStreaming
            self._fallback_diarizer = SortformerStreaming()
        return self._fallback_diarizer

    def warm(self) -> None:
        """Pre-warm the primary engine (multitalker) at server boot. The
        fallback engine stays cold until first >4-spk session — most
        sessions never trigger it, so paying its load cost upfront would
        waste ~3 GB VRAM. If you expect frequent >4-spk traffic, override
        this to also call `self._load_fallback().warm()`."""
        engine = self._load_multitalker()
        if hasattr(engine, "warm"):
            engine.warm()

    async def stream(
        self,
        pcm_frames: AsyncIterable[bytes],
        transcription_config: dict,
    ) -> AsyncIterable[StreamEvent]:
        max_spks = _resolve_max_speakers(transcription_config)
        if max_spks <= _MULTITALKER_SPK_CEILING:
            engine = self._load_multitalker()
            async for event in engine.stream(pcm_frames, transcription_config):
                yield event
            return

        # Fallback path: nemotron-native emits speaker=None on every word;
        # tee the PCM into a buffer so sortformer-streaming v2.1 can label
        # words inline before we yield them to the session. Sortformer
        # maintains AOSC state across .label() calls in the same session so
        # speaker ids stay locked once assigned.
        engine = self._load_fallback()
        diarizer = self._load_fallback_diarizer()
        audio_so_far = bytearray()

        async def _tee_frames():
            async for frame in pcm_frames:
                if frame:
                    audio_so_far.extend(frame)
                yield frame

        async for event in engine.stream(_tee_frames(), transcription_config):
            if isinstance(event, StreamFinal) and event.words:
                # Sortformer-streaming only mutates speaker labels in place;
                # word content + order are preserved, so event.transcript
                # stays accurate without a rebuild.
                event.words = await diarizer.label(
                    event.words, memoryview(audio_so_far).tobytes()
                )
            yield event
