"""Gnani speech layer for the ZentroMart agent.
STT: realtime WebSocket (raw PCM 16-bit, 16 kHz mono, 1,024-byte frames at 32 ms pace).
     listen() streams the microphone to Gnani WHILE the customer speaks.
TTS: realtime WebSocket, collected to WAV and played.
Run directly for a self-test: python voice.py
"""
import asyncio
import io
import logging
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient, GnaniSTTStreamClient
from gnani.tts import GnaniTTSRealtimeClient, AudioConfig

load_dotenv()
logging.getLogger("gnani").setLevel(logging.WARNING)   # hide per-chunk DEBUG lines
API_KEY = os.getenv("GNANI_API_KEY")
SR = 16000
FRAME_BYTES = 1024            # 512 samples x 2 bytes
FRAME_SAMPLES = 512
FRAME_SECONDS = 0.032


def record(seconds=7):
    """Record from the default microphone as 16 kHz mono int16 (no streaming)."""
    print(f"   Listening for {seconds} s... speak now.")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)


def load_wav(path):
    """Read any WAV and convert to 16 kHz mono int16."""
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    mono = data.mean(axis=1).astype(np.int16)
    if sr != SR:
        idx = np.linspace(0, len(mono) - 1, int(len(mono) * SR / sr))
        mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.int16)
    return mono


class _Listener:
    """Collects transcript events from an open Gnani stream in the background."""

    def __init__(self, stream):
        self.stream, self.texts, self.processing, self.errors = stream, [], 0, []

    async def run(self):
        try:
            async for event in self.stream:
                kind = type(event).__name__
                if kind == "StreamProcessingEvent":
                    self.processing += 1
                elif kind == "StreamTranscriptEvent" and getattr(event, "text", ""):
                    self.texts.append(event.text)
                elif kind == "StreamErrorEvent":
                    self.errors.append(getattr(event, "message", str(event)))
        except Exception as e:      # stream closed
            if not self.texts:
                self.errors.append(repr(e))

    async def finish(self, min_wait=0.8, max_wait=4.0):
        """Wait until every detected speech segment has its transcript."""
        start = time.time()
        while time.time() - start < max_wait:
            if time.time() - start >= min_wait and len(self.texts) >= self.processing:
                break
            await asyncio.sleep(0.05)


async def _close(stream, task):
    try:
        await stream.close()
    except Exception:
        pass
    try:
        await asyncio.wait_for(task, timeout=2)
    except Exception:
        task.cancel()


async def _stt(pcm, language):
    stream = GnaniSTTStreamClient(api_key=API_KEY, language_code=language, sample_rate=SR)
    await stream.connect()
    listener = _Listener(stream)
    task = asyncio.create_task(listener.run())
    data = pcm + b"\x00" * FRAME_BYTES * 25        # ~0.8 s silence closes the last segment
    for i in range(0, len(data), FRAME_BYTES):
        await stream.send_audio(data[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00"))
        await asyncio.sleep(FRAME_SECONDS)         # real-time pacing for Gnani's VAD
    await listener.finish()
    await _close(stream, task)
    if listener.errors and not listener.texts:
        print("   STT stream messages:", listener.errors[:3])
    return " ".join(listener.texts).strip()


def transcribe(audio, language="hi-IN"):
    """int16 numpy audio (e.g. a voice-message file) -> (text, seconds) via the WebSocket stream."""
    t0 = time.time()
    text = asyncio.run(_stt(audio.astype(np.int16).tobytes(), language))
    return text, round(time.time() - t0, 2)


async def _listen(seconds, language):
    loop = asyncio.get_running_loop()
    frames = asyncio.Queue()
    captured = []
    stream = GnaniSTTStreamClient(api_key=API_KEY, language_code=language, sample_rate=SR)
    await stream.connect()
    listener = _Listener(stream)
    task = asyncio.create_task(listener.run())

    def on_audio(indata, n, time_info, status):     # runs in the audio thread
        loop.call_soon_threadsafe(frames.put_nowait, bytes(indata))

    print(f"   Listening for {seconds} s... speak now.")
    with sd.RawInputStream(samplerate=SR, channels=1, dtype="int16",
                           blocksize=FRAME_SAMPLES, callback=on_audio):
        end = loop.time() + seconds
        while loop.time() < end:
            try:
                frame = await asyncio.wait_for(frames.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            captured.append(frame)
            await stream.send_audio(frame.ljust(FRAME_BYTES, b"\x00"))   # sent while speaking
    t_stop = time.time()
    for _ in range(25):                               # ~0.8 s silence closes the last segment
        await stream.send_audio(b"\x00" * FRAME_BYTES)
        await asyncio.sleep(FRAME_SECONDS)
    await listener.finish()
    latency = round(time.time() - t_stop, 2)          # transcript delay after the customer stops
    await _close(stream, task)
    if listener.errors and not listener.texts:
        print("   STT stream messages:", listener.errors[:3])
    audio = np.frombuffer(b"".join(captured), dtype=np.int16)
    return audio, " ".join(listener.texts).strip(), latency


def listen(seconds=7, language="hi-IN"):
    """Live mic streamed to Gnani STT while speaking -> (audio, text, seconds after speech)."""
    return asyncio.run(_listen(seconds, language))


def transcribe_rest(path, language="hi-IN"):
    """Diagnostic only: Gnani REST transcription of a WAV file."""
    return GnaniSTTClient(api_key=API_KEY).transcribe(path, language_code=language).get("transcript", "")


async def _tts(text, voice):
    async with GnaniTTSRealtimeClient(api_key=API_KEY) as client:
        return await client.synthesize_and_collect(
            text, voice=voice,
            audio_config=AudioConfig(sample_rate=SR, encoding="linear_pcm", container="wav"))


def speak(text, voice="Kaveri", save_to=None, play=True):
    """Text -> Gnani TTS audio; saves WAV if asked, plays it. Returns seconds to synthesize."""
    t0 = time.time()
    wav = asyncio.run(_tts(text, voice))
    took = round(time.time() - t0, 2)
    if save_to:
        with open(save_to, "wb") as f:
            f.write(wav)
    if play:
        data, sr = sf.read(io.BytesIO(wav), dtype="int16")
        sd.play(data, sr)
        sd.wait()
    return took


if __name__ == "__main__":
    print("Live STT test (streams while you speak):")
    audio, text, took = listen(6)
    print(f"   Transcript: {text!r}  ({took}s after you stopped)")
    print("TTS test:")
    took = speak("Namaste! Main ZentroMart ki AI assistant hoon. Aapki kya madad kar sakti hoon?")
    print(f"   Spoken (synthesis took {took}s)")