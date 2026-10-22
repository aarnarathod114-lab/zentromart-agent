"""Gnani speech layer for the ZentroMart agent.
STT: realtime WebSocket (raw PCM 16-bit, 16 kHz mono, 1,024-byte frames at 32 ms pace),
     with transcripts received as live events while audio is still streaming.
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
FRAME_SECONDS = 0.032


def record(seconds=7):
    """Record from the default microphone as 16 kHz mono int16."""
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


async def _stt(pcm, language):
    texts, errors = [], []
    stream = GnaniSTTStreamClient(api_key=API_KEY, language_code=language, sample_rate=SR)
    await stream.connect()

    async def receive():                       # transcripts arrive as live events
        try:
            async for event in stream:
                kind = type(event).__name__
                if kind == "StreamTranscriptEvent" and getattr(event, "text", ""):
                    texts.append(event.text)
                elif kind == "StreamErrorEvent":
                    errors.append(getattr(event, "message", str(event)))
        except Exception as e:                 # stream closed or connection ended
            if not texts:
                errors.append(repr(e))

    listener = asyncio.create_task(receive())
    data = pcm + b"\x00" * FRAME_BYTES * 40      # ~1.3 s of silence closes the last segment
    for i in range(0, len(data), FRAME_BYTES):
        await stream.send_audio(data[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00"))
        await asyncio.sleep(FRAME_SECONDS)       # real-time pacing for Gnani's VAD
    await asyncio.sleep(2.5)                     # wait for the final transcript event
    try:
        await stream.close()
    except Exception:
        pass
    try:
        await asyncio.wait_for(listener, timeout=3)
    except Exception:
        listener.cancel()
    if errors and not texts:
        print("   STT stream messages:", errors[:3])
    return " ".join(texts).strip()


def transcribe(audio, language="hi-IN"):
    """int16 numpy audio -> (text, seconds taken) via the WebSocket stream."""
    t0 = time.time()
    text = asyncio.run(_stt(audio.astype(np.int16).tobytes(), language))
    return text, round(time.time() - t0, 2)


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
    print("1) STT over WebSocket on mic_test.wav")
    text, took = transcribe(load_wav("mic_test.wav"))
    print(f"   WebSocket transcript: {text!r}  ({took}s incl. real-time streaming)")
    print("2) STT over REST on the same file (diagnostic)")
    print(f"   REST transcript: {transcribe_rest('mic_test.wav')!r}")
    print("3) TTS test")
    took = speak("Namaste! Main ZentroMart ki AI assistant hoon. Aapki kya madad kar sakti hoon?",
                 save_to="tts_test.wav")
    print(f"   Spoke and saved tts_test.wav (synthesis took {took}s)")