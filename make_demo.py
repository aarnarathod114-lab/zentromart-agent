"""Build demo.mp4 from a real run of the agent: recorded customer voice messages are played
into ONE conversation (Gnani STT -> agent -> Gnani TTS); every screen shows the actual output.
Usage: python make_demo.py [model]
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from agent import SupportAgent
from retriever import Retriever
from run import ask_agent, describe, stt_file
from voice import SR, load_wav, speak

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
TURNS = ["sample_audio/q1_return_window.wav",     # answer from the documents
         "sample_audio/q2_refund_upi.wav",        # follow-up
         "sample_audio/q4_unanswerable.wav"]      # "I don't know" + human handover
OUT = Path("demo_build")
W, H = 1280, 720
FONT_DIR = "C:/Windows/Fonts/"
INK, MUTED = (31, 41, 51), (91, 102, 112)


def font(size, bold=False, devanagari=False):
    names = (["Nirmala.ttc", "Nirmala.ttf"] if devanagari else []) + \
            (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(FONT_DIR + name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap(draw, text, f, width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=f) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def frame(path, label, body, extra="", color=(47, 93, 138)):
    img = Image.new("RGB", (W, H), (250, 248, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 70], fill=INK)
    d.text((40, 18), "ZentroMart Voice Support Agent  ·  recorded run of the agent",
           font=font(26, bold=True), fill="white")
    d.text((40, 100), label, font=font(30, bold=True), fill=color)
    f = font(34, devanagari=any("\u0900" <= c <= "\u097f" for c in body))
    y = 160
    for line in wrap(d, body, f, W - 80)[:9]:
        d.text((40, y), line, font=f, fill=INK)
        y += 48
    if extra:
        y = H - 120
        for line in wrap(d, extra, font(22), W - 80)[:3]:
            d.text((40, y), line, font=font(22), fill=MUTED)
            y += 30
    d.text((40, H - 36), "Gnani STT (speech to text)  →  agent (RAG + memory + tools)  →  Gnani TTS (text to speech)",
           font=font(18), fill=MUTED)
    img.save(path)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.int16)


def main():
    OUT.mkdir(exist_ok=True)
    agent = SupportAgent(Retriever(), MODEL)          # one conversation for all turns
    segments = []
    title = OUT / "f00.png"
    frame(title, "Demo: one customer call, three questions",
          "1. A question answered from ZentroMart's documents\n2. A follow-up question\n"
          "3. A question the documents don't cover: \"I don't know\" + handover to a human\n\n"
          "Aarna Rathod  ·  VoiceOps 2026")
    segments.append((title, silence(4)))

    for i, path in enumerate(TURNS, 1):
        text, stt_s, _ = stt_file(path, "en-IN")
        out, agent_s = ask_agent(agent, text)
        reply_wav = OUT / f"reply{i}.wav"
        speak(out["reply"], save_to=str(reply_wav), play=False)
        print(f"Turn {i}: heard {text!r}\n   -> [{out['decision']}] {out['reply']}")

        f1, f2 = OUT / f"f{i}a.png", OUT / f"f{i}b.png"
        frame(f1, f"Customer  (voice message {i})", text, f"Transcribed by Gnani STT in {stt_s}s")
        sources = ", ".join(out["evidence"]) or "none"
        tools = " ; ".join(describe(c) for c in out["tool_calls"])
        frame(f2, f"Agent reply  [decision: {out['decision']}]", out["reply"],
              f"Sources: {sources}" + (f"   ·   Tools: {tools}" if tools else "") + f"   ·   agent {agent_s}s",
              color=(190, 110, 40))
        segments += [(f1, np.concatenate([load_wav(path), silence(0.4)])),
                     (f2, np.concatenate([load_wav(str(reply_wav)), silence(0.8)]))]

    audio = np.concatenate([a for _, a in segments])
    sf.write(OUT / "audio.wav", audio, SR)
    with open(OUT / "frames.txt", "w") as f:
        for png, a in segments:
            f.write(f"file '{png.name}'\nduration {len(a) / SR:.3f}\n")
        f.write(f"file '{segments[-1][0].name}'\n")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(OUT / "frames.txt"),
                    "-i", str(OUT / "audio.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", "demo.mp4"], check=True)
    print(f"\nSaved demo.mp4 ({len(audio) / SR:.0f} seconds)")


if __name__ == "__main__":
    main()