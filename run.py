"""Step 4: ZentroMart voice support agent.
Customer voice -> Gnani STT (WebSocket, streamed while speaking) -> agent (memory + RAG + tools)
-> Gnani TTS (WebSocket) -> spoken reply.

Usage:
  python run.py                                 # live mic, press Enter before each turn
  python run.py --text                          # type messages instead (quick testing)
  python run.py --wav msg1.wav msg2.wav         # voice messages from files, in order
Options: --model openai/gpt-oss-20b  --lang hi-IN  --voice Kaveri  --seconds 7
"""
import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf

from agent import SupportAgent
from retriever import Retriever
from voice import SR, listen, load_wav, speak, transcribe

REC = Path("recordings")
LOG = Path("results") / "step4_voice_log.csv"
FIELDS = ["time", "model", "mode", "user_audio", "transcript", "stt_s", "decision", "tools",
          "reply", "agent_s", "tts_s", "reply_audio", "total_s"]
HINDI_NOTE = "\n(The customer is speaking Hindi: reply in Hindi, written in Devanagari script.)"


def log_turn(row):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def describe(call):
    """lookup_order(order_id=..., verification_value=****) -> verified=True"""
    args = ", ".join(f"{k}={'****' if k == 'verification_value' else v}"
                     for k, v in call["args"].items() if k != "summary")
    res = call["result"]
    status = (f"verified={bool(res.get('verified'))}" if call["name"] == "lookup_order"
              else f"created={bool(res.get('created'))}")
    return f"{call['name']}({args}) -> {status}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--lang", default="hi-IN", help="Gnani STT language, e.g. hi-IN or en-IN")
    p.add_argument("--voice", default="Kaveri", help="Gnani TTS voice")
    p.add_argument("--seconds", type=int, default=7, help="listening time per turn")
    p.add_argument("--text", action="store_true", help="type messages instead of speaking")
    p.add_argument("--wav", nargs="*", help="voice-message WAV files to play in, in order")
    args = p.parse_args()

    REC.mkdir(exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "text" if args.text else ("wav" if args.wav else "mic")
    print("Loading knowledge base...")
    agent = SupportAgent(Retriever(), args.model)
    print(f"\nZentroMart voice agent ready ({mode} mode, model {args.model}).")
    print("Type 'exit' (or press Ctrl+C) to stop.\n")

    queue = list(args.wav or [])
    turn = 0
    while True:
        turn += 1
        user_audio, stt_s = "", 0.0

        if args.text:
            transcript = input("You: ").strip()
        elif args.wav is not None:
            if not queue:
                break
            user_audio = queue.pop(0)
            print(f"Voice message: {user_audio}")
            transcript, stt_s = transcribe(load_wav(user_audio), args.lang)
            print(f"You (STT, {stt_s}s): {transcript}")
        else:
            if input("Press Enter and speak (or type exit): ").strip().lower() == "exit":
                break
            audio, transcript, stt_s = listen(args.seconds, args.lang)
            user_audio = str(REC / f"{session}_t{turn}_customer.wav")
            sf.write(user_audio, audio, SR)
            print(f"You (STT, {stt_s}s after you stopped): {transcript}")

        if transcript.lower() in ("exit", "quit"):
            break
        if not transcript:
            print("   (nothing heard; please try again)\n")
            continue

        message = transcript + (HINDI_NOTE if re.search(r"[\u0900-\u097F]", transcript) else "")
        a0 = time.time()
        out = agent.turn(message)
        agent_s = round(time.time() - a0, 2)
        tools = " ; ".join(describe(c) for c in out["tool_calls"])
        print(f"Agent [{out['decision']}] ({agent_s}s): {out['reply']}")
        if tools:
            print(f"   tools: {tools}")

        reply_audio = str(REC / f"{session}_t{turn}_agent.wav")
        tts_s = speak(out["reply"], voice=args.voice, save_to=reply_audio)
        total_s = round(stt_s + agent_s + tts_s, 2)
        print(f"   (STT {stt_s}s + agent {agent_s}s + TTS {tts_s}s = {total_s}s)\n")

        log_turn({"time": datetime.now().isoformat(timespec="seconds"), "model": args.model,
                  "mode": mode, "user_audio": user_audio, "transcript": transcript,
                  "stt_s": stt_s, "decision": out["decision"], "tools": tools,
                  "reply": out["reply"], "agent_s": agent_s, "tts_s": tts_s,
                  "reply_audio": reply_audio, "total_s": total_s})

    print("Session ended. Audio in recordings/, timings in results/step4_voice_log.csv")


if __name__ == "__main__":
    main()