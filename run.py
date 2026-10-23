"""ZentroMart voice support agent (Step 4).

BATCH (used for evaluation): every audio file in a folder is one customer call.
  python run.py --input audio_folder --output outputs
  -> outputs/<name>.json        transcript, reply, decision, tools, cited IDs, retrieved chunks
     outputs/<name>_reply.wav   spoken reply (Gnani TTS)
     outputs/results.csv        one row per audio file

INTERACTIVE:
  python run.py                          # live mic, streamed to Gnani STT while you speak
  python run.py --wav a.wav b.wav        # voice messages as one conversation
  python run.py --text                   # type instead of speaking
Options: --model openai/gpt-oss-120b  --lang en-IN  --voice Kaveri  --seconds 7
"""
import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf

from agent import DailyLimitReached, SupportAgent
from retriever import Retriever
from voice import SR, listen, load_wav, speak, transcribe, transcribe_rest

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
REC = Path("recordings")
LOG = Path("results") / "step4_voice_log.csv"
LOG_FIELDS = ["time", "model", "mode", "user_audio", "transcript", "stt_s", "decision", "tools",
              "reply", "agent_s", "tts_s", "reply_audio", "total_s"]
HINDI_NOTE = "\n(The customer is speaking Hindi: reply in Hindi, written in Devanagari script.)"
NOT_HEARD = "Sorry, I didn't catch that. Could you please say it again?"


def describe(call):
    """lookup_order(order_id=..., verification_value=****) -> verified=True"""
    args = ", ".join(f"{k}={'****' if k == 'verification_value' else v}"
                     for k, v in call["args"].items() if k != "summary")
    res = call["result"]
    status = (f"verified={bool(res.get('verified'))}" if call["name"] == "lookup_order"
              else f"created={bool(res.get('created'))}")
    return f"{call['name']}({args}) -> {status}"


def ask_agent(agent, transcript):
    message = transcript + (HINDI_NOTE if re.search(r"[\u0900-\u097F]", transcript) else "")
    t0 = time.time()
    out = agent.turn(message)
    return out, round(time.time() - t0, 2)


def stt_file(path, lang):
    """Gnani STT for a voice-message file: REST first, WebSocket stream as fallback."""
    t0 = time.time()
    try:
        text = transcribe_rest(str(path), lang)
        return (text or "").strip(), round(time.time() - t0, 2), "rest"
    except Exception as rest_error:
        try:
            text, _ = transcribe(load_wav(str(path)), lang)
            return text, round(time.time() - t0, 2), "websocket"
        except Exception:
            raise rest_error


# ---------------------------------------------------------------- batch mode
def run_batch(args):
    src, out_dir = Path(args.input), Path(args.output)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        print(f"No audio files ({', '.join(sorted(AUDIO_EXT))}) found in {src}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading knowledge base... ({len(files)} audio files, model {args.model})")
    retriever = Retriever()
    chunk_by_id = {c["id"]: c for c in retriever.chunks}
    agent = SupportAgent(retriever, args.model)

    rows = []
    for i, path in enumerate(files, 1):
        agent.reset()                                   # each file is a separate call
        rec = {"file": path.name, "transcript": "", "stt_method": "", "stt_s": 0,
               "decision": "", "reply": "", "cited_ids": [], "tool_calls": [],
               "retrieved_chunks": [], "agent_s": 0, "tts_s": 0, "reply_audio": "", "error": ""}
        try:
            rec["transcript"], rec["stt_s"], rec["stt_method"] = stt_file(path, args.lang)
            if rec["transcript"]:
                out, rec["agent_s"] = ask_agent(agent, rec["transcript"])
                rec["decision"], rec["reply"] = out["decision"], out["reply"]
                rec["cited_ids"] = out["evidence"]
                rec["tool_calls"] = [describe(c) for c in out["tool_calls"]]
                rec["retrieved_chunks"] = [
                    {"id": cid, "title": chunk_by_id.get(cid, {}).get("title", ""),
                     "heading": chunk_by_id.get(cid, {}).get("heading", ""),
                     "status": chunk_by_id.get(cid, {}).get("status", ""),
                     "text": chunk_by_id.get(cid, {}).get("text", "")}
                    for cid in out["retrieved"]]
            else:
                rec["decision"], rec["reply"] = "clarify", NOT_HEARD   # unclear audio: ask again
        except DailyLimitReached:
            print("Daily LLM token limit reached; stopping. Finished files are saved.")
            break
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            if not rec["reply"]:
                rec["decision"], rec["reply"] = "escalate", ("Sorry, something went wrong on my side. "
                                                            "Let me connect you with a colleague.")
        try:
            reply_wav = out_dir / f"{path.stem}_reply.wav"
            rec["tts_s"] = speak(rec["reply"], voice=args.voice, save_to=str(reply_wav), play=False)
            rec["reply_audio"] = reply_wav.name
        except Exception as e:
            rec["error"] = (rec["error"] + " | " if rec["error"] else "") + f"TTS {type(e).__name__}: {e}"

        (out_dir / f"{path.stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        rows.append(rec)
        print(f"[{i}/{len(files)}] {path.name}: {rec['decision'] or 'error'} | "
              f"heard: {rec['transcript'][:60]!r} | cited: {', '.join(rec['cited_ids'])}"
              + (f" | ERROR {rec['error'][:80]}" if rec["error"] else ""))

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "transcript", "decision", "reply", "cited_ids", "retrieved_ids",
                    "tool_calls", "reply_audio", "stt_s", "agent_s", "tts_s", "error"])
        for r in rows:
            w.writerow([r["file"], r["transcript"], r["decision"], r["reply"], " ; ".join(r["cited_ids"]),
                        " ; ".join(c["id"] for c in r["retrieved_chunks"]), " ; ".join(r["tool_calls"]),
                        r["reply_audio"], r["stt_s"], r["agent_s"], r["tts_s"], r["error"]])
    print(f"\nDone: {len(rows)} files. Replies and sources in {out_dir}/ (see results.csv)")


# ---------------------------------------------------------- interactive mode
def log_turn(row):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def run_interactive(args):
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
            transcript, stt_s, _ = stt_file(user_audio, args.lang)
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
        reply_audio = str(REC / f"{session}_t{turn}_agent.wav")
        if not transcript:
            print(f"Agent [clarify]: {NOT_HEARD}")
            speak(NOT_HEARD, voice=args.voice, save_to=reply_audio)
            print()
            continue

        out, agent_s = ask_agent(agent, transcript)
        tools = " ; ".join(describe(c) for c in out["tool_calls"])
        print(f"Agent [{out['decision']}] ({agent_s}s): {out['reply']}")
        if out["evidence"]:
            print(f"   sources: {', '.join(out['evidence'])}")
        if tools:
            print(f"   tools: {tools}")
        tts_s = speak(out["reply"], voice=args.voice, save_to=reply_audio)
        total_s = round(stt_s + agent_s + tts_s, 2)
        print(f"   (STT {stt_s}s + agent {agent_s}s + TTS {tts_s}s = {total_s}s)\n")
        log_turn({"time": datetime.now().isoformat(timespec="seconds"), "model": args.model,
                  "mode": mode, "user_audio": user_audio, "transcript": transcript,
                  "stt_s": stt_s, "decision": out["decision"], "tools": tools,
                  "reply": out["reply"], "agent_s": agent_s, "tts_s": tts_s,
                  "reply_audio": reply_audio, "total_s": total_s})
    print("Session ended. Audio in recordings/, timings in results/step4_voice_log.csv")


def main():
    p = argparse.ArgumentParser(description="ZentroMart voice support agent")
    p.add_argument("--input", help="folder of customer audio files (batch mode)")
    p.add_argument("--output", default="outputs", help="where batch replies are saved")
    p.add_argument("--model", default="openai/gpt-oss-120b", help="LLM for the agent")
    p.add_argument("--lang", default="en-IN", help="Gnani STT language: en-IN (English/Hinglish), hi-IN, ...")
    p.add_argument("--voice", default="Kaveri", help="Gnani TTS voice")
    p.add_argument("--seconds", type=int, default=7, help="listening time per live turn")
    p.add_argument("--text", action="store_true", help="type messages instead of speaking")
    p.add_argument("--wav", nargs="*", help="voice-message files played in as one conversation")
    args = p.parse_args()
    if args.input:
        run_batch(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()