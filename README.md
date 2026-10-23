# ZentroMart Voice Support Agent

VoiceOps 2026 capstone (LLM & Generative AI course, C&A Club IIT Guwahati x Gnani.ai) by **Aarna Rathod**.

A support agent for ZentroMart, a fictional Indian online store. It answers in **English, Hindi and Hinglish** from the company documents and cites the section IDs it used, looks up orders only after identity verification, hands over to a human when policy requires it, says "I don't know" when the documents don't cover a question, and works by **voice** through Gnani STT and Gnani TTS.

- **Report:** [report.pdf](report.pdf) (3 pages)
- **Demo video:** [demo.mp4](demo.mp4) (47 s: answer from the documents, a follow-up, and "I don't know" with handover)

## Run command (evaluation)

```powershell
python run.py --input <folder of customer audio files> --output <output folder>
```

Each audio file (WAV, MP3, FLAC, OGG or M4A, up to 60 s) is treated as one customer call:

1. **Gnani STT** transcribes it (REST for files, falling back to the WebSocket stream).
2. The **agent** answers with retrieval, memory, tools and safety checks.
3. **Gnani TTS** speaks the reply.

For every file, the output folder gets:

| File | Contents |
|---|---|
| `<name>.json` | transcript, decision, reply text, tool calls, **cited section IDs**, and the **full text of every retrieved document chunk** |
| `<name>_reply.wav` | the spoken reply (Gnani TTS) |
| `results.csv` | one row per audio file |

Unclear or silent audio gets a spoken request to repeat. One failing file never stops the batch. See `sample_outputs/` for a real run on the recordings in `sample_audio/`.

Options: `--model` (default `openai/gpt-oss-120b`), `--lang` (Gnani STT language, default `en-IN`, which handles English and Hinglish; use `hi-IN` for Hindi), `--voice` (default `Kaveri`).

## Setup

Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. Put the course dataset at `dataset/Capstone_Project_Dataset_FTTT_2026/` (not included; check it with `python dataset\Capstone_Project_Dataset_FTTT_2026\validate_dataset.py`).
2. Copy `.env.example` to `.env` and add your own keys: `GROQ_API_KEY` and `GNANI_API_KEY` (required), `GEMINI_API_KEY` (only for Gemini models). Never commit `.env`.

## Other ways to run

| What | Command |
|---|---|
| Live voice conversation (mic streamed to Gnani while speaking) | `python run.py` |
| Voice messages as one conversation | `python run.py --wav msg1.wav msg2.wav` |
| Typed test | `python run.py --text` |
| Step 1 baseline (no documents) | `python step1_baseline.py openai/gpt-oss-120b` |
| Step 2 RAG evaluation | `python step2_rag.py openai/gpt-oss-120b` |
| Step 3 conversation evaluation | `python step3_conversations.py gemini-flash-lite-latest` |
| Step 4 ASR-robustness evaluation | `python step4_asr_eval.py openai/gpt-oss-120b` |
| Rebuild the demo video | `python make_demo.py` |

## Results (practice data, one shared scorer)

| Metric (52 questions, gpt-oss-120b) | Step 1: prompt only | Step 2: RAG | Step 2 on ASR-style text |
|---|---|---|---|
| Decision accuracy | 76.9% | **86.5%** | 73.1% |
| Must-include facts | 30.1% | **86.5%** | 74.4% |
| Evidence recall (cited IDs) | 0% | **76.2%** | 64.9% |
| Retrieval recall | – | **86.9%** | 80.9% |
| Forbidden claims | 0 | 0 | 0 |

**Step 3** (23 conversations, 96 turns): 92.7% decisions, 88.0% facts, 98.1% expected tool calls, 0 forbidden claims.

**Step 4 voice**: transcript 2.0 s after the caller stops (microphone streamed to Gnani while speaking; it was ~16 s when recording first), TTS 2.3–2.6 s.

## How it works

- **Retrieval:** one chunk per `[SECTION-ID]` with `status` and `effective_from`, plus product rows. Multilingual-e5 embeddings and BM25 are merged by rank fusion with a precedence prior (policy > agent guide > help > FAQ > marketing). Questions are rewritten into English search queries, and SKU/pincode rows are looked up exactly.
- **Dates:** business days (Mon–Sat, minus the listed holidays) are computed in Python and given to the model as a `[CALENDAR]` block.
- **Conversation:** memory of recent turns, `lookup_order` and `create_ticket` tools.
- **Safety enforced in code:**
  - The order record reaches the model without the pincode or customer ID, and verification digits are redacted from replies and tickets.
  - At most one verification retry, and no ticket in the same turn verification succeeds.
  - The AI introduction appears only in the first reply.
  - A **grounding check** replaces replies whose times, rupee amounts or policy claims are not in the sources with "I don't have that information" plus a human handover.

| File | Purpose |
|---|---|
| `run.py` | Run command (batch), live voice, WAV and text modes |
| `agent.py` | The agent: memory, retrieval, tools, safety checks; Groq or Gemini models |
| `retriever.py` | Hybrid multilingual retrieval |
| `calendar_helper.py` | Business-day calendar |
| `voice.py` | Gnani STT (WebSocket and REST) and TTS (WebSocket) |
| `score.py` | Shared scorer |
| `step1_baseline.py`, `step2_rag.py`, `step3_conversations.py`, `step4_asr_eval.py` | Evaluations |
| `make_demo.py`, `record_samples.py`, `inspect_results.py` | Demo builder, sample recorder, result viewer |

## Known limitations

- **Spoken numbers** (SKUs, order IDs) cause most voice errors.
- **The grounding check is a second model call, so it is not perfect.** In `sample_outputs/`, run on the Gemini Flash-Lite fallback, the midnight-delivery question (`q4`) still got an invented "daytime hours" answer.
- **Free-tier rate limits** affect latency and which model each evaluation used.
- **Voice recordings** other than the scripted samples are not committed, for privacy.