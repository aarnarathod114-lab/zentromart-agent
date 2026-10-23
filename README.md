# ZentroMart Voice Support Agent

VoiceOps 2026 capstone ("From Tokens to Transformers", GenAI x LLM, C&A Club IIT Guwahati x Gnani.ai) by **Aarna Rathod**.

A support agent for ZentroMart, a fictional Indian online store. It answers customers in **English, Hindi and Hinglish**, grounds every answer in the company documents, looks up orders only after identity verification, hands over to humans when policy requires it, and accepts **voice** (Gnani STT over WebSocket, streamed while the customer speaks) and replies with **speech** (Gnani TTS over WebSocket).

The full write-up with all results is in **[report.pdf](report.pdf)**.

All data is the synthetic course dataset. Today's date for the dataset is fixed at **20 October 2026**.

## Pipeline

```
customer voice ──► Gnani STT (WebSocket, 16 kHz PCM, frames streamed while speaking)
                        │ transcript
                        ▼
              query rewrite (English search query, resolves "it / same item")
                        │
                        ▼
  hybrid retrieval: multilingual-e5 embeddings + BM25, rank fusion,
  precedence prior, exact SKU/pincode lookups, always-on rule & agent-guide sections
                        │
                        ▼
  LLM agent ── tools: lookup_order (identity-checked), create_ticket
     + conversation memory + business-day [CALENDAR] computed in Python
                        │ reply + decision + cited section IDs
                        ▼
              Gnani TTS (WebSocket) ──► spoken reply
```

## Setup

Python 3.12 on Windows (PowerShell) was used.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. Put the course dataset at `dataset/Capstone_Project_Dataset_FTTT_2026/` (not committed; every participant has it). Check it with `python dataset\Capstone_Project_Dataset_FTTT_2026\validate_dataset.py`.
2. Create a `.env` file (never committed) with your own keys:
```
   GROQ_API_KEY=...
   GEMINI_API_KEY=...
   GNANI_API_KEY=...
```
   Groq and Gemini are used on their free tiers; no paid plan is needed.

## How to run

| Step | Command | Output |
|---|---|---|
| 1 Baseline (no documents) | `python step1_baseline.py openai/gpt-oss-120b` | `results/step1_baseline_*.csv` |
| 2 RAG | `python step2_rag.py openai/gpt-oss-120b` | `results/step2_rag_*.csv` |
| 3 Conversations (memory + tools) | `python step3_conversations.py gemini-flash-lite-latest` | `results/step3_conversations_*.csv` |
| 4 Voice agent (live mic) | `python run.py --model openai/gpt-oss-20b` | `results/step4_voice_log.csv` |
| 4 Voice messages from files | `python run.py --wav msg1.wav msg2.wav` | same |
| 4 Typed test of the voice agent | `python run.py --text` | same |
| 4 ASR robustness | `python step4_asr_eval.py openai/gpt-oss-120b` | `results/step4_asr_*.csv` |
| Voice self-test | `python voice.py` | live STT + TTS check |

Add a number at the end of steps 1–3 to run only the first N items, e.g. `python step2_rag.py openai/gpt-oss-20b 8`. Step 3 is resumable: if a daily rate limit stops it, rerun the same command and it continues.

## Files

| File | Purpose |
|---|---|
| `score.py` | One scorer for every step: decision, must-include facts (` / ` alternatives), forbidden claims, evidence recall; Hindi digits normalised |
| `step1_baseline.py` | LLM only, no documents |
| `retriever.py` | Section-level chunking with `status` / `effective_from`, products as chunks, multilingual-e5-small + BM25 with reciprocal rank fusion, precedence prior, SKU/pincode lookups |
| `calendar_helper.py` | Business days per KB-00.3 (Mon–Sat, warehouse holidays skipped), dates found in English/Hindi text, `[CALENDAR]` block |
| `step2_rag.py` | Single-question RAG agent and evaluation |
| `agent.py` | Multi-turn agent: memory, retrieval, `lookup_order` / `create_ticket`, safety rules enforced in code, Groq or Gemini backend |
| `step3_conversations.py` | Plays the 23 practice conversations and scores every turn including tool calls |
| `voice.py` | Gnani STT (WebSocket, 1,024-byte frames at 32 ms, mic streamed while speaking) and Gnani TTS (WebSocket) |
| `run.py` | Voice agent: mic / WAV / text input, spoken replies, per-turn timing log |
| `step4_asr_eval.py` | Step 2 pipeline on the dataset's ASR-style text |
| `inspect_results.py` | Prints details for chosen or failed questions |
| `report.pdf` | Full project report |

## Results

### Single questions (52 practice questions)

| Metric | Step 1 baseline (no docs) | Step 2 RAG | Step 4 ASR-style text |
|---|---|---|---|
| Model | gpt-oss-120b | gpt-oss-120b | gpt-oss-120b |
| Decision accuracy | 76.9% | **86.5%** | 73.1% |
| Must-include facts (avg) | 30.1% | **86.5%** | 74.4% |
| All facts present | 25.0% | **84.6%** | – |
| Forbidden-claim hits | 0 | 0 | 0 |
| Evidence recall | 0% | **76.2%** | 64.9% |
| Retrieval recall | – | **86.9%** | 80.9% |

By reply language (Step 2): English 85% facts, Hindi 100%, Hinglish 82%.

### Conversations (23 conversations, 96 customer turns, Step 3)

| Metric | gemini-flash-lite-latest |
|---|---|
| Decision accuracy | **92.7%** |
| Must-include facts | **88.0%** |
| Forbidden claims (approvals, leaked secrets) | **0** |
| Expected tool calls made | **98.1%** (54 turns) |
| Evidence recall | 70.5% |
| Turns fully correct | 82.3% |
| Unexpected tickets | 3 (early escalations; afterwards blocked in code) |

### Voice (Step 4)

| Stage | First version | Final version |
|---|---|---|
| STT: transcript after the customer stops | 15.6–16.9 s | **2.0 s** (mic streamed to Gnani while speaking) |
| Agent (Groq gpt-oss-20b) | 2.7–9.6 s | ~3 s unthrottled; up to 40 s when the free-tier per-minute limit was hit |
| TTS synthesis (Gnani WebSocket) | 2.0–3.6 s | 2.3–2.6 s |

Two live sessions were run: a **success path** (greeting → order ID → verification → damaged TV → ticket for the human team, never saying "approved") and a **real ASR error** (the order number was mis-transcribed, verification failed, the agent allowed one retry, shared nothing and offered a human).

## Design decisions

- **Retrieval that survives messy language.** Customers say "nobody came to collect it"; documents say "reverse pickup". Multilingual embeddings handle paraphrase and Hindi, BM25 handles exact SKUs and IDs, and an English query rewrite lets Hindi questions reach English policies (the Hindi refund question found no refund policy until the rewrite was added).
- **Policy precedence and versions.** Each chunk carries `doc_type`, `status` and `effective_from`; the prompt applies KB-00.1 (policy > agent guide > help > FAQ > marketing), and superseded sections apply only to the cases the current policy names (e.g. electronics delivered before 1 Aug 2026 keep the 10-day window).
- **Dates computed, not guessed.** Both models miscounted "5–7 business days from 17 October" (they counted the start day). Business days are computed in Python and given as an explicit `[CALENDAR]` block, with the counting rule stated.
- **Safety rules enforced in code, not only in the prompt.** Verification secrets are stripped from the order record before the model sees it and redacted from replies and tickets; at most one retry; the decision label follows what the tools actually did; and a ticket cannot be created in the same turn as verification.
- **Real-time voice.** Microphone frames are streamed to Gnani STT while the customer speaks, so the transcript arrives about 2 s after they stop instead of ~16 s.
- **Free tiers only.** Groq's free tier allows 200K tokens/day per model, so Step 3 (~600K tokens) ran on Gemini Flash-Lite; Gemini Flash took ~250 s per request on the free tier and was not usable. Voice uses Groq for low latency.

## Limitations and next steps

- **Spoken numbers** cause most of the Step 4 drop ("ELE-0012" heard as "ele-zero zero one to"). Next step: an ASR normaliser for IDs, and confirming an order number with the customer before lookup.
- After a second failed verification the agent offers a human but does not yet create the ticket itself.
- Several single questions are answered with "clarify" where a direct answer was expected.
- Latency figures include free-tier throttling; Step 3 averaged ~75 s per turn on Gemini, almost all queueing.
- Step 3 was scored on a smaller model than Step 2 because of daily token limits.
- Gnani `hi-IN` STT writes Hinglish speech in Devanagari, so voice replies are in Hindi script.
- Voice recordings are not committed, for privacy.