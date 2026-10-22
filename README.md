\# ZentroMart Voice Support Agent



VoiceOps 2026 capstone ("From Tokens to Transformers", GenAI x LLM, C\&A Club IIT Guwahati x Gnani.ai) by \*\*Aarna Rathod\*\*.



A support agent for ZentroMart, a fictional Indian online store. It answers customers in \*\*English, Hindi and Hinglish\*\*, grounds every answer in the company documents, looks up orders only after identity verification, hands over to humans when policy requires it, and accepts \*\*voice messages\*\* (Gnani STT over WebSocket) and replies with \*\*speech\*\* (Gnani TTS over WebSocket).



All data is the synthetic course dataset. Today's date for the dataset is fixed at \*\*20 October 2026\*\*.



\## Pipeline



```

customer voice ──► Gnani STT (WebSocket, 16 kHz PCM, real-time frames)

&#x20;                       │ transcript

&#x20;                       ▼

&#x20;             query rewrite (English search query, resolves "it / same item")

&#x20;                       │

&#x20;                       ▼

&#x20; hybrid retrieval: multilingual-e5 embeddings + BM25, rank fusion,

&#x20; precedence prior, exact SKU/pincode lookups, always-on rule \& agent-guide sections

&#x20;                       │

&#x20;                       ▼

&#x20; LLM agent ── tools: lookup\_order (identity-checked), create\_ticket

&#x20;    + conversation memory + business-day \[CALENDAR] computed in Python

&#x20;                       │ reply + decision + cited section IDs

&#x20;                       ▼

&#x20;             Gnani TTS (WebSocket) ──► spoken reply

```



\## Setup



Python 3.12 on Windows (PowerShell) was used.



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

```



1\. Put the course dataset at `dataset/Capstone\_Project\_Dataset\_FTTT\_2026/` (not committed; every participant has it). Check it with `python dataset\\Capstone\_Project\_Dataset\_FTTT\_2026\\validate\_dataset.py`.

2\. Create a `.env` file (never committed) with your own keys:

```

&#x20;  GROQ\_API\_KEY=...

&#x20;  GEMINI\_API\_KEY=...

&#x20;  GNANI\_API\_KEY=...

```

&#x20;  Groq and Gemini are used on their free tiers; no paid plan is needed.



\## How to run



| Step | Command | Output |

|---|---|---|

| 1 Baseline (no documents) | `python step1\_baseline.py openai/gpt-oss-120b` | `results/step1\_baseline\_\*.csv` |

| 2 RAG | `python step2\_rag.py openai/gpt-oss-120b` | `results/step2\_rag\_\*.csv` |

| 3 Conversations (memory + tools) | `python step3\_conversations.py gemini-flash-lite-latest` | `results/step3\_conversations\_\*.csv` |

| 4 Voice agent (live mic) | `python run.py --model openai/gpt-oss-20b` | `results/step4\_voice\_log.csv` |

| 4 Voice messages from files | `python run.py --wav msg1.wav msg2.wav` | same |

| 4 Typed test of the voice agent | `python run.py --text` | same |

| 4 ASR robustness | `python step4\_asr\_eval.py openai/gpt-oss-120b` | `results/step4\_asr\_\*.csv` |



Add a number at the end of steps 1–3 to run only the first N items, e.g. `python step2\_rag.py openai/gpt-oss-20b 8`. Step 3 is resumable: if a daily rate limit stops it, rerun the same command and it continues.



\## Files



| File | Purpose |

|---|---|

| `score.py` | One scorer for every step: decision, must-include facts (` / ` alternatives), forbidden claims, evidence recall; Hindi digits normalised |

| `step1\_baseline.py` | LLM only, no documents |

| `retriever.py` | Section-level chunking with `status` / `effective\_from`, products as chunks, multilingual-e5-small + BM25 with reciprocal rank fusion, precedence prior, SKU/pincode lookups |

| `calendar\_helper.py` | Business days per KB-00.3 (Mon–Sat, warehouse holidays skipped), dates found in English/Hindi text, `\[CALENDAR]` block |

| `step2\_rag.py` | Single-question RAG agent and evaluation |

| `agent.py` | Multi-turn agent: memory, retrieval, `lookup\_order` / `create\_ticket`, safety rules enforced in code, Groq or Gemini backend |

| `step3\_conversations.py` | Plays the 23 practice conversations and scores every turn including tool calls |

| `voice.py` | Gnani STT (WebSocket, 1,024-byte frames at 32 ms) and Gnani TTS (WebSocket) |

| `run.py` | Voice agent: mic / WAV / text input, spoken replies, per-turn timing log |

| `step4\_asr\_eval.py` | Step 2 pipeline on the dataset's ASR-style text |

| `inspect\_results.py` | Prints details for chosen or failed questions |



\## Results



\### Single questions (52 practice questions)



| Metric | Step 1 baseline (no docs) | Step 2 RAG | Step 4 ASR text |

|---|---|---|---|

| Model | gpt-oss-120b | gpt-oss-120b | gpt-oss-120b |

| Decision accuracy | 76.9% | \*\*86.5%\*\* | \_TBD\_ |

| Must-include facts (avg) | 30.1% | \*\*86.5%\*\* | \_TBD\_ |

| All facts present | 25.0% | \*\*84.6%\*\* | \_TBD\_ |

| Forbidden-claim hits | 0 | 0 | \_TBD\_ |

| Evidence recall | 0% | \*\*76.2%\*\* | \_TBD\_ |

| Retrieval recall | – | \*\*86.9%\*\* | \_TBD\_ |



By reply language (Step 2): English 85% facts, Hindi 100%, Hinglish 82%.



\### Conversations (23 conversations, 96 customer turns, Step 3)



| Metric | gemini-flash-lite-latest |

|---|---|

| Decision accuracy | \*\*92.7%\*\* |

| Must-include facts | \*\*88.0%\*\* |

| Forbidden claims (approvals, leaked secrets) | \*\*0\*\* |

| Expected tool calls made | \*\*98.1%\*\* (54 turns) |

| Evidence recall | 70.5% |

| Turns fully correct | 82.3% |

| Unexpected tickets | 3 (fixed afterwards in code, see below) |



\### Voice (Step 4)



\_TBD: per-turn STT / agent / TTS latency from `results/step4\_voice\_log.csv`.\_



\## Design decisions



\- \*\*Retrieval that survives messy language.\*\* Customers say "nobody came to collect it"; documents say "reverse pickup". Multilingual embeddings handle paraphrase and Hindi, BM25 handles exact SKUs and IDs, and an English query rewrite lets Hindi questions reach English policies (the Hindi refund question found no refund policy until the rewrite was added).

\- \*\*Policy precedence and versions.\*\* Each chunk carries `doc\_type`, `status` and `effective\_from`; the prompt applies KB-00.1 (policy > agent guide > help > FAQ > marketing), and superseded sections apply only to the cases the current policy names (e.g. electronics delivered before 1 Aug 2026 keep the 10-day window).

\- \*\*Dates computed, not guessed.\*\* Both models miscounted "5–7 business days from 17 October" (they counted the start day). Business days are computed in Python and given as an explicit `\[CALENDAR]` block, with the counting rule stated.

\- \*\*Safety rules enforced in code, not only in the prompt.\*\* Verification secrets are stripped from the order record before the model sees it and redacted from replies and tickets; at most one retry; the decision label follows what the tools actually did; and a ticket cannot be created in the same turn as verification (the model escalated before the customer had asked anything).

\- \*\*Free tiers only.\*\* Groq's free tier allows 200K tokens/day per model, so Step 3 (\~600K tokens) ran on Gemini Flash-Lite; Gemini Flash took \~250 s per request on the free tier and was not usable. Voice uses Groq for low latency.



\## Limitations



\- Step 3 latency (\~75 s/turn) is free-tier queueing, not model time.

\- Some decisions in failed-verification flows and a few early escalations remain (see `results/step3\_\*.csv`).

\- Gnani `hi-IN` STT writes Hinglish speech in Devanagari, so voice replies come back in Hindi script.

\- Voice recordings are not committed, for privacy.

