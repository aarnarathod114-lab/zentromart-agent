"""Step 1 baseline: LLM only, no documents, no tools.
Usage: python step1_baseline.py [model] [limit]
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, BadRequestError

from score import decision_ok, include_score, violates, evidence_recall

load_dotenv()
MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
DATA = Path("dataset/Capstone_Project_Dataset_FTTT_2026")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

SYSTEM = """You are the customer support agent for ZentroMart, an Indian online store.
Today's date is 20 October 2026.
In this version you have NO access to company documents, order records or tools.

Choose exactly one decision:
- "answer": you can reply to the customer's question
- "clarify": you need more information from the customer first
- "escalate": a human colleague must handle it
- "refuse": the request is something you must not do

Reply in the customer's language: English -> English; Hinglish (Hindi in Roman letters) -> Hinglish;
Hindi in Devanagari script -> Hindi in Devanagari. Keep the reply to 2-4 sentences.

Return ONLY a JSON object:
{"decision": "answer|clarify|escalate|refuse", "reply": "...", "evidence": ["SECTION-ID", ...]}
"evidence" lists the document section IDs you relied on; use [] if none."""

client = Groq(max_retries=6)


def call(question, json_mode=True):
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0,
        max_completion_tokens=2048,
        reasoning_effort="low",
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)


def ask(question):
    try:
        r = call(question)
    except BadRequestError:
        r = call(question, json_mode=False)
    text = r.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        data = {"decision": "answer", "reply": text, "evidence": []}
    return data, r.usage.total_tokens


def main():
    q = pd.read_csv(DATA / "practice" / "questions.csv", dtype=str).fillna("")
    if LIMIT:
        q = q.head(LIMIT)

    rows, tokens = [], 0
    for _, row in q.iterrows():
        t0 = time.time()
        data, used = ask(row["question"])
        latency = time.time() - t0
        tokens += used

        decision = str(data.get("decision", "")).strip().lower()
        reply = str(data.get("reply", ""))
        evidence = data.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]

        rec = {
            "id": row["id"],
            "reply_language": row["reply_language"],
            "expected_decision": row["expected_decision"],
            "decision": decision,
            "decision_ok": decision_ok(decision, row["acceptable_decisions"]),
            "include": include_score(reply, row["must_include"]),
            "violation": violates(reply, row["must_not_include"]),
            "evidence_recall": evidence_recall(evidence, row["gold_evidence"]),
            "evidence": " ; ".join(map(str, evidence)),
            "latency_s": round(latency, 2),
            "reply": reply,
        }
        rows.append(rec)
        print(f"{rec['id']}  {decision:9} ok={rec['decision_ok']!s:5} "
              f"include={rec['include']:.2f}  {latency:.1f}s")

    df = pd.DataFrame(rows)
    name = MODEL.split("/")[-1]
    path = OUT / f"step1_baseline_{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n=== STEP 1 BASELINE:", MODEL, "===")
    print(f"Questions:            {len(df)}")
    print(f"Decision accuracy:    {df['decision_ok'].mean():.1%}")
    print(f"Must-include (avg):   {df['include'].mean():.1%}")
    print(f"All facts present:    {(df['include'] == 1).mean():.1%}")
    print(f"Forbidden-claim hits: {int(df['violation'].sum())}")
    print(f"Evidence recall:      {df['evidence_recall'].dropna().mean():.1%}")
    print(f"Avg latency:          {df['latency_s'].mean():.2f}s")
    print(f"Tokens used:          {tokens}")
    print("\nBy reply language:")
    print(df.groupby("reply_language")[["decision_ok", "include"]].mean().round(2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()