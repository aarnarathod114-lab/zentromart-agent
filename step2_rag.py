"""Step 2: retrieval-augmented agent (hybrid search over KB + catalog + business-day calendar).
Usage: python step2_rag.py [model] [limit]
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, BadRequestError, RateLimitError

from calendar_helper import calendar_block
from retriever import Retriever, format_context
from score import decision_ok, include_score, violates, evidence_recall

load_dotenv()
MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
REWRITE_MODEL = "openai/gpt-oss-20b"
TOP_K = 8
RULE_IDS = ("KB-00.1", "KB-00.3")   # precedence + business-day rules, always in context
DATA = Path("dataset/Capstone_Project_Dataset_FTTT_2026")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

SYSTEM = """You are the customer support agent for ZentroMart, an Indian online store.
Today's date is 20 October 2026. Use it for every deadline and delay calculation.

Answer ONLY from the CONTEXT. Each context block starts with [ID] and shows its type and status.
Rules:
- When sources disagree, follow KB-00.1: current policy > agent_guide > help > faq > marketing.
  Marketing pages never override a policy. If you chose one source over a conflicting one,
  cite KB-00.1 too.
- Sections with status "superseded" apply only to the cases the current policy says they still
  cover (for example, items delivered before a policy change date). Otherwise use the current one.
- Return windows are calendar days from delivery. Refund and logistics timelines are business
  days (KB-00.3). For ANY date calculation, read the answer from the [CALENDAR] block; never
  count days yourself. Write dates like "27 October 2026". Cite KB-00.3 whenever you use
  business days.
- Combine a policy section with a catalog or pincode row when the answer needs both
  (for example a product's return_category, warranty or seller).
- Never invent numbers, fees, timelines or rules that are not in the CONTEXT.
- Before answering, check that the CONTEXT explicitly describes the exact thing asked (a time
  slot, a service, a fee). General knowledge about shops or couriers is NOT enough: if it is not described, choose "escalate".
- Counting rule: "N business days from <date>" means the first business day AFTER that date is
  day 1, so the start date itself is never counted. Use the [CALENDAR] line
  "N business days after <date>" and copy its date exactly; give the LATEST date of a range
  (for "5 to 7 business days" give the 7-day date).
Decision:
- "answer": the CONTEXT answers the question.
- "clarify": the CONTEXT covers the topic, but the answer depends on a detail only the
  customer can give (for example which product or when it was delivered). Ask ONE short question.
- "escalate": the CONTEXT does not cover the topic at all, or a document says a human must
  handle it. Say plainly that you don't have that information and offer to connect the customer
  with a human colleague. Never ask a question about a topic the documents don't cover.
- "refuse": requests to share another person's data, skip identity verification, or break policy.

Reply in the customer's language: English -> English; Hinglish (Hindi in Roman letters) ->
Hinglish; Hindi in Devanagari script -> Hindi in Devanagari. Use 2-4 sentences and include the
specific numbers (days, rupee amounts, dates) from the context.

Return ONLY a JSON object:
{"decision": "answer|clarify|escalate|refuse", "reply": "...", "evidence": ["ID", ...]}
"evidence" lists the [ID]s from the CONTEXT you actually used, written exactly as shown."""

REWRITE = ("Rewrite the customer's message as one short English search query for a customer "
           "support knowledge base (returns, refunds, delivery, warranty, payments, etc.). "
           "Name the topic and payment method or product type if mentioned. "
           "Keep any SKU, pincode or order ID. Output only the query.")

client = Groq(max_retries=3)


def create(**kwargs):
    """Chat call that waits out per-minute rate limits."""
    for attempt in range(10):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"   rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("Still rate limited after 10 attempts")


def rewrite(question):
    r = create(model=REWRITE_MODEL, temperature=0, max_completion_tokens=400,
               reasoning_effort="low",
               messages=[{"role": "system", "content": REWRITE},
                         {"role": "user", "content": question}])
    return (r.choices[0].message.content or "").strip().strip('"'), r.usage.total_tokens


def build_context(retriever, question, query_en):
    items = retriever.context(question, [query_en], k=TOP_K)
    have = {c["id"] for c in items}
    items += [c for c in retriever.chunks if c["id"] in RULE_IDS and c["id"] not in have]
    return items, format_context(items) + "\n\n" + calendar_block(question)


def answer(question, context):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nCUSTOMER MESSAGE:\n{question}"}]
    kwargs = dict(model=MODEL, messages=messages, temperature=0,
                  max_completion_tokens=2048, reasoning_effort="low")
    try:
        r = create(**kwargs, response_format={"type": "json_object"})
    except BadRequestError:
        r = create(**kwargs)
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

    print("Loading retriever...")
    retriever = Retriever()

    rows, tokens = [], 0
    for _, row in q.iterrows():
        t0 = time.time()
        query_en, used1 = rewrite(row["question"])
        items, context = build_context(retriever, row["question"], query_en)
        retrieved = [c["id"] for c in items]
        data, used2 = answer(row["question"], context)
        latency = time.time() - t0
        tokens += used1 + used2

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
            "retrieval_recall": evidence_recall(retrieved, row["gold_evidence"]),
            "gold_evidence": row["gold_evidence"],
            "evidence": " ; ".join(map(str, evidence)),
            "retrieved": " ; ".join(retrieved),
            "query_en": query_en,
            "latency_s": round(latency, 2),
            "reply": reply,
        }
        rows.append(rec)
        rr = rec["retrieval_recall"]
        print(f"{rec['id']}  {decision:9} ok={rec['decision_ok']!s:5} "
              f"include={rec['include']:.2f}  retr={'-' if rr is None else f'{rr:.2f}'}  {latency:.1f}s")

    df = pd.DataFrame(rows)
    name = MODEL.split("/")[-1]
    path = OUT / f"step2_rag_{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n=== STEP 2 RAG:", MODEL, "===")
    print(f"Questions:            {len(df)}")
    print(f"Decision accuracy:    {df['decision_ok'].mean():.1%}")
    print(f"Must-include (avg):   {df['include'].mean():.1%}")
    print(f"All facts present:    {(df['include'] == 1).mean():.1%}")
    print(f"Forbidden-claim hits: {int(df['violation'].sum())}")
    print(f"Evidence recall:      {df['evidence_recall'].dropna().mean():.1%}")
    print(f"Retrieval recall:     {df['retrieval_recall'].dropna().mean():.1%}")
    print(f"Avg latency:          {df['latency_s'].mean():.2f}s")
    print(f"Tokens used:          {tokens}")
    print("\nBy reply language:")
    print(df.groupby("reply_language")[["decision_ok", "include"]].mean().round(2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()