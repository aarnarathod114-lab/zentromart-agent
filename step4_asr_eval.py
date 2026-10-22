"""Step 4 evaluation: robustness to speech-recognition errors.
Runs Step 2's exact RAG pipeline on the dataset's ASR-style text (asr_text) instead of the
clean question, so results compare directly with step2_rag_<model>.csv.
Usage: python step4_asr_eval.py [model] [limit]
"""
import time
from pathlib import Path

import pandas as pd

import step2_rag as s2          # same model, prompts, retrieval and calendar as Step 2
from retriever import Retriever
from score import decision_ok, include_score, violates, evidence_recall


def main():
    q = pd.read_csv(s2.DATA / "practice" / "questions.csv", dtype=str).fillna("")
    if s2.LIMIT:
        q = q.head(s2.LIMIT)
    print("Loading retriever...")
    retriever = Retriever()

    rows, tokens = [], 0
    for _, row in q.iterrows():
        text = row["asr_text"] or row["question"]
        t0 = time.time()
        query_en, used1 = s2.rewrite(text)
        items, context = s2.build_context(retriever, text, query_en)
        data, used2 = s2.answer(text, context)
        latency = time.time() - t0
        tokens += used1 + used2

        decision = str(data.get("decision", "")).strip().lower()
        reply = str(data.get("reply", ""))
        evidence = data.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        rec = {
            "id": row["id"], "reply_language": row["reply_language"],
            "expected_decision": row["expected_decision"], "decision": decision,
            "decision_ok": decision_ok(decision, row["acceptable_decisions"]),
            "include": include_score(reply, row["must_include"]),
            "violation": violates(reply, row["must_not_include"]),
            "evidence_recall": evidence_recall(evidence, row["gold_evidence"]),
            "retrieval_recall": evidence_recall([c["id"] for c in items], row["gold_evidence"]),
            "asr_text": text, "query_en": query_en,
            "evidence": " ; ".join(map(str, evidence)),
            "latency_s": round(latency, 2), "reply": reply,
        }
        rows.append(rec)
        print(f"{rec['id']}  {decision:9} ok={rec['decision_ok']!s:5} include={rec['include']:.2f}  {latency:.1f}s")

    df = pd.DataFrame(rows)
    name = s2.MODEL.split("/")[-1]
    path = s2.OUT / f"step4_asr_{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n=== STEP 4 ASR ROBUSTNESS:", s2.MODEL, "===")
    print(f"Questions:            {len(df)}")
    print(f"Decision accuracy:    {df['decision_ok'].mean():.1%}")
    print(f"Must-include (avg):   {df['include'].mean():.1%}")
    print(f"Forbidden-claim hits: {int(df['violation'].sum())}")
    print(f"Evidence recall:      {df['evidence_recall'].dropna().mean():.1%}")
    print(f"Retrieval recall:     {df['retrieval_recall'].dropna().mean():.1%}")
    print(f"Tokens used:          {tokens}")

    clean = s2.OUT / f"step2_rag_{name}.csv"
    if clean.exists():
        c = pd.read_csv(clean)
        print("\nClean text (Step 2) vs ASR text (Step 4):")
        print(f"  Decision accuracy: {c['decision_ok'].mean():.1%} -> {df['decision_ok'].mean():.1%}")
        print(f"  Must-include:      {c['include'].mean():.1%} -> {df['include'].mean():.1%}")
        print(f"  Retrieval recall:  {c['retrieval_recall'].dropna().mean():.1%} -> {df['retrieval_recall'].dropna().mean():.1%}")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()