"""Step 3: multi-turn conversations with memory and identity-checked tools.
Usage: python step3_conversations.py [model] [max_conversations]
Resumable: conversations already in the results file are skipped.
"""
import re
import sys
from pathlib import Path

import pandas as pd

from agent import SupportAgent, DailyLimitReached
from retriever import Retriever
from score import decision_ok, include_score, violates, evidence_recall, split_list

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
MAX_CONV = int(sys.argv[2]) if len(sys.argv) > 2 else None
DATA = Path("dataset/Capstone_Project_Dataset_FTTT_2026")
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def parse_expected(cell):
    """'lookup_order(order_id=8978 2837, verification_value=4533)' -> [('lookup_order', {...})]"""
    calls = []
    for part in split_list(cell):
        m = re.match(r"\s*(\w+)\((.*)\)\s*$", part)
        if m:
            args = {k: re.sub(r"\D", "", v) or v.strip()
                    for k, v in re.findall(r"(\w+)\s*=\s*([^,]+)", m.group(2))}
            calls.append((m.group(1), args))
    return calls


def call_matches(name, args, actual):
    if actual["name"] != name:
        return False
    got = actual["args"]
    if name == "lookup_order":
        if bool(re.sub(r"\D", "", str(got.get("verification_value") or ""))) != ("verification_value" in args):
            return False
    return all(re.sub(r"\D", "", str(got.get(k, ""))) == v for k, v in args.items())


def tool_score(expected, actual):
    """(all expected calls made?, number of create_ticket calls that were not expected)"""
    ok = all(any(call_matches(n, a, c) for c in actual) for n, a in expected)
    extra = sum(c["name"] == "create_ticket" for c in actual) - [n for n, _ in expected].count("create_ticket")
    return ok, max(0, extra)


def fmt_calls(calls):
    return " ; ".join(f"{c['name']}({', '.join(f'{k}={v}' for k, v in c['args'].items() if k != 'summary')})"
                      for c in calls)


def summarize(path):
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ("decision_ok", "tools_ok", "violation"):
        df[col] = df[col] == "True"
    df["include"] = df["include"].astype(float)
    df["extra_tickets"] = df["extra_tickets"].astype(int)
    df["latency_s"] = df["latency_s"].astype(float)
    ev = pd.to_numeric(df["evidence_recall"], errors="coerce")
    df["turn_ok"] = df["decision_ok"] & df["tools_ok"] & (df["include"] == 1) & ~df["violation"]
    with_tools = df[df["expected_tools"] != ""]

    print("\n=== STEP 3 CONVERSATIONS:", MODEL, "===")
    print(f"Conversations / turns:   {df['conversation_id'].nunique()} / {len(df)}")
    print(f"Decision accuracy:       {df['decision_ok'].mean():.1%}")
    print(f"Must-include (avg):      {df['include'].mean():.1%}")
    print(f"Forbidden-claim hits:    {int(df['violation'].sum())}")
    print(f"Evidence recall:         {ev.dropna().mean():.1%}")
    print(f"Expected tool calls made:{with_tools['tools_ok'].mean():.1%}  ({len(with_tools)} turns)")
    print(f"Unexpected tickets:      {int(df['extra_tickets'].sum())}")
    print(f"Turns fully correct:     {df['turn_ok'].mean():.1%}")
    print(f"Conversations all-correct: {df.groupby('conversation_id')['turn_ok'].all().mean():.1%}")
    print(f"Avg latency per turn:    {df['latency_s'].mean():.2f}s")
    print("\nBy language:")
    print(df.groupby("language")[["decision_ok", "include", "turn_ok"]].mean().round(2))
    print(f"\nSaved: {path}")


def main():
    conv = pd.read_csv(DATA / "practice" / "conversations.csv", dtype=str).fillna("")
    conv["turn_n"] = conv["turn"].astype(int)
    path = OUT / f"step3_conversations_{MODEL.split('/')[-1]}.csv"
    done = set(pd.read_csv(path, dtype=str)["conversation_id"]) if path.exists() else set()
    todo = [c for c in conv["conversation_id"].drop_duplicates() if c not in done]
    if MAX_CONV:
        todo = todo[:MAX_CONV]
    print(f"{len(done)} conversations already done; running {len(todo)} now.")

    print("Loading retriever...")
    agent = SupportAgent(Retriever(), MODEL)
    try:
        for cid in todo:
            agent.reset()
            rows = []
            for _, row in conv[conv["conversation_id"] == cid].sort_values("turn_n").iterrows():
                out = agent.turn(row["customer_message"])
                expected = parse_expected(row["expected_tool_calls"])
                tools_ok, extra = tool_score(expected, out["tool_calls"])
                rec = {
                    "conversation_id": cid, "turn": row["turn"], "scenario": row["scenario"],
                    "language": row["language"],
                    "expected_decision": row["expected_decision"], "decision": out["decision"],
                    "decision_ok": decision_ok(out["decision"], row["expected_decision"]),
                    "include": include_score(out["reply"], row["must_include"]),
                    "violation": violates(out["reply"], row["must_not_include"]),
                    "evidence_recall": evidence_recall(out["evidence"], row["gold_evidence"]),
                    "expected_tools": row["expected_tool_calls"], "tools": fmt_calls(out["tool_calls"]),
                    "tools_ok": tools_ok, "extra_tickets": extra,
                    "gold_evidence": row["gold_evidence"], "evidence": " ; ".join(out["evidence"]),
                    "retrieved": " ; ".join(out["retrieved"]), "query_en": out["query_en"],
                    "latency_s": out["latency_s"],
                    "message": row["customer_message"], "reply": out["reply"],
                }
                rows.append(rec)
                print(f"{cid} t{row['turn']}  {out['decision']:9} ok={rec['decision_ok']!s:5} "
                      f"tools={tools_ok!s:5} include={rec['include']:.2f}  {out['latency_s']:.1f}s  "
                      f"[{rec['tools']}]")
            enc = "utf-8" if path.exists() else "utf-8-sig"
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False, encoding=enc)
    except DailyLimitReached:
        print("\nDaily token limit reached. Finished conversations are saved;"
              " rerun the same command after 5:30 AM IST to continue.")
    summarize(path)


if __name__ == "__main__":
    main()