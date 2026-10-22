"""Show details for chosen questions from a results CSV.
Usage: python inspect_results.py results/step2_rag_gpt-oss-20b.csv PQ0003 PQ0004
       python inspect_results.py results/step2_rag_gpt-oss-20b.csv fails
"""
import sys

import pandas as pd

QUESTIONS = r"dataset\Capstone_Project_Dataset_FTTT_2026\practice\questions.csv"

res = pd.read_csv(sys.argv[1], dtype=str).fillna("")
q = pd.read_csv(QUESTIONS, dtype=str).fillna("")
cols = ["id", "question", "acceptable_decisions", "must_include", "must_not_include", "notes"]
if "gold_evidence" not in res.columns:
    cols.append("gold_evidence")
m = res.merge(q[cols], on="id")

ids = sys.argv[2:]
if ids == ["fails"]:
    m = m[(m["decision_ok"] != "True") | (m["include"].astype(float) < 1)]
elif ids:
    m = m[m["id"].isin(ids)]

for x in m.to_dict("records"):
    print(f"--- {x['id']} | expected {x['acceptable_decisions']} | got {x['decision']}")
    print(f"Q:            {x['question']}")
    if x.get("query_en"):
        print(f"query_en:     {x['query_en']}")
    print(f"gold:         {x.get('gold_evidence', '')}")
    if x.get("retrieved"):
        print(f"retrieved:    {x['retrieved']}")
    print(f"cited:        {x['evidence']}")
    print(f"must_include: {x['must_include']}")
    if x["must_not_include"]:
        print(f"must_not:     {x['must_not_include']}")
    if x["notes"]:
        print(f"notes:        {x['notes']}")
    print(f"reply:        {x['reply']}")
    print()