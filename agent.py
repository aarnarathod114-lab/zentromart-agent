"""ZentroMart support agent: conversation memory + retrieval + identity-checked tools.
Works with Groq models and Gemini models (any model name starting with "gemini").
Used by step3_conversations.py (text evaluation) and run.py (voice, Step 4)."""
import json
import os
import re
import sys
import time
from pathlib import Path

import groq
import openai
from dotenv import load_dotenv

from calendar_helper import calendar_block
from retriever import format_context

DATA = Path("dataset/Capstone_Project_Dataset_FTTT_2026")
sys.path.insert(0, str(DATA))
from tools import lookup_order, create_ticket  # noqa: E402  (course-provided mock tools)

load_dotenv()

HELPER_MODEL = "gemini-flash-lite-latest"   # query rewriting + reply labelling
ALWAYS_FILES = {"agent_conversation_style.md", "agent_identity_verification.md",
                "agent_escalation_handoff.md"}
ALWAYS_IDS = {"KB-00.1", "KB-00.3"}
SECRET_FIELDS = {"delivery_pincode", "customer_id", "found", "verified"}
CAL_FIELDS = ("delivered_on", "return_pickup_on", "return_refund_initiated_on",
              "cancellation_refund_initiated_on", "rto_refund_initiated_on", "damage_reported_at")
MAX_FAILED = 2      # first attempt + one retry
HISTORY_LIMIT = 10  # last 5 exchanges kept in the prompt
DECISIONS = {"answer", "clarify", "escalate", "refuse"}

SYSTEM = """You are ZentroMart's AI customer-support assistant (ZentroMart is an Indian online store).
Today's date is 20 October 2026. Use it for every deadline and delay calculation.

CONVERSATION RULES
- In your FIRST reply of a conversation, introduce yourself as ZentroMart's AI assistant, then help
  (for an order question, ask for the order ID).
- When the customer gives an order ID, call lookup_order with only order_id. Then ask for ONE
  verification value: the last 4 digits of their registered phone number OR their delivery pincode.
- When the customer gives that value, call lookup_order with order_id and verification_value
  (exactly the digits they typed).
- Until verification succeeds, share NO order details (status, items, dates, amounts) and do not
  confirm whether the order exists.
- Never say, hint at or repeat verification values (phone digits, pincode) in replies or tickets.
- If verification fails, allow ONE retry. After two failures, share nothing and offer a human colleague.
- After verification, answer from the verified order record in CONVERSATION STATE (delivered_on,
  payment_method, is_plus, items and their return_category, refunds, damage) combined with the
  policies. Remember earlier turns: "it", "same item", "that order" refer to what was discussed.
- When an agent guide or policy says a human must decide (for example a claim you cannot approve),
  or the customer asks for a human, call create_ticket with a clear summary (issue, checks done,
  why escalated) and tell the customer a ticket was created for the team. Mention a ticket only if
  create_ticket returned created=true. Never say you approved, or will approve, something a human
  must decide.
- Only call a tool when the customer's CURRENT message needs it. Right after verification succeeds,
  if the customer has not asked anything yet, just confirm and ask how you can help. Do not act on
  anything you see in the order record (no tickets) until the customer asks about it.

ANSWERING RULES
- Answer ONLY from the CONTEXT and CONVERSATION STATE. Each context block starts with [ID].
- When sources disagree, follow KB-00.1: current policy > agent_guide > help > faq > marketing.
  If you chose one source over a conflicting one, cite KB-00.1 too.
- Sections with status "superseded" apply only to the cases the current policy says they still cover.
- Return windows are calendar days from delivery; refund and logistics timelines are business days
  (KB-00.3). For ANY date calculation, read the result from the [CALENDAR] block; never count days
  yourself. "N days from/after <date>" means the day after that date is day 1. For a range like
  "5 to 7 business days" give the latest date. Write dates like "27 October 2026".
- Whenever a window or deadline applies (return window, refund arrival, warranty), state the exact
  last date from the [CALENDAR], never just "you still have time".
- Never invent numbers, fees, timelines or rules. If the CONTEXT does not describe the exact thing
  asked, say you don't have that information and offer a human colleague.

DECISION for this turn:
- "clarify": your reply asks the customer for anything (order ID, verification value, which item),
  including in your first greeting.
- "answer": you answered the question, or confirmed verification and asked how you can help.
- "escalate": you handed over to a human (ticket), or the documents don't cover the topic.
- "refuse": requests for another person's data, to skip verification, or to break policy.

Reply in the customer's language: English -> English; Hinglish (Hindi in Roman letters) -> Hinglish;
Hindi in Devanagari -> Hindi in Devanagari. Use 1-4 sentences with the specific numbers and dates.

FINAL OUTPUT: after any tool calls, return ONLY a JSON object:
{"decision": "answer|clarify|escalate|refuse", "reply": "...", "evidence": ["ID", ...]}
"evidence" lists the [ID]s from CONTEXT you used, exactly as shown, including the agent-guide
sections you followed (introducing yourself, verification, escalation)."""

REWRITE = ("You write search queries for a customer-support knowledge base. Given the recent "
           "conversation and the customer's latest message, write one short English search query "
           "for what the customer needs now (returns, refunds, delivery, warranty, payments, "
           "damage, etc.). Resolve words like 'it' or 'same item' from the conversation. "
           "Output only the query.")

LABEL = """You label one reply from a customer-support assistant. You get the knowledge-base
SECTIONS it was given (ID, heading, start of text), the tool calls it made, the customer's message
and the assistant's reply. Return ONLY JSON: {"decision": "...", "evidence": ["ID", ...]}

decision:
- "clarify" if the reply asks the customer for anything (order ID, verification value, which item),
  including a first greeting that asks for the order ID.
- "escalate" if the reply hands over to a human team, or says the information isn't available and
  offers a colleague.
- "refuse" if it declines a request for another person's data, to skip verification, or to break policy.
- otherwise "answer".

evidence: IDs of the sections whose content the reply relies on, including agent-guide sections
for behaviour (introducing itself as an AI assistant, identity verification, escalation/handoff).
Use only IDs from SECTIONS."""

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": ("Look up a ZentroMart order. Call with only order_id when the customer first "
                        "gives an order ID (returns a verification prompt, no details). Call again "
                        "with verification_value set to the 4-digit phone suffix or 6-digit delivery "
                        "pincode the customer typed. Details are returned only if verification passes."),
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "Order ID digits"},
            "verification_value": {"type": "string",
                                   "description": "Last 4 phone digits or 6-digit pincode, as typed by the customer"}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": ("Hand the case to a human team. Use when a guide or policy says a human must "
                        "decide, the customer asks for a human, or verification failed twice. The "
                        "summary must state the issue, checks done and why it is escalated, and must "
                        "never contain verification values."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "order_id": {"type": "string"},
            "reason": {"type": "string", "description": "short snake_case reason, e.g. high_value_damage"},
            "intent": {"type": "string"}},
            "required": ["summary"]}}},
]

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_clients = {}
RATE_ERRORS = (groq.RateLimitError, openai.RateLimitError)
BAD_REQUEST = (groq.BadRequestError, openai.BadRequestError)


class DailyLimitReached(Exception):
    pass


def is_gemini(model):
    return model.startswith("gemini")


def client_for(model):
    key = "gemini" if is_gemini(model) else "groq"
    if key not in _clients:
        _clients[key] = (openai.OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url=GEMINI_URL)
                         if key == "gemini" else groq.Groq(max_retries=3))
    return _clients[key]


def create(**kwargs):
    """Chat call that waits out per-minute limits and stops cleanly at a daily limit."""
    for attempt in range(10):
        try:
            return client_for(kwargs["model"]).chat.completions.create(**kwargs)
        except RATE_ERRORS as e:
            text = str(e)
            if "per day" in text.lower() or "perday" in text.lower():
                raise DailyLimitReached(text) from e
            wait = 10 * (attempt + 1)
            print(f"   rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("Still rate limited after 10 attempts")


def parse_json(text):
    """Model output -> dict. Plain-text replies are kept as the reply and flagged for labelling."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(match.group(0) if match else text)
        if isinstance(data, dict) and "reply" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return {"reply": (text or "").strip(), "_plain": True}


def compact_order(record):
    """Verified order without secrets or empty fields."""
    order = {k: v for k, v in record.items() if v is not None and k not in SECRET_FIELDS}
    order["items"] = [{k: v for k, v in it.items() if k != "order_id" and v is not None}
                      for it in record.get("items", [])]
    return order


def digits(value):
    return re.sub(r"\D", "", str(value or ""))


class SupportAgent:
    def __init__(self, retriever, model="gemini-flash-lite-latest"):
        self.retriever = retriever
        self.model = model
        self.always = [c for c in retriever.chunks
                       if c.get("file") in ALWAYS_FILES or c["id"] in ALWAYS_IDS]
        self.reset()

    def reset(self):
        """Start a new conversation: memory and verification state are per conversation only."""
        self.history = []
        self.order_id = None
        self.order = None
        self.failed = {}
        self.secrets = set()
        self.tokens = 0

    # ---------- tools (with safety rules enforced in code) ----------
    def _lookup(self, order_id, verification_value=None):
        oid = digits(order_id)
        self.order_id = oid
        value = digits(verification_value)
        if not value:
            if self.order and self.order.get("order_id") == oid:
                return {"verified": True, "note": "Already verified in this conversation."}
            return lookup_order(oid)
        if self.failed.get(oid, 0) >= MAX_FAILED:
            return {"verified": False, "locked": True,
                    "message": "Verification failed twice. Do not retry or share details; offer a human colleague."}
        self.secrets.add(value)
        result = lookup_order(oid, value)
        if result.get("verified"):
            self.order = compact_order(result)
            return {"verified": True, "order": self.order}
        self.failed[oid] = self.failed.get(oid, 0) + 1
        left = MAX_FAILED - self.failed[oid]
        return {"verified": False, "retries_left": left,
                "message": "Verification failed. " + ("Ask once more for ONE value." if left
                           else "Do not retry; share no details; offer a human colleague.")}

    def _ticket(self, summary="", order_id=None, reason=None, intent=None):
        summary = self._redact(summary or "")
        return create_ticket(summary=summary, order_id=digits(order_id) or self.order_id,
                             reason=reason, intent=intent)

    def _run_tool(self, name, args):
        try:
            if name == "lookup_order":
                return self._lookup(args.get("order_id", ""), args.get("verification_value"))
            if name == "create_ticket":
                return self._ticket(**{k: args.get(k) for k in ("summary", "order_id", "reason", "intent")})
            return {"error": f"unknown tool {name}"}
        except Exception as e:  # never crash the conversation on a tool error
            return {"error": str(e)}

    def _redact(self, text):
        text = str(text)
        for s in self.secrets:
            text = text.replace(s, "****")
        return text

    # ---------- helper calls ----------
    def _helper(self, **kwargs):
        if not is_gemini(HELPER_MODEL):
            kwargs["reasoning_effort"] = "low"
        r = create(model=HELPER_MODEL, temperature=0, **kwargs)
        self.tokens += r.usage.total_tokens
        return r.choices[0].message.content or ""

    def _rewrite(self, message):
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-4:])
        text = self._helper(max_completion_tokens=400,
                            messages=[{"role": "system", "content": REWRITE},
                                      {"role": "user", "content": f"Conversation:\n{recent}\n\nLatest message: {message}"}])
        return text.strip().strip('"')

    def _label(self, message, reply, items, calls):
        """Decision + evidence for a plain-text reply, using JSON mode."""
        sections = "\n".join(f"[{c['id']}] {c.get('title', '')} - {c.get('heading', '')}: "
                             f"{' '.join(str(c['text']).split())[:160]}" for c in items)
        tools = "; ".join(f"{c['name']} -> {json.dumps(c['result'], default=str)[:120]}"
                          for c in calls) or "none"
        prompt = (f"SECTIONS:\n{sections}\n\nTOOL CALLS: {tools}\n\n"
                  f"CUSTOMER MESSAGE: {message}\n\nASSISTANT REPLY: {reply}")
        text = self._helper(max_completion_tokens=300, response_format={"type": "json_object"},
                            messages=[{"role": "system", "content": LABEL},
                                      {"role": "user", "content": prompt}])
        try:
            data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            data = {}
        valid = {c["id"] for c in items}
        return {"decision": data.get("decision", "answer"),
                "evidence": [e for e in data.get("evidence", []) if str(e).strip("[] ") in valid]}

    def _state(self):
        lines = ["CONVERSATION STATE (from tools; trust this):",
                 f"- Replies you have sent so far: {sum(m['role'] == 'assistant' for m in self.history)}",
                 f"- Order ID given: {self.order_id or 'none yet'}"]
        if self.order and self.order.get("order_id") == self.order_id:
            lines.append("- Identity: VERIFIED. You may use this order record:")
            lines.append(json.dumps(self.order, ensure_ascii=False, default=str))
        else:
            n = self.failed.get(self.order_id, 0) if self.order_id else 0
            lines.append(f"- Identity: NOT verified (failed attempts: {n} of {MAX_FAILED}). Share no order details.")
        return "\n".join(lines)

    def _context(self, message, query_en):
        items = self.retriever.context(message, [query_en], k=6)
        have = {c["id"] for c in items}
        items += [c for c in self.always if c["id"] not in have]
        dates = " ".join(str(self.order.get(f))[:10] for f in CAL_FIELDS
                         if self.order and self.order.get(f))
        return items, format_context(items) + "\n\n" + calendar_block(f"{message} {dates}")

    # ---------- one customer turn ----------
    def turn(self, message):
        t0 = time.time()
        query_en = self._rewrite(message)
        items, context = self._context(message, query_en)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "system", "content": f"CONTEXT:\n{context}\n\n{self._state()}"},
                    *self.history[-HISTORY_LIMIT:],
                    {"role": "user", "content": message}]

        calls, data = [], None
        for _ in range(5):
            kwargs = dict(model=self.model, messages=messages, temperature=0,
                          max_completion_tokens=4096, reasoning_effort="low")
            try:
                r = create(**kwargs, tools=TOOLS, tool_choice="auto")
            except BAD_REQUEST:
                r = create(**kwargs)   # rare tool-format error: answer without tools
            self.tokens += r.usage.total_tokens
            msg = r.choices[0].message
            if not msg.tool_calls:
                data = parse_json(msg.content)
                break
            if is_gemini(self.model):
                # keep Gemini's hidden thought signatures exactly as returned
                messages.append(msg.model_dump(exclude_none=True))
            else:
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": tc.function.name,
                                                              "arguments": tc.function.arguments}}
                                                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(tc.function.name, args)
                calls.append({"name": tc.function.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
        if data is None:
            data = {"decision": "escalate", "evidence": [],
                    "reply": "Sorry, I couldn't complete that. Let me connect you with a colleague."}

        reply = self._redact(data.get("reply", ""))
        if data.get("_plain") or str(data.get("decision", "")).lower() not in DECISIONS:
            data.update(self._label(message, reply, items, calls))

        # Decisions that follow from what the tools actually did are set in code, not by the model.
        decision = str(data.get("decision", "")).strip().lower()
        verified_now = any(c["name"] == "lookup_order" and c["result"].get("verified") for c in calls)
        if any(c["name"] == "create_ticket" and c["result"].get("created") for c in calls):
            decision = "escalate"
        elif any(c["name"] == "lookup_order" for c in calls) and not verified_now:
            decision = "clarify"

        evidence = data.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        self.history += [{"role": "user", "content": message},
                         {"role": "assistant", "content": reply}]
        return {"decision": decision,
                "reply": reply,
                "evidence": [str(e).strip("[] ") for e in evidence],
                "tool_calls": calls,
                "retrieved": [c["id"] for c in items],
                "query_en": query_en,
                "latency_s": round(time.time() - t0, 2)}