"""
Clean raw threads → training pairs → data/cleaned/pairs.jsonl

  python data/clean.py

Input:  data/raw/threads.json  (output of pull_gmail.py)
Output: data/cleaned/pairs.jsonl — one JSON object per line:
  {
    "type": "reply" | "cold" | "broadcast",
    "register": "",      # filled in by pair_and_tag.py
    "input": "...",
    "output": "...",
    "thread_id": "...",
    "subject": "..."
  }
"""

import json
import re
from pathlib import Path

RAW_IN    = Path("data/raw/threads.json")
CLEAN_OUT = Path("data/cleaned/pairs.jsonl")
CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)

MIN_OUTPUT_WORDS = 20
MIN_INPUT_WORDS  = 5

BROADCAST_SUBJECT_RE = re.compile(
    r"(newsletter|announcement|reminder|update|recap|digest|invite|invitation|"
    r"event|launch|release)",
    re.IGNORECASE,
)

# "On Mon, Jun 17 ... wrote:" and everything after
QUOTE_HEADER_RE = re.compile(
    r"\s*On [A-Z][a-z]{2},? .{5,120} wrote:\s.*$",
    re.DOTALL,
)

# RFC 3676 signature delimiter + common closers
SIGNATURE_RE = re.compile(
    r"\s*-- ?\n.*$"
    r"|\s*(?:Best|Thank You|Thanks),?\s*\n?Michelle.*$",
    re.DOTALL | re.IGNORECASE,
)

URL_RE         = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE  = re.compile(r"\n{3,}")
BRACKET_IMG_RE = re.compile(r"\[https?://[^\]]+\]")
BOM_RE         = re.compile(r"[﻿ ]")


# ── Cleaning ──────────────────────────────────────────────────────────────────

def strip_quoted_chain(text: str) -> str:
    text = QUOTE_HEADER_RE.sub("", text)
    lines = [line for line in text.splitlines() if not line.startswith(">")]
    return "\n".join(lines).strip()


def strip_signature(text: str) -> str:
    return SIGNATURE_RE.sub("", text).strip()


def normalize(text: str) -> str:
    text = BOM_RE.sub(" ", text)
    text = BRACKET_IMG_RE.sub("", text)
    text = URL_RE.sub("[link]", text)
    text = WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


def clean_body(text: str) -> str:
    return normalize(strip_signature(strip_quoted_chain(text)))


# ── Pair extraction ───────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def is_mine(msg: dict) -> bool:
    return "SENT" in msg["label_ids"]


def recipient_count(msg: dict) -> int:
    return len([r for r in msg.get("to", "").split(",") if r.strip()])


def sender_name(msg: dict) -> str:
    raw = msg["from"]
    match = re.match(r"^(.+?)\s*<", raw)
    return match.group(1).strip().strip('"') if match else raw.split("@")[0]


def classify_initiated(msg: dict) -> str:
    subject = msg.get("subject", "")
    if recipient_count(msg) > 1 or BROADCAST_SUBJECT_RE.search(subject):
        return "broadcast"
    return "cold"


def make_prompt(msg_type: str, subject: str, n_recipients: int) -> str:
    if msg_type == "broadcast":
        return f"Write a broadcast email to {n_recipients} recipients. Subject: {subject}"
    return f"Write an outreach email. Subject: {subject}"


def extract_pairs(thread: dict) -> list[dict]:
    messages = thread["messages"]
    subject  = messages[0]["subject"] if messages else ""
    tid      = thread["thread_id"]
    pairs    = []

    for i, msg in enumerate(messages):
        if not is_mine(msg):
            continue

        output = clean_body(msg["body"])
        if word_count(output) < MIN_OUTPUT_WORDS:
            continue

        if i == 0:
            msg_type = classify_initiated(msg)
            pairs.append({
                "type": msg_type,
                "register": "",
                "input": make_prompt(msg_type, subject, recipient_count(msg)),
                "output": output,
                "thread_id": tid,
                "subject": subject,
            })
            continue

        preceding = next(
            (messages[j] for j in range(i - 1, -1, -1) if not is_mine(messages[j])),
            None,
        )
        if preceding is None:
            continue

        input_text = clean_body(preceding["body"])
        if word_count(input_text) < MIN_INPUT_WORDS:
            msg_type = classify_initiated(msg)
            pairs.append({
                "type": msg_type,
                "register": "",
                "input": make_prompt(msg_type, subject, recipient_count(msg)),
                "output": output,
                "thread_id": tid,
                "subject": subject,
            })
            continue

        pairs.append({
            "type": "reply",
            "register": "",
            "input": input_text,
            "output": output,
            "thread_id": tid,
            "subject": subject,
            "from_name": sender_name(preceding),
        })

    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def clean():
    with open(RAW_IN) as f:
        threads = json.load(f)

    all_pairs = []
    for thread in threads:
        all_pairs.extend(extract_pairs(thread))

    counts = {t: sum(1 for p in all_pairs if p["type"] == t)
              for t in ("reply", "cold", "broadcast")}

    print(f"Threads processed: {len(threads)}")
    print(f"Pairs extracted:   {len(all_pairs)}")
    print(f"  reply:     {counts['reply']}")
    print(f"  cold:      {counts['cold']}")
    print(f"  broadcast: {counts['broadcast']}")

    with open(CLEAN_OUT, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWritten → {CLEAN_OUT}")


if __name__ == "__main__":
    clean()
