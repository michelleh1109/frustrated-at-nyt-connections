"""
Pull sent emails from Gmail → data/raw/threads.json

  python data/pull_gmail.py            # full pull
  python data/pull_gmail.py --limit 20 # test run

Saves raw decoded bodies — no cleaning. All stripping happens in clean.py
so you can re-clean without re-pulling.

Server-side filters (Google evaluates before any content transfers):
  - Only sent mail in the date window
  - Strips spam, trash, promotions, updates, social, forums

Client-side filters (after fetch, before saving):
  - Noreply senders as the input message
  - Threads with no sent messages
"""

import json
import os
import re
import time
from base64 import urlsafe_b64decode
from datetime import date, timedelta
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH       = os.getenv("GMAIL_TOKEN_PATH", "token.json")
USER             = os.getenv("GMAIL_USER", "me")

_default_cutoff = (date.today() - timedelta(days=3 * 365)).strftime("%Y/%m/%d")
AFTER_DATE = os.getenv("GMAIL_AFTER_DATE", _default_cutoff)

NOREPLY_RE = re.compile(
    r"(noreply|no-reply|donotreply|do-not-reply|notifications?@|mailer-daemon|"
    r"postmaster|bounce|automated|auto-confirm)",
    re.IGNORECASE,
)

RAW_OUT = Path("data/raw/threads.json")
RAW_OUT.parent.mkdir(parents=True, exist_ok=True)


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_service():
    creds = None
    if Path(TOKEN_PATH).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(CREDENTIALS_PATH).exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_PATH}.\n"
                    "Download from Google Cloud Console → APIs & Services → Credentials.\n"
                    "Choose 'Desktop app' as the application type."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ── Gmail helpers ─────────────────────────────────────────────────────────────

def list_sent_thread_ids(service, max_threads: int) -> list[str]:
    """Server-side filtered: Google evaluates the query before any content transfers."""
    thread_ids = []
    page_token = None
    query = (
        f"in:sent after:{AFTER_DATE} "
        "-label:spam -label:trash "
        "-category:promotions -category:updates -category:social -category:forums"
    )
    print(f"Query: {query!r}\n")
    while True:
        resp = service.users().threads().list(
            userId=USER,
            q=query,
            maxResults=min(500, max_threads - len(thread_ids)),
            pageToken=page_token,
        ).execute()
        for t in resp.get("threads", []):
            thread_ids.append(t["id"])
        page_token = resp.get("nextPageToken")
        print(f"  {len(thread_ids)} thread IDs fetched…", end="\r")
        if not page_token or len(thread_ids) >= max_threads:
            break
    print(f"\n  Total: {len(thread_ids)}")
    return thread_ids


def fetch_thread(service, thread_id: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            return service.users().threads().get(
                userId=USER,
                id=thread_id,
                format="full",
            ).execute()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ── Body decoding (raw, no cleaning) ─────────────────────────────────────────

def decode_body(payload: dict) -> str:
    """Decode base64 body to plain text. Saves raw — clean.py does the stripping."""
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return urlsafe_b64decode(data + "==").decode("utf-8", errors="replace") if data else ""
    if mime == "text/html":
        data = payload.get("body", {}).get("data", "")
        if not data:
            return ""
        html = urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        return BeautifulSoup(html, "html.parser").get_text(separator="\n")
    parts = payload.get("parts", [])
    plain = next((p for p in parts if p.get("mimeType") == "text/plain"), None)
    if plain:
        return decode_body(plain)
    html_part = next((p for p in parts if p.get("mimeType") == "text/html"), None)
    if html_part:
        return decode_body(html_part)
    for part in parts:
        text = decode_body(part)
        if text.strip():
            return text
    return ""


def header(msg: dict, name: str) -> str:
    for h in msg.get("payload", {}).get("headers", []):
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def parse_message(msg: dict) -> dict:
    return {
        "id": msg["id"],
        "thread_id": msg["threadId"],
        "timestamp": int(msg.get("internalDate", 0)) // 1000,
        "from": header(msg, "From"),
        "to": header(msg, "To"),
        "subject": header(msg, "Subject"),
        "label_ids": msg.get("labelIds", []),
        "body": decode_body(msg.get("payload", {})),
    }


# ── Client-side filters ───────────────────────────────────────────────────────

def is_noreply(sender: str) -> bool:
    return bool(NOREPLY_RE.search(sender))


def should_keep_thread(messages: list[dict]) -> bool:
    sent = [m for m in messages if "SENT" in m["label_ids"]]
    if not sent:
        return False
    # drop threads where the only non-noreply check fails on the preceding message
    my_reply = sent[-1]
    my_idx = messages.index(my_reply)
    if my_idx > 0 and is_noreply(messages[my_idx - 1]["from"]):
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_all_threads(service, thread_ids: list[str]) -> list[dict]:
    results, skipped, total = [], 0, len(thread_ids)
    for i, tid in enumerate(thread_ids):
        try:
            raw = fetch_thread(service, tid)
            messages = [parse_message(m) for m in raw.get("messages", [])]
            if should_keep_thread(messages):
                results.append({"thread_id": tid, "messages": messages})
            else:
                skipped += 1
        except Exception as e:
            print(f"\n  Skipping {tid}: {e}")
            skipped += 1
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} fetched — {len(results)} kept, {skipped} skipped…", end="\r")
    print(f"\n  Done — {len(results)} kept, {skipped} skipped")
    return results


def pull_all(max_threads: int = 5000):
    service = get_service()
    print(f"Pulling sent threads since {AFTER_DATE}…")
    thread_ids = list_sent_thread_ids(service, max_threads)
    threads = fetch_all_threads(service, thread_ids)
    print(f"\nWriting {len(threads)} threads → {RAW_OUT}")
    with open(RAW_OUT, "w") as f:
        json.dump(threads, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap threads fetched (e.g. --limit 20 for a test run)")
    args = parser.parse_args()

    if args.limit:
        print(f"[TEST MODE] limit={args.limit}")
    pull_all(max_threads=args.limit or 5000)
