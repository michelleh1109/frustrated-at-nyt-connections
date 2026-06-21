"""
Split pairs.jsonl into eval holdout + training set.

  python data/holdout_split.py

Samples 50 pairs stratified by output length. Short replies are capped so
scheduling one-liners don't dominate the holdout; longer outputs (cold emails,
substantive replies) are kept well-represented without draining training data.

Bucket allocations (total = 50):
  short     < 50 words  →  8   (~2% drain on 339 pairs)
  medium   50-100 words → 14   (~5% drain on 262 pairs)
  long    100-250 words → 18   (~8% drain on 216 pairs)
  very long  250+ words → 10   (~18% drain on 55 pairs)

Output:
  data/eval_holdout/holdout.jsonl   — 50 sacred pairs, never seen by model
  data/cleaned/train.jsonl          — everything else
"""

import json
import random
from collections import defaultdict
from pathlib import Path

PAIRS_IN    = Path("data/cleaned/pairs.jsonl")
HOLDOUT_OUT = Path("data/eval_holdout/holdout.jsonl")
TRAIN_OUT   = Path("data/cleaned/train.jsonl")
HOLDOUT_OUT.parent.mkdir(parents=True, exist_ok=True)

HOLDOUT_SIZE = 50
RANDOM_SEED  = 42

# (min_words_inclusive, max_words_exclusive, target_count)
LENGTH_BUCKETS = [
    (0,   50,   8),
    (50,  100, 14),
    (100, 250, 18),
    (250, 999999, 10),
]


def load_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(pairs: list[dict], path: Path):
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


def word_count(pair: dict) -> int:
    return len(pair["output"].split())


def length_stratified_sample(pairs: list[dict], n: int) -> list[dict]:
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        wc = word_count(p)
        for i, (lo, hi, _) in enumerate(LENGTH_BUCKETS):
            if lo <= wc < hi:
                by_bucket[i].append(p)
                break

    sampled = []
    for i, (_, _, target) in enumerate(LENGTH_BUCKETS):
        bucket = by_bucket[i]
        take = min(target, len(bucket))
        sampled.extend(random.sample(bucket, take))

    # top up from longest bucket first if we came up short
    if len(sampled) < n:
        already = set(id(p) for p in sampled)
        remaining = [p for p in pairs if id(p) not in already]
        remaining.sort(key=word_count, reverse=True)
        sampled.extend(remaining[:n - len(sampled)])

    random.shuffle(sampled)
    return sampled[:n]


def split():
    random.seed(RANDOM_SEED)
    pairs = load_pairs(PAIRS_IN)

    if len(pairs) <= HOLDOUT_SIZE:
        raise ValueError(f"Only {len(pairs)} pairs — not enough to hold out {HOLDOUT_SIZE}")

    holdout = length_stratified_sample(pairs, HOLDOUT_SIZE)
    holdout_ids = {(p["thread_id"], p["input"][:50]) for p in holdout}
    train = [p for p in pairs if (p["thread_id"], p["input"][:50]) not in holdout_ids]

    save_jsonl(holdout, HOLDOUT_OUT)
    save_jsonl(train, TRAIN_OUT)

    buckets_out = defaultdict(int)
    for p in holdout:
        wc = word_count(p)
        for lo, hi, _ in LENGTH_BUCKETS:
            if lo <= wc < hi:
                buckets_out[f"{lo}-{hi}w"] += 1
                break

    print(f"Total pairs:   {len(pairs)}")
    print(f"Holdout:       {len(holdout)}  (by length: {dict(buckets_out)})")
    print(f"Training set:  {len(train)}")
    print(f"\nHoldout → {HOLDOUT_OUT}")
    print(f"Train   → {TRAIN_OUT}")


if __name__ == "__main__":
    split()
