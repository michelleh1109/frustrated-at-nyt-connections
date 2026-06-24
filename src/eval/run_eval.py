"""
Benchmark the model against the lechmazur 940-puzzle set.

Usage:
  modal run src/eval/run_eval.py --lora-version v1

For each puzzle:
  1. Build prompt from 16 words
  2. Call inference endpoint
  3. Parse JSON response
  4. Score with reward.py

Reports:
  - % puzzles fully solved (all 4 groups correct)
  - % groups correct (out of 940 * 4)
  - Mean reward
  - Breakdown by difficulty score bucket
"""
