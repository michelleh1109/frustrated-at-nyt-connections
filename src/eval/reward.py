"""
Graded reward function for a Connections solution.

This file runs both:
  - Inside Modal Sandboxes during RL training (called as a subprocess)
  - Directly during eval (run_eval.py imports it)

Input:  solution dict  {"groups": [{"category": str, "words": [str, str, str, str]}, ...]}
        ground_truth   same schema

Scoring per group (order-independent word matching):
  4/4 correct  → +1.0  (correct group)
  3/4 correct  → +0.3  (one away)
  <3/4 correct →  0.0

Aggregate reward = sum of per-group scores.
Budget penalty  = -0.5 if no groups correct (all 4 wrong guesses exhausted).
Max reward      =  4.0 (all groups correct on first try).
"""
