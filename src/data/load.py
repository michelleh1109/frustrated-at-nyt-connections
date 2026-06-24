"""
Load NYT Connections puzzles from HuggingFace and produce a game-level train/holdout split.

Datasets:
  - Eyefyre/NYT-Connections-Answers  — training + internal eval
  - lechmazur/nyt-connections         — 940-puzzle benchmark (eval only, never trained on)

Holdout strategy: full games only (never split a puzzle), stratified by difficulty score.

Output:
  data/puzzles/train.jsonl
  data/puzzles/holdout.jsonl
  data/puzzles/lechmazur.jsonl
"""
