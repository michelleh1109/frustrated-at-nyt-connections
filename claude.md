# connections-rl

Train a small open-weight LLM to solve NYT Connections puzzles using RL with graded reward shaping, deployed on Modal Sandboxes.

## Goal

"I took Qwen 3 8B from X% to Y% on the lechmazur 940-puzzle benchmark using RL with graded reward shaping on Modal Sandboxes." — plus a blog post Modal's team would want to publish.

## Why this is interesting

The NYT Connections benchmark (arXiv: NYT-Connections paper) shows GPT-4 falls ~30% below human performance, and CoT / Self-Consistency show *diminishing returns at higher difficulty*. The puzzle is designed to punish System 1 (intuitive, associative) thinking — the hard part is reasoning about 16 words *in aggregate*, since any word can plausibly belong to multiple groups. Standard prompting hits a ceiling. RL with game-level graded rewards directly optimizes for the thing that's hard.

## Task

Given 16 words, identify 4 groups of 4 that share a category. Difficulty: yellow (easiest) → green → blue → purple (hardest). The puzzle actively deceives — words that look obviously related often belong to different groups.

## Datasets

- **Eyefyre/NYT-Connections-Answers** (HF Hub) — training + internal eval
- **lechmazur/nyt-connections** (HF Hub) — 940-puzzle benchmark, eval only, never trained on

## Base model

**Qwen/Qwen3-8B-Instruct** — strong lateral reasoning, thinking mode toggle (`enable_thinking=True` during RL rollouts for chain-of-thought, `False` at eval for speed), GRPO-compatible, Apache 2.0.

No SFT stage. The model already knows the task format; the signal we want is game-level reasoning quality, which SFT on correct answers doesn't teach.

## Pipeline

### Phase 1: Data

`src/data/load.py`

- Load both HF datasets
- Game-level holdout split (never split a puzzle): stratified by difficulty score (mean group difficulty 1–4)
- Output: `data/puzzles/train.jsonl`, `data/puzzles/holdout.jsonl`, `data/puzzles/lechmazur.jsonl`

### Phase 2: RL training

`src/training/rl_modal.py`

- GRPO via TRL
- Runs on Modal (GPU: A100)
- Each rollout: sample puzzle → model generates JSON solution → Modal Sandbox executes `reward.py` → GRPO update
- Weights saved to Modal Volume at `/weights/rl-v1/`

### Phase 3: Eval

`src/eval/run_eval.py`

- Benchmark against lechmazur 940-puzzle set
- Reports: % puzzles fully solved, % groups correct, mean reward, breakdown by difficulty bucket

## Reward function (eval/reward.py)

Runs inside Modal Sandboxes during training AND in run_eval.py during benchmarking.

| Result | Reward |
| --- | --- |
| Group correct (4/4 words) | +1.0 |
| One away (3/4 words) | +0.3 |
| Wrong (<3/4) | 0.0 |
| No groups correct (budget exhausted) | −0.5 |

Max reward per puzzle: 4.0

## Inference

`src/inference.py` — Modal vLLM endpoint, L40S GPU, LoRA adapter from Volume.
Temperature 0.0 at eval (greedy). `enable_thinking=False` for fast inference.

## File structure

```text
src/
├── common.py               # Modal app, volumes, constants, system prompt
├── data/
│   ├── load.py             # HF import + holdout split
│   └── puzzles/            # gitignored
├── training/
│   └── rl_modal.py         # GRPO + Modal Sandboxes
├── eval/
│   ├── reward.py           # Graded reward (shared: training + eval)
│   └── run_eval.py         # lechmazur benchmark
└── inference.py            # Modal vLLM endpoint
```

## Key decisions

- **No SFT** — model knows the task format; reward shaping is the signal
- **No Tinker** — TRL GRPO directly, no managed API with vLLM workarounds
- **Game-level holdout** — puzzles are never split; stratified by difficulty score
- **lechmazur as the benchmark** — 940 puzzles, consistent with published numbers
- **Modal Sandboxes for rewards** — isolated, parallel, reproducible; the demo artifact
- **Thinking mode during RL, off at inference** — Qwen 3 chain-of-thought helps during rollouts, not needed at serve time

## Archive

The original voice-lora (email LoRA) project lives at `archive/voice-lora` branch on GitHub.
Repo: [michelleh1109/ghost-drafter](https://github.com/michelleh1109/ghost-drafter)
