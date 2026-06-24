"""
RL training loop: GRPO via TRL, reward computed inside Modal Sandboxes.

Each rollout:
  1. Sample a puzzle from train.jsonl
  2. Model generates a solution (JSON: 4 groups with category labels)
  3. A Modal Sandbox executes eval/reward.py to score the solution
  4. GRPO update on the reward signal

Rewards (defined in eval/reward.py):
  - Correct group:    +1.0
  - One away:         +0.3  (3/4 words correct)
  - Wrong:             0.0
  - Exhausting budget: -0.5 (used all 4 wrong guesses)

Weights saved to Modal Volume at /weights/rl-{VERSION}.
"""
