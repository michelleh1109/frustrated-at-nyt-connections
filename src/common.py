from pathlib import Path

import modal

app = modal.App("connections-rl")

BASE_MODEL = "Qwen/Qwen3-8B-Instruct"
RL_VERSION = "v1"
LORA_PATH = f"/weights/rl-{RL_VERSION}"

vol = modal.Volume.from_name("connections-weights", create_if_missing=True)
model_cache = modal.Volume.from_name("connections-model-cache", create_if_missing=True)

SYSTEM_PROMPT = """You are solving NYT Connections puzzles.

You are given 16 words. Your task is to find exactly 4 groups of 4 words that share a common category.

Rules:
- Every word belongs to exactly one group.
- Categories are specific — "things that can precede X" or "types of Y", not vague associations.
- The puzzle is designed to deceive: words that seem obviously related often belong to different groups.
- Commit to your groupings. Do not hedge.

Output valid JSON only, with this exact structure:
{
  "groups": [
    {"category": "CATEGORY LABEL", "words": ["WORD1", "WORD2", "WORD3", "WORD4"]},
    {"category": "CATEGORY LABEL", "words": ["WORD1", "WORD2", "WORD3", "WORD4"]},
    {"category": "CATEGORY LABEL", "words": ["WORD1", "WORD2", "WORD3", "WORD4"]},
    {"category": "CATEGORY LABEL", "words": ["WORD1", "WORD2", "WORD3", "WORD4"]}
  ]
}"""


@app.local_entrypoint()
def upload_weights(local_path: str, remote_path: str):
    with vol.batch_upload() as upload:
        upload.put_directory(local_path, remote_path)
