"""
SFT fine-tune on email pairs via Tinker cookbook.

  python training/sft.py

1. Converts data/cleaned/train.jsonl → Tinker's conversations format
2. Runs LoRA SFT on Qwen 2.5 7B Instruct via train.Config
3. Saves checkpoint — download weights and push to Modal Volume after

Requires: TINKER_API_KEY in .env
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from tinker_cookbook import model_info, weights
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

load_dotenv()

logging.getLogger("tinker_cookbook").setLevel(logging.WARNING)
logging.getLogger("tinker").setLevel(logging.WARNING)

TRAIN_IN = Path("data/cleaned/train.jsonl")
MODEL_NAME = "Qwen/Qwen3-8B"


def convert_to_conversations(src: Path, dst: Path):
    """Convert (input, output) pairs → {"messages": [...]} format Tinker expects."""
    pairs = [json.loads(line) for line in src.open() if line.strip()]
    with dst.open("w") as f:
        for p in pairs:
            record = {
                "messages": [
                    {"role": "user", "content": p["input"]},
                    {"role": "assistant", "content": p["output"]},
                ]
            }
            f.write(json.dumps(record) + "\n")
    print(f"Converted {len(pairs)} pairs → {dst}")
    return len(pairs)


async def main():
    # Write conversations to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    conversations_path = Path(tmp.name)
    tmp.close()
    n = convert_to_conversations(TRAIN_IN, conversations_path)

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=2048,
        batch_size=8,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(conversations_path),
    )

    config = train.Config(
        log_path="/tmp/tinker-sft-voice",
        model_name=MODEL_NAME,
        recipe_name="sft-voice-v1",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=2e-4,
        lora_rank=32,
        lr_schedule="cosine",
        num_epochs=3,
        save_every=50,
        eval_every=25,
    )

    print(f"Starting SFT on {n} pairs — model: {MODEL_NAME}")
    await train.main(config)
    print()
    print("Done. Checkpoint saved as 'sft-voice-v1'.")


async def save_weights(sampler_path: str, output_dir: str = "./weights/sft-v1"):
    """
    Download a Tinker checkpoint to disk, then print the Modal push command.

    Call this after training when you're happy with the results:
        asyncio.run(save_weights("tinker://run-id/sampler_weights/checkpoint-name"))

    sampler_path: the tinker:// path shown in the Tinker console for your checkpoint
    """
    print(f"Downloading weights from {sampler_path} → {output_dir}")
    adapter_dir = await asyncio.to_thread(
        weights.download,
        tinker_path=sampler_path,
        output_dir=output_dir,
    )
    print(f"Downloaded to: {adapter_dir}")


if __name__ == "__main__":
    asyncio.run(main())
