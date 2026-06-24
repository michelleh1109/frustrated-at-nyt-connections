"""
Modal vLLM inference endpoint serving the fine-tuned Connections model.

Loads base model + LoRA adapter from Modal Volume.
Accepts a puzzle (16 words as a list or newline-separated string).
Returns JSON: {"groups": [{"category": str, "words": [str, str, str, str]}, ...]}
"""

import modal
import time

from common import BASE_MODEL, SYSTEM_PROMPT, LORA_PATH, RL_VERSION, app, vol, model_cache

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .pip_install("vllm>=0.8.0")
    .add_local_file("src/common.py", "/root/common.py")
)

with vllm_image.imports():
    from vllm.sampling_params import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm import LLM


@app.cls(
    image=vllm_image,
    gpu="L40S",
    scaledown_window=300,
    timeout=600,
    volumes={
        "/weights": vol,
        "/root/.cache/huggingface": model_cache,
    },
)
@modal.concurrent(max_inputs=50)
class Inference:
    @modal.enter()
    def enter(self):
        self.llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_lora_rank=32,
        )
        self.lora_request = LoRARequest(f"rl-{RL_VERSION}", 1, LORA_PATH)

    @modal.method()
    def solve(self, words: list[str], enable_thinking: bool = False) -> dict:
        puzzle_input = "Words: " + ", ".join(words)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": puzzle_input},
        ]
        t0 = time.time()
        outputs = self.llm.chat(
            messages,
            sampling_params=SamplingParams(
                temperature=0.0,  # greedy at eval time
                max_tokens=512,
            ),
            lora_request=self.lora_request,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        elapsed = time.time() - t0
        output = outputs[0].outputs[0]
        tokens = len(output.token_ids)
        print(f"tokens: {tokens}, time: {elapsed:.2f}s, throughput: {tokens/elapsed:.1f} tok/s")
        return {"raw": output.text.strip()}
