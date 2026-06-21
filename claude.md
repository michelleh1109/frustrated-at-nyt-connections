# Email Voice Agent — Project Spec

## What this is

An end-to-end applied AI project: fine-tune an open-weight LLM on my email corpus to learn my writing voice, deploy it on Modal, and connect it to Gmail as a draft-only email assistant. The project produces a publishable comparison of personalization tiers and a concrete demonstration of RL-in-Sandboxes on a real agentic use case.

## Learning goals

- **Agentic engineering stack** — orchestration loops, OAuth, mock/real client patterns, Modal inference endpoints
- **Modal Sandboxes for RL** — using Sandboxes as isolated reward executors in an RL training loop; this is Modal's strategic direction and the most differentiated thing to demonstrate
- **Eval design** — LLM-as-judge, human eval, overfitting canary
- Training mechanics (SFT, RL/GRPO) are inputs to the project, not the focus — use managed APIs where possible

## Why this architecture

- **Voice fidelity alone doesn't justify fine-tuning for email** — professional email is a convergent register, and few-shot prompting gets ~80-85% of the signal. The real value is learning **behavioral patterns** (how I follow up, decline, calibrate formality, strategically ignore questions).
- **The tiered comparison IS the project.** Every outcome is a real finding and a real X post. The fine-tune is one tier, not the assumed winner.
- **RL-in-Sandboxes is the novel contribution** — SFT teaches cadence from pairs; RL with sandbox-computed rewards teaches constraint satisfaction (anti-patterns, task completion). The delta between them is the interesting finding.

## Build sequence

### Phase 1: Gmail data pipeline ✓ (complete)

#### 1a. Gmail API pull
- OAuth2 with Gmail API (read-only scope for pull phase)
- Pull all threads from Sent folder, paginate
- Store raw threads as JSON (message ID, timestamp, sender, recipient, subject, body HTML, thread ID)

#### 1b. Cleaning pipeline
- HTML → plain text (beautifulsoup or html2text)
- Strip quoted reply chains (detect `On Mon, X wrote:`, `>` blocks, `div.gmail_quote`)
- Strip signature (detect once, regex everywhere)
- Strip automated/transactional emails (calendar accepts, shipping notifs, unsubscribe confirms — filter by sender domain + template patterns)
- Drop anything under ~30 words after cleaning

#### 1c. Pair construction
- For each surviving sent message, grab the preceding message in the thread → `(input, output)` pair
- This is the natural supervised fine-tuning format

#### 1d. Holdout split
- **Hold out 50 threads for eval BEFORE anything else** — these are sacred, model never sees them
- Length-stratified sampling (8 short / 14 medium / 18 long / 10 very-long) — oversamples longer emails so eval covers cold emails and substantive replies, not just scheduling one-liners
- Tradeoff: ~18% drain on the 250+ word bucket (55 pairs → 45 remain in training); acceptable
- Everything else → training set (~822 pairs)

**Expected yield:** 872 total pairs → 822 training, 50 holdout

### Phase 2: SFT via Tinker

Use Tinker (managed LoRA API) to get a fine-tuned model without managing training infrastructure.
Don't optimize training — get weights, move on to the interesting parts.

- Model: **Qwen 3 8B** (latest, permissive license — Tinker's own examples use it)
- At inference time, set `enable_thinking=False` to suppress Qwen 3's chain-of-thought traces — you want direct email output, not reasoning monologue
- Format data as chat SFT pairs using Tinker's `SupervisedDatasetBuilder`
- Submit job via `tinker.TrainingClient`, save checkpoint, download weights
- Upload weights to **Modal Volume** at `/weights/sft-v1/`

See "How to use Tinker → Modal" section below for step-by-step.

### Phase 3: Eval — Tier 0 vs Tier 1

Run eval on holdout before RL to measure what SFT alone adds.

**Tier 0:** vanilla Claude (no fine-tuning, no prompting)
**Tier 1:** SFT model served from Modal Volume

For each of 50 held-out threads:
1. Generate candidate reply from Tier 0 and Tier 1
2. LLM-as-judge (Claude) blind pairwise comparison on three dimensions (1-5):
   - **Voice fidelity** — given 5 reference emails, which sounds more like the same person?
   - **Task completion** — did the candidate address what was asked?
   - **Naturalness** — does this read like a real email or AI output?
3. Automatic disqualifiers: em dashes, "I'd be happy to", "certainly", "I hope this email finds you well", eager closers
4. Randomize presentation order every time
5. Aggregate into win-rates → Tier 0 vs Tier 1 comparison table

### Phase 4: RL in Modal Sandboxes

Second training stage on top of SFT. The model generates replies; a reward function runs inside a Modal Sandbox to score them.

**Why Sandboxes:** reward computation runs arbitrary code (LLM calls, regex, heuristics) in a tight loop. Sandboxing gives isolation, parallelism, and reproducibility — the Modal-native pattern for RL.

**Reward function (runs inside each Sandbox):**
- Anti-pattern penalty: detect em dashes, banned phrases, eager closers
- Task completion score: did the reply address all questions in the incoming email?
- Voice fidelity score: LLM-as-judge against 5 reference emails
- Length appropriateness: penalize replies wildly longer/shorter than training distribution

**RL algorithm:** GRPO (same as DeepSeek-R1 — Tinker supports it natively)

**Training loop:**
```python
for thread in training_batch:
    candidate = model.sample(build_prompt(thread))
    with modal.Sandbox.create() as sb:
        reward = sb.exec("python", "reward.py", candidate, thread)
    grpo_update(candidate, reward)
```

Weights saved to Modal Volume at `/weights/rl-v1/`.

### Phase 5: Eval — Tier 1 vs Tier 2

Same eval harness as Phase 3, now comparing:

**Tier 1:** SFT model
**Tier 2:** SFT + RL model

The reward function used in RL training IS the same as the eval reward — consistent signal throughout.
The delta here is the finding: what does reward shaping add on top of supervised voice learning?

### Phase 6: Modal inference endpoint

```python
@app.cls(gpu="a10g", volumes={"/weights": vol})
class VoiceModel:
    @modal.enter()
    def load(self):
        self.llm = load_merged_model("/weights/rl-v1")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, payload: dict):
        # payload: {"thread_context": str}
        return self.llm.generate(build_prompt(payload))
```

- `modal deploy` → persistent URL
- Scale-to-zero by default (email assistant is bursty)
- Cold start 15-30s for 7B; add `min_containers=1` if painful

### Phase 7: Gmail agent — draft-only

#### Architecture
- Modal Function for agent orchestration (NOT the model endpoint)
- **OAuth scopes: read + drafts ONLY — no send scope** (load-bearing safety decision)
- EmailClient interface with MockGmailClient (dev) and RealGmailClient (prod)

#### Agent flow
1. Pull recent unread threads via Gmail API
2. Build prompt: thread context + system instructions
3. Call fine-tuned inference endpoint (Phase 6)
4. Write result to Gmail Drafts folder via API
5. I open Gmail, see draft, edit, send myself

#### MockGmailClient (for development)
- Fixture inbox with fake threads
- `send()` appends to a log file instead of calling Google
- Same interface as real client — swap via dependency injection

#### Google Calendar integration (planned)
- Detect scheduling intent before generating draft
- Fetch availability via Google Calendar API → natural language ("I'm free Tues 6/23 2-4pm PT")
- Inject into system prompt so model proposes real times
- OAuth scope: `calendar.readonly` only
- Lives in `agent/orchestrator.py` as a pre-generation step

### Phase 8: Iteration
- Tier 2.5: RAG over sent corpus (embed past replies, retrieve similar ones for context)
- DPO/KTO from usage: log `(context, model_draft, my_edited_version)` preference pairs every time I edit a draft
- Retrain RL in batches as preference data accumulates
- Blog post / X thread series

## How to use Tinker → Modal

### 1. Install and authenticate
```bash
uv pip install tinker tinker-cookbook
# set TINKER_API_KEY in .env
```

### 2. Format training data
```python
from tinker_cookbook import SupervisedDatasetBuilder

builder = SupervisedDatasetBuilder()
for pair in training_pairs:  # from data/cleaned/train.jsonl
    builder.add_example(
        messages=[
            {"role": "user", "content": pair["input"]},
            {"role": "assistant", "content": pair["output"]},
        ]
    )
dataset = builder.build()
```

### 3. Run SFT job
```python
import tinker

client = tinker.TrainingClient(model="qwen2.5-7b-instruct")
client.load_dataset(dataset)
client.train()                  # blocks until done; watch loss in terminal
client.save_state("sft-v1")    # checkpoint on Tinker's side
```

### 4. Download weights
```python
weights_path = client.save_weights_and_get_sampling_client(output_dir="./weights/sft-v1")
```

### 5. Push weights to Modal Volume
```bash
modal volume create lora-weights          # one-time
modal volume put lora-weights ./weights/sft-v1 /weights/sft-v1
```

From here, Phase 4 RL training and Phase 6 inference both read from the Volume at `/weights/sft-v1`.

## Key decisions

- **Tinker for SFT** — managed LoRA API, don't optimize training internals, get weights and move on
- **Modal Sandboxes for RL rewards** — isolated execution environment for reward computation; the native Modal pattern and the most differentiating thing to demonstrate
- **No register tagging** — model learns appropriate tone implicitly from (input, output) pairs; the incoming email already signals the right context
- **Eval runs between every tier** — Tier 0 → Tier 1 (Phase 3), Tier 1 → Tier 2 (Phase 5); each delta is a real finding
- **No bonus corpora (LinkedIn, Substacks)** — skip for v1; different register from email voice, risk of format bleed
- **OAuth scopes: read + drafts only** — no send scope ever; safety comes from architecture, not runtime checks
- **Adapter versioning** — `/weights/sft-v1/`, `/weights/rl-v1/` in Volume for A/B and rollback
- **Privacy** — email corpus goes to Tinker's servers for SFT (accepted tradeoff); inference runs on Modal (cloud); local option via Ollama if needed

## Voice anti-patterns (automatic eval disqualifiers)
- Em dashes
- "I'd be happy to" / "certainly" / "I hope this email finds you well"
- Punchy-but-empty colon constructions
- Eager closers
- Perfect-fit cadence phrases

## File structure

```
voice-lora/
├── CLAUDE.md
├── data/
│   ├── pull_gmail.py            # Gmail API → raw JSON
│   ├── clean.py                 # HTML cleaning, dedup, filtering
│   ├── holdout_split.py         # length-stratified holdout split
│   ├── format_training.py       # → JSONL for Tinker
│   ├── raw/                     # gitignored
│   ├── cleaned/                 # pairs.jsonl, train.jsonl
│   └── eval_holdout/            # holdout.jsonl (sacred)
├── training/
│   ├── sft_tinker.py            # Tinker SFT job
│   └── rl_modal.py              # RL training loop with Modal Sandboxes
├── eval/
│   ├── run_eval.py              # LLM-as-judge across tiers
│   ├── reward.py                # reward function (also used in RL Sandbox)
│   ├── human_eval.py            # side-by-side blind review
│   └── results/                 # tier comparison tables
├── serving/
│   └── inference.py             # Modal inference endpoint
├── agent/
│   ├── email_client.py          # EmailClient interface + mock/real
│   ├── orchestrator.py          # thread → prompt → model → draft
│   └── gmail_drafter.py         # write to Drafts folder
└── ui/
    └── ...
```

## X content plan
1. "I tested vanilla Claude on 50 of my real email threads. Here's how it did." (Tier 0 baseline)
2. "I LoRA'd Qwen on 800 of my emails. Here's what changed." (Tier 0 → Tier 1 delta)
3. "I added RL with Modal Sandboxes as the reward executor. Here's what that actually means and what it changed." (Tier 1 → Tier 2 delta)
4. "Here's the full Gmail agent — OAuth, mock client, Modal inference, draft-only safety." (Phase 7)
5. Retrospective: "Where fine-tuning beats prompting — and where RL beats SFT."

## References
- Chakrabarty et al., "Readers Prefer Outputs of AI Trained on Copyrighted Books over Expert Human Writers" (Oct 2025) — fine-tuned models preferred 8x on style fidelity; training data volume didn't matter
- Key difference: literary voice is high-entropy, email is convergent — the delta will be smaller, which is itself the interesting finding
