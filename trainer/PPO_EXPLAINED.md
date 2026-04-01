# PPO Training Explained — `train_ppo.py`

> A complete guide aligned to the actual code, written for curious high-school students and anyone learning RLHF from scratch.  
> Every formula here has a matching line of code. Every line of code here has a matching formula.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The 5 Models — Who's Who](#2-the-5-models--whos-who)
3. [The Training Loop — Step by Step](#3-the-training-loop--step-by-step)
4. [Math ↔ Code Alignment](#4-math--code-alignment)
   - [4.1 Rollout: Actor Generates a Response](#41-rollout-actor-generates-a-response)
   - [4.2 Reward Signal](#42-reward-signal)
   - [4.3 Advantage Estimation](#43-advantage-estimation)
   - [4.4 Log-Probability of the Response](#44-log-probability-of-the-response)
   - [4.5 Probability Ratio](#45-probability-ratio)
   - [4.6 Clipped Surrogate Objective — Actor Loss](#46-clipped-surrogate-objective--actor-loss)
   - [4.7 Value Loss — Critic Loss](#47-value-loss--critic-loss)
   - [4.8 KL Penalties](#48-kl-penalties)
   - [4.9 Total Loss](#49-total-loss)
5. [CriticModel Architecture](#5-criticmodel-architecture)
6. [Reward Composition — Format + Model Score](#6-reward-composition--format--model-score)
7. [The School Analogy — Why 5 Models?](#7-the-school-analogy--why-5-models)
8. [Hyperparameters Cheat Sheet](#8-hyperparameters-cheat-sheet)

---

## 1. The Big Picture

**RLHF (Reinforcement Learning from Human Feedback)** is how you teach a language model to give *better* answers, not just grammatically correct ones.  
**PPO (Proximal Policy Optimization)** is the specific algorithm used inside RLHF to update the model safely — it clips updates so the model never changes too drastically in a single step.

The loop looks like this:

```text
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                             PPO Training Loop                                    │
 │                                                                                  │
 │  Prompt ──▶ Actor Model ──▶ response text ──▶ Reward Model ──▶ reward r ──┐     │
 │                │                                                            │    │
 │                ├──▶ actor log-probs ─────────────────────────────────────▶ │    │
 │                                                                             ▼    │
 │  Old Actor ──▶ old log-probs ──────────────────────────────────────▶ PPO Update │
 │  Reference ──▶ ref log-probs ──────────────────────────────────────▶ (backprop) │
 │  Critic ────▶ value V ─────────────────────────────────────────────▶            │
 │                                                                          │       │
 │                       Actor  (new weights) ◀─────────────────────────────┤       │
 │                       Critic (new weights) ◀─────────────────────────────┘       │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

In plain English:
1. The **Actor** reads a prompt and writes an answer.
2. The **Reward Model** reads the answer and gives it a score (like a teacher grading a test).
3. PPO uses that score — plus several other models — to nudge the Actor toward writing better answers, without changing so much that it forgets everything it learned before.

---

## 2. The 5 Models — Who's Who

This is the most confusing part. There are **5 models** alive at the same time. Here is a clear breakdown:

| Model | Variable in code | Trainable? | Frozen how? | Role |
|---|---|---|---|---|
| **Actor** | `actor_model` | **YES** | — | The LLM being improved. Gets gradient updates every step. |
| **Old Actor** | `old_actor_model` | NO | `requires_grad_(False)` | A recent snapshot of the Actor. Copied from Actor every `update_old_actor_freq` steps. Used to measure how much the Actor has changed. |
| **Reference** | `ref_model` | NO | `requires_grad_(False)` | The original SFT-trained model, frozen forever. Acts as an anchor so the Actor can't drift too far from sensible language. |
| **Critic** | `critic_model` | **YES** | — | Estimates how "good" a state is. Outputs a single number V(s) per sequence. Gets gradient updates every step. |
| **Reward** | `reward_model` | NO | `requires_grad_(False)` | External pre-trained model (InternLM2-Reward). Gives the real quality score r. Never updated. |

All five models receive **the same token sequence** (`gen_out = [prompt + response]`) but produce very different outputs:

```text
                        gen_out [B, P+R]  (token ids)
                                 │
      ┌──────────────────────────┼──────────────────────┬─────────────────┬──────────────────┐
      ▼                          ▼                      ▼                 ▼                  ▼
 Actor Model             Old Actor Model          Ref Model          Critic Model       Reward Model
      │                          │                      │                 │              (text input)
      ▼                          ▼                      ▼                 ▼                  ▼
 logits [B,P+R,V]        logits [B,P+R,V]       logits [B,P+R,V]  values [B,P+R]      reward score
 → actor_logp [B]        → old_logp [B]         → ref_logp [B]    → V scalar [B]       r [B]
```

**Initialization** (from code lines 311–333):

```python
# Actor — loaded from SFT checkpoint, will be trained
actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)

# Old Actor — same weights as Actor at start, frozen
old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
old_actor_model = old_actor_model.eval().requires_grad_(False)

# Reference — same weights as Actor at start, frozen forever
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)

# Critic — LM backbone + scalar head, initialized from SFT weights
critic_model = CriticModel(lm_config)
critic_model.load_state_dict(state_dict, strict=False)

# Reward — external model, never updated
reward_model = AutoModel.from_pretrained(args.reward_model_path, ...)
reward_model = reward_model.eval().requires_grad_(False)
```

---

## 3. The Training Loop — Step by Step

Each batch goes through these steps inside `ppo_train_epoch`:

```text
 STEP   WHO                     WHAT HAPPENS                                   SHAPE
 ───────────────────────────────────────────────────────────────────────────────────────
  A     DataLoader → Actor       prompt tokens                                 [B, P]

  B     Actor (no_grad)          generate() → gen_out                          [B, P+R]

  C     Actor → Reward Model     decoded response text
        Reward Model → Actor     rewards  (line 138)                           [B]

  D     Actor → Critic           gen_out                                       [B, P+R]
        Critic → Actor           values  (lines 141–143)                       [B]

  E     Actor (internal)         advantages = rewards − values.detach()        [B]
                                 (line 144)

  F     Actor forward            actor_logp  (lines 151–156)                   [B]
        Old Actor (no_grad)      old_logp    (lines 158–161)                   [B]
        Ref Model (no_grad)      ref_logp    (lines 163–165)                   [B]

  G     Actor (internal)         ratio = exp(actor_logp − old_logp)            [B]
                                 policy_loss, value_loss, kl_ref  (170–174)

  H     loss.backward()
        Optimizer → Actor        actor_optimizer.step()   (line 180)
        Optimizer → Critic       critic_optimizer.step()  (line 181)

  I     (every N steps)          old_actor.load_state_dict(actor)  (lines 222–227)
 ───────────────────────────────────────────────────────────────────────────────────────
```

---

## 4. Math ↔ Code Alignment

### 4.1 Rollout: Actor Generates a Response

The Actor takes the prompt tokens and **samples** a response autoregressively. This happens **outside of gradient tracking** (`torch.no_grad`) because we are only collecting experience here — not computing a loss yet.

**Code (lines 129–135):**

```python
with torch.no_grad():
    model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
    gen_out = model_for_gen.generate(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask,
        max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
```

Result shape: `gen_out` is `[B, P+R]` where B = batch size, P = prompt length, R = response length.

---

### 4.2 Reward Signal

The reward model gives a quality score $r_{model} \in [-3, 3]$. For reasoning models an additional **format reward** $r_{format}$ is added for correctly using `<think>` and `<answer>` tags.

$$r = r_{\text{format}} + r_{\text{model}}$$

**Code (line 138):**

```python
rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]
```

`rewards` is a 1-D tensor with one score per sequence in the batch.

---

### 4.3 Advantage Estimation

**What is advantage?** It answers: *"Was this response better or worse than what we expected?"*

$$A = r - V(s)$$

- $r$ = actual reward (from Reward Model)
- $V(s)$ = predicted reward (from Critic Model)
- $A > 0$ → the response was *better* than expected → encourage it
- $A < 0$ → the response was *worse* than expected → discourage it

**Code (lines 141–144):**

```python
values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)   # [B, P+R]
last_indices = (full_mask * torch.arange(full_mask.size(1), ...)).argmax(dim=1)
values = values_seq[torch.arange(...), last_indices]                     # [B]
advantages = rewards - values.detach()                                   # [B]
```

> **Why `.detach()`?**  
> We use `values.detach()` here so that the advantage $A$ does not carry any gradient from the Critic into the Actor's loss. The Actor only learns from the *magnitude* of the advantage, not from the Critic's internals. The Critic gets its own separate gradient through `value_loss`.

---

### 4.4 Log-Probability of the Response

For a response of $T$ tokens, the log-probability under the model $\pi_\theta$ is:

$$\log \pi_\theta(\text{response} \mid \text{prompt}) = \sum_{t=P}^{P+T-1} \log \pi_\theta(a_t \mid a_{<t})$$

We only sum over the **response** tokens, not the prompt tokens (masked out). Padding tokens are also excluded.

**Code (lines 151–156):**

```python
labels = gen_out[:, 1:].clone()                                          # [B, P+R-1]  (next-token targets)
logp_tokens = F.log_softmax(logits[:, :-1], dim=-1)
             .gather(2, labels.unsqueeze(-1)).squeeze(-1)                # [B, P+R-1]
seq_len = gen_out.size(1) - 1
resp_mask = torch.arange(seq_len, ...).unsqueeze(0) >= prompt_length - 1  # mask: only response positions
final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))           # [B, P+R-1]
actor_logp = (logp_tokens * final_mask).sum(dim=1)                      # [B]
```

The same operation is repeated for `old_actor_model` (lines 158–161) and `ref_model` (lines 163–165), both inside `torch.no_grad()`.

---

### 4.5 Probability Ratio

The core idea of PPO: compare **how likely** the Actor *now* is to give this response vs. how likely it *was* before the update.

$$\rho = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$$

- $\rho = 1.0$ → Actor hasn't changed
- $\rho > 1$ → Actor now assigns *more* probability to this response
- $\rho < 1$ → Actor now assigns *less* probability to this response

**Code (line 169):**

```python
ratio = torch.exp(actor_logp - old_logp)   # [B]
```

Using log-space subtraction and then `exp` is numerically more stable than dividing raw probabilities (which can be astronomically small numbers).

---

### 4.6 Clipped Surrogate Objective — Actor Loss

This is the heart of PPO. The naive idea would be:

$$L = -\rho \cdot A$$

But if $\rho$ becomes very large (Actor changed a lot), the gradient explodes. PPO **clips** $\rho$ to stay in $[1-\varepsilon,\; 1+\varepsilon]$:

$$L^{\text{CLIP}} = -\mathbb{E}\left[\min\left(\underbrace{\rho \cdot A}_{\text{surr1}},\quad \underbrace{\text{clip}(\rho,\, 1-\varepsilon,\, 1+\varepsilon) \cdot A}_{\text{surr2}}\right)\right]$$

**Visualizing why clipping helps:**

```text
 Advantage A > 0  (good response — we want MORE of it)

   surr1 = ρ·A  grows with ρ  →  could become huge if ρ >> 1
   surr2 = clip(ρ)·A  is capped at (1+ε)·A

   min(surr1, surr2) = surr2 when ρ > 1+ε
   → gradient is ZERO when ρ is too large  (stop over-optimizing)

 Advantage A < 0  (bad response — we want LESS of it)

   surr1 = ρ·A  becomes very negative if ρ >> 1
   surr2 = clip(ρ)·A  is capped at (1+ε)·A  (less negative)

   min(surr1, surr2) = surr1 when ρ > 1+ε
   → still penalizes, but ratio doesn't run away
```

**Code (lines 170–172):**

```python
surr1 = ratio * advantages                                                      # [B]
surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
policy_loss = -torch.min(surr1, surr2).mean()                                   # scalar
```

The negative sign converts from "maximize reward" to "minimize loss" (standard for PyTorch).

---

### 4.7 Value Loss — Critic Loss

The Critic learns to predict the reward more accurately by minimizing the squared error between its prediction $V(s)$ and the actual reward $r$:

$$L^{\text{VF}} = \mathbb{E}[(V(s) - r)^2]$$

**Code (line 173):**

```python
value_loss = F.mse_loss(values, rewards)   # scalar
```

This is a simple regression loss — the Critic is just trying to get better at guessing the score.

---

### 4.8 KL Penalties

KL divergence measures how different two probability distributions are. Here we compute two KL terms:

**KL from Old Actor** (diagnostic only):

$$\text{KL}_{\text{old}} = \log \pi_\theta - \log \pi_{\theta_{\text{old}}}$$

**KL from Reference model** (enters the loss):

$$\text{KL}_{\text{ref}} = \log \pi_\theta - \log \pi_{\text{ref}}$$

`kl_ref` punishes the Actor for drifting too far from the original SFT model. This prevents "reward hacking" — where the model finds weird tricks to get high scores while producing incoherent text.

**Code (lines 167–168):**

```python
kl     = (actor_logp - old_logp).mean()   # scalar  — logged only, not in loss
kl_ref = (actor_logp - ref_logp).mean()   # scalar  — ENTERS the loss
```

> **Key difference:** `kl` (vs old actor) is logged to help you monitor training stability. `kl_ref` (vs reference) is the actual regularizer in the loss function. If `kl_ref` grows large, the Actor is wandering far from well-behaved language.

---

### 4.9 Total Loss

All terms are combined into a single scalar that is backpropagated:

$$L = \underbrace{L^{\text{CLIP}}}_{\text{actor loss}} + c_1 \underbrace{L^{\text{VF}}}_{\text{critic loss}} + c_2 \underbrace{L^{\text{KL}}_{\text{ref}}}_{\text{KL penalty}} + \underbrace{L^{\text{aux}}}_{\text{MoE aux loss}}$$

| Term | Coefficient | Default value | Purpose |
|---|---|---|---|
| $L^{\text{CLIP}}$ | 1.0 | — | Improve response quality |
| $L^{\text{VF}}$ | `vf_coef` | 0.5 | Keep Critic accurate |
| $L^{\text{KL}}_{\text{ref}}$ | `kl_coef` | 0.02 | Stay close to SFT model |
| $L^{\text{aux}}$ | 1.0 | 0 if no MoE | Mixture-of-Experts load balance |

**Code (line 174):**

```python
loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps
```

The division by `accumulation_steps` scales the loss correctly when gradients are accumulated over multiple mini-batches before an optimizer step.

**Optimizer step (lines 177–185):**

```python
if (step + 1) % args.accumulation_steps == 0:
    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
    actor_optimizer.step()
    critic_optimizer.step()
    actor_scheduler.step()
    critic_scheduler.step()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
```

Both Actor and Critic are updated from the *same* `loss.backward()` call because `values` (used in `value_loss`) and `actor_logp` (used in `policy_loss`) were both computed from their respective models in the same forward pass.

---

## 5. CriticModel Architecture

The Critic is a language model with its prediction head **replaced**. Instead of predicting the next token (vocabulary-sized output), it predicts a single scalar — the *value* of the current state.

```text
 Token IDs    Token          Transformer    RMS Norm    Last non-pad     value_head        V(s)
  [B, P+R] ──▶ Embeddings ──▶   Layers   ──▶         ──▶ token [B,H] ──▶ Linear(H→1) ──▶  [B]
                                                                                           scalar per
                                                                                            sequence
```

**Code (lines 29–41):**

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)   # replaces lm_head

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])          # [B, P+R, H]
        values = self.value_head(hidden_states).squeeze(-1)  # [B, P+R]
        return values
```

Then in the training loop, only the value at the **last non-padding token** is used (lines 142–143):

```python
last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
values = values_seq[torch.arange(values_seq.size(0), ...), last_indices]   # [B]
```

This gives one number per sequence — the Critic's estimate of "how good was this entire conversation?"

---

## 6. Reward Composition — Format + Model Score

The total reward combines two sources:

$$r = r_{\text{format}} + r_{\text{model}}$$

### Format Reward (reasoning mode only)

Rewards the model for using the correct `<think>...</think><answer>...</answer>` structure:

| Condition | Reward |
|---|---|
| Perfect format match | +0.5 |
| Has `<think>` (×1) | +0.25 |
| Has `</think>` (×1) | +0.25 |
| Has `<answer>` (×1) | +0.25 |
| Has `</answer>` (×1) | +0.25 |
| Max format reward total | **+1.5** |

### Model Score

The InternLM2-Reward model scores the full response, clipped to `[-3.0, +3.0]`.

In reasoning mode, a blended score is used:

$$r_{\text{model}} = 0.4 \cdot \text{score}(\text{full response}) + 0.6 \cdot \text{score}(\text{answer only})$$

This focuses the reward signal on the *answer quality*, not just verbosity.

**Code (lines 81–114):**

```python
rewards = torch.zeros(len(responses), device=args.device)   # start at 0

if args.reasoning == 1:
    rewards = reasoning_model_reward(rewards)               # add format reward

# reward model score, clipped to [-3, 3]
score = max(min(score, 3.0), -3.0)

# reasoning blend
if args.reasoning == 1:
    score = score * 0.4 + answer_score * 0.6

rewards += reward_model_scores                              # add model score
```

---

## 7. The School Analogy — Why 5 Models?

Imagine a student (the Actor) learning to write better essays:

| PPO Component | School Analogy |
|---|---|
| **Actor** | The student writing the essay right now |
| **Old Actor** | A photocopy of the student's essay from last week — used to check "how much have I changed?" |
| **Reference Model** | The original textbook — makes sure the student doesn't go off on wild tangents and forget English grammar |
| **Critic** | The student's internal gut feeling: "I think I'll get about a B on this" — before the teacher grades it |
| **Reward Model** | The actual teacher grading the essay |

**Why do we need the Old Actor if we have the Reference?**

- The **Reference** is fixed from the very beginning (SFT model). It prevents the student from forgetting how to write English altogether.
- The **Old Actor** is a recent snapshot updated every few steps. It's used to compute the *ratio* $\rho$ — how much has the student changed *in the last few practice sessions*? This is what PPO clips to ensure stable updates.

**Why do we need the Critic if we have the Reward Model?**

- The **Reward Model** only tells you the score *after* you've written the whole essay. You can't look at the essay mid-sentence and ask for a score.
- The **Critic** learns to *predict* the score in advance. The advantage $A = r - V(s)$ then tells the student: "You expected a B, but got an A — the ending was better than you thought, so do more of that."

---

## 8. Hyperparameters Cheat Sheet

| Parameter | Default | What it controls |
|---|---|---|
| `clip_epsilon` | 0.1 | The PPO clipping range $[1-\varepsilon, 1+\varepsilon]$. Smaller = more conservative updates. Typical range: 0.1–0.2. |
| `vf_coef` | 0.5 | Weight of the Critic (value) loss in the total loss. Higher = Critic trained more aggressively. |
| `kl_coef` | 0.02 | Weight of the KL penalty vs. Reference model. Higher = Actor stays closer to SFT model. |
| `update_old_actor_freq` | 4 | How many steps between syncing `old_actor ← actor`. Lower = ratio stays close to 1 (safer). Higher = more reuse of old experience. |
| `learning_rate` | 8e-8 | Actor learning rate. Very small because the Actor is already well-trained; we only fine-tune. |
| `critic_learning_rate` | 8e-8 | Critic learning rate. Often the same as Actor. |
| `grad_clip` | 1.0 | Maximum gradient norm — prevents exploding gradients. |
| `accumulation_steps` | 1 | Number of mini-batches whose gradients are accumulated before one optimizer step. Simulates a larger batch size. |
| `reasoning` | 1 | 0 = normal chat mode (no format reward). 1 = reasoning mode (format reward for `<think>/<answer>` tags). |
| `max_seq_len` | 66 | Maximum prompt length in tokens. |
| `max_gen_len` | 1536 | Maximum response length in tokens. |

---

## Quick Reference: Tensor Shapes

| Variable | Shape | Meaning |
|---|---|---|
| `enc.input_ids` | `[B, P]` | Tokenized prompts |
| `gen_out` | `[B, P+R]` | Prompt + generated response |
| `rewards` | `[B]` | One reward score per sequence |
| `values` | `[B]` | Critic's value estimate per sequence |
| `advantages` | `[B]` | rewards − values |
| `actor_logp` | `[B]` | Sum of log-probs over response tokens |
| `old_logp` | `[B]` | Same but from old actor |
| `ref_logp` | `[B]` | Same but from reference model |
| `ratio` | `[B]` | $\exp(\text{actor\_logp} - \text{old\_logp})$ |
| `policy_loss` | scalar | Clipped surrogate loss |
| `value_loss` | scalar | MSE between values and rewards |
| `kl_ref` | scalar | Mean KL from reference model |
| `loss` | scalar | Total loss = everything combined |

*B = batch size, P = prompt length, R = response length, H = hidden size, V = vocabulary size*

---

## Full Data Flow Diagram

```text
 ══════════════════════════════ SETUP (run once) ═════════════════════════════════

 SFT Checkpoint ──▶ actor_model        (trainable)
                ──▶ old_actor_model     (frozen, synced from actor every N steps)
                ──▶ ref_model           (frozen forever)
                ──▶ critic_model        (trainable, lm_head replaced by value_head)
 External Model ──▶ reward_model        (frozen)

 ══════════════════════════════ PER-STEP LOOP ════════════════════════════════════

 Prompt  ──▶  Tokenize  →  enc [B, P]
                │
                ▼
 Actor.generate  (no_grad)  →  gen_out [B, P+R]
                │
       ┌────────┴───────────────────────────────────────────────────────┐
       │                                                                │
       ▼                                                                ▼
 Decode responses (text list)                         Critic forward  →  values [B]
       │                                                                │
       ▼                                                                │
 calculate_rewards  →  rewards [B]  ────────────────────────────────── ┤
                                                                        ▼
                                             advantages = rewards − values.detach()  [B]
       │
       ├──▶ Actor forward   (WITH grad)   →  actor_logp [B]
       ├──▶ Old Actor fwd   (no_grad)     →  old_logp   [B]
       └──▶ Ref Model fwd   (no_grad)     →  ref_logp   [B]
                │
                ▼
 ratio    = exp(actor_logp − old_logp)                          [B]
 kl_ref   = (actor_logp − ref_logp).mean()                      scalar
 surr1    = ratio × advantages                                   [B]
 surr2    = clamp(ratio, 1−ε, 1+ε) × advantages                 [B]
 policy_loss = −min(surr1, surr2).mean()                         scalar
 value_loss  = MSE(values, rewards)                              scalar
                │
                ▼
 Total Loss = policy_loss + vf_coef·value_loss + kl_coef·kl_ref
                │
                ▼
 loss.backward()
 actor_optimizer.step()   critic_optimizer.step()
                │
                └──▶ every N steps:  old_actor.load_state_dict(actor)
```
