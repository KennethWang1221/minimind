# RLHF Algorithms Explained: DPO, PPO, GRPO, and SPO

> A complete ground-up tutorial for high-school students and anyone learning RLHF from scratch.
> Every math formula is derived step by step. Every formula is mapped to the exact code line.
> No steps are skipped. No terms are assumed known.

---

## Table of Contents

- [Section 0 — Foundation: Every Term, From Scratch](#section-0--foundation-every-term-from-scratch)
- [Section 1 — DPO: Direct Preference Optimization](#section-1--dpo-direct-preference-optimization)
- [Section 2 — PPO: Proximal Policy Optimization](#section-2--ppo-proximal-policy-optimization)
- [Section 3 — GRPO: Group Relative Policy Optimization](#section-3--grpo-group-relative-policy-optimization)
- [Section 4 — SPO: Self-Play Optimization](#section-4--spo-self-play-optimization-with-adaptive-baseline)
- [Section 5 — Side-by-Side Comparison](#section-5--side-by-side-comparison)

**Tensor shape legend used throughout:**  
`B` = batch size · `P` = prompt token length · `R` = response token length · `G` = num\_generations (GRPO) · `T` = any sequence length · `V` = vocabulary size · `H` = hidden size

---

## Section 0 — Foundation: Every Term, From Scratch

> **No algorithm yet.** This section defines every symbol that appears later.  
> If you already know log-probabilities, KL divergence, and gradient descent, skip to Section 1.

---

### 0.1 What Is a Token?

A language model does not read raw letters. It reads **tokens** — small chunks of text produced by a *tokenizer*.

```
"Hello, how are you?"
→ ["Hello", ",", " how", " are", " you", "?"]
→ [15043,  11,  703,  527,  499,   30]   ← integer IDs
```

Each token maps to an integer ID. The model's entire vocabulary is typically 32 000 tokens.  
Everything from this point on works with **integer ID sequences**, not text.

---

### 0.2 How a Language Model Produces a Prediction

A language model is a function: given a sequence of token IDs, it outputs a **logit** (raw score) for every possible next token.

```text
Token IDs    Transformer     RMS Norm      lm_head        Logits        Softmax
 [B, T]   ──▶   Layers   ──▶          ──▶  Linear     ──▶ [B,T,V]  ──▶  Probs
                                            (H→V)                       [B,T,V]
```

- `B` sequences in the batch, each of length `T`, each position producing `V` scores.  
- The **logit** `z_i` for token `i` is just a raw number — positive or negative, any size.  
- We need to turn those raw numbers into actual **probabilities** that sum to 1.

---

### 0.3 Softmax — Turning Logits into Probabilities

**Definition:**

$$P(i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

- `e^{z_i}` makes every value positive (exponential is always > 0).  
- Dividing by the total sum forces all probabilities to sum to 1.

**Worked example** with a 4-token vocabulary and logits `[2.0, 1.0, 0.5, -1.0]`:

| Token | Logit z | e^z | Probability |
|---|---|---|---|
| A | 2.0 | 7.39 | 7.39 / 10.87 = **0.680** |
| B | 1.0 | 2.72 | 2.72 / 10.87 = **0.250** |
| C | 0.5 | 1.65 | 1.65 / 10.87 = **0.152** |
| D | -1.0 | 0.37 | 0.37 / 10.87 = **0.034** |
| **Sum** | | **10.87** | **1.000** ✓ |

The model would most likely generate token A next.

---

### 0.4 Log-Probability — Why Use Log?

A response of 100 tokens means multiplying 100 probabilities together:

$$P(\text{response}) = 0.68 \times 0.25 \times 0.15 \times \cdots \times 0.04$$

This product becomes astronomically small — computers lose precision ("underflow").

**Solution:** use the **logarithm**, which turns products into sums:

$$\log(a \times b) = \log a + \log b$$

So instead of multiplying tiny probabilities, we **add** log-probabilities:

$$\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(a_t \mid a_1, \ldots, a_{t-1})$$

This is the **log-probability of a full response** — a sum of per-token log-probs.

#### How the code computes this (D0-B)

```text
 Logits      log_softmax      gather(labels)     × mask        sum dim=1    log π(y|x)
[B,T,V]  ──▶  log-probs    ──▶  actual next   ──▶  zero      ──▶  [B]    one number
              all tokens         token            prompt                   per sequence
             [B, T, V]          [B, T]            [B, T]
```

The `gather` operation picks, for each position, only the log-probability of the token that was **actually generated** — not all 32 000 vocabulary scores.

**Code pattern** (appears in all four algorithms):

```python
log_probs = F.log_softmax(logits, dim=-1)          # [B, T, V]
log_probs_per_token = log_probs.gather(             # [B, T]
    dim=2, index=labels.unsqueeze(2)
).squeeze(-1)
log_prob_sequence = (log_probs_per_token * mask).sum(dim=1)  # [B]
```

---

### 0.5 What Is "Training"? Gradient Descent in Plain English

A **loss function** measures how wrong the model is — a single number. Lower is better.

**Gradient** = the direction in parameter space where loss increases the fastest (like the uphill direction on a hillside).

**Training step** = move the parameters in the *opposite* direction of the gradient (go downhill):

$$\theta \leftarrow \theta - \alpha \cdot \nabla_\theta L$$

where $\alpha$ is the **learning rate** (step size).

**`requires_grad_(False)` / `.detach()`** — marks a tensor as a *constant*. No gradient is computed through it. Used to *freeze* a model (prevent it from being updated).

---

### 0.6 The RL Framework Applied to Language Models

Reinforcement Learning has three core concepts. Here is how they map to an LLM:

| RL term | LLM meaning |
|---|---|
| **State** `s` | The prompt `x` — what the user asked |
| **Action** `a` | The response `y` — what the model wrote |
| **Policy** `π_θ(y\|x)` | The LLM itself: probability of generating `y` given `x` |
| **Reward** `r(x, y)` | A quality score (from Reward Model) |

**Goal:** find model parameters `θ` that maximize the **expected reward**:

$$\max_\theta \; \mathbb{E}_{(x, y) \sim \pi_\theta} [ r(x, y) ]$$

```text
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 ▼                                                             │
 State: Prompt x  (from dataset)                               │
 │                                                             │
 ▼                                                             │
 Policy π_θ  (the LLM)                                         │
 │                                                             │
 ▼                                                             │
 Action: Response y  (sampled tokens)                          │
 │                                                             │
 ▼                                                             │
 Environment  (Reward Model scores the response)               │
 │                                                             │
 ▼                                                             │
 Reward r(x,y)  (quality score)                                │
 │                                                             │
 ▼                                                             │
 Update θ  (gradient step toward higher reward)  ──────────────┘
```

---

### 0.7 Why We Need a Reference Model

If we just maximize reward with no constraint, the model quickly finds **reward hacks** — responses that game the reward model but are meaningless:

> Repeating "excellent! great answer!" scores high on superficial metrics but is useless.

**Solution:** Add a penalty for drifting too far from the original, well-behaved SFT model:

$$\max_\theta \; \mathbb{E}[r(x, y)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})$$

The `ref_model` is a frozen copy of the **SFT model** (Supervised Fine-Tuning model — a model already trained to give helpful human-like answers). It acts as a guardrail.

- **β small** → less constraint, model can change more freely  
- **β large** → strong constraint, model stays close to SFT behavior

---

### 0.8 KL Divergence — Measuring Distance Between Distributions

**Definition:**

$$\text{KL}(P \| Q) = \sum_x P(x) \cdot \log \frac{P(x)}{Q(x)}$$

In words: for every possible outcome `x`, we compute how surprised we would be using `Q` when the truth is `P`, weighted by how likely `x` is under `P`.

**Properties:**
- Always ≥ 0
- Equals 0 **only** when `P = Q` (distributions are identical)
- Not symmetric: `KL(P‖Q) ≠ KL(Q‖P)` in general

**For log-probs** of an LLM response, the per-sequence KL is approximately:

$$\text{KL}(\pi_\theta \| \pi_\text{ref}) \approx \sum_t [\log \pi_\theta(t) - \log \pi_\text{ref}(t)]$$

**Schulman unbiased approximation** (used in GRPO and SPO code):

Let $d = \log \pi_\text{ref}(t) - \log \pi_\theta(t)$. Then:

$$\text{KL}_t \approx e^d - d - 1$$

- This is always ≥ 0 (from convexity of `exp`)  
- Equals 0 when `d = 0` (i.e., `π_θ = π_ref` at this token)  
- Derived from the Taylor expansion of `e^d ≈ 1 + d + d²/2`, so `e^d - d - 1 ≈ d²/2` for small `d`

**Code (GRPO line 144–145, SPO line 184–185):**

```python
kl_div = ref_per_token_logps - per_token_logps   # d = log π_ref - log π_θ
per_token_kl = torch.exp(kl_div) - kl_div - 1    # e^d - d - 1  ≥ 0
```

---

### 0.9 Sigmoid Function σ

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

| x | σ(x) | Meaning |
|---|---|---|
| -4 | 0.018 | very unlikely |
| -2 | 0.119 | unlikely |
| 0 | 0.500 | 50/50 |
| 2 | 0.881 | likely |
| 4 | 0.982 | very likely |

Sigmoid maps any real number to the range (0, 1), making it useful to model probabilities. Used in the DPO loss.

`F.logsigmoid(x)` = `log(σ(x))` computed with a numerically stable formula (avoids `log(0)` when `x << 0`).

---

### 0.10 The Full RLHF Training Pipeline

```text
 Step 1: Pretrain  ──▶  Step 2: SFT  ──▶  Step 3: Preference Alignment
 (next-token pred)     (imitate demos)                  │
                                         ┌──────────────┼──────────────┬──────────────┐
                                         ▼              ▼              ▼              ▼
                                       DPO            PPO            GRPO            SPO
                                     Offline         Online         Online          Online
                                     2 models        5 models       3 models      3 + tracker
                                   No generation   Generates      G responses     1 response
                                                   + Critic       per prompt      per prompt
```

The **SFT model** is the starting point for all alignment algorithms. All four algorithms improve on this model to better follow human preferences.

---

## Section 1 — DPO: Direct Preference Optimization

> **Source file:** [`trainer/train_dpo.py`](train_dpo.py)  
> **Core idea:** No online generation. No reward model at training time. Learn purely from pre-collected (chosen, rejected) response pairs.

---

### 1.1 Models Used (2 total)

| Variable | Role | Trainable? |
|---|---|---|
| `model` | Policy — the LLM being improved | **YES** |
| `ref_model` | Reference — frozen SFT copy | NO — `requires_grad_(False)` |

**Initialization (lines 175–183):**

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)   # frozen forever
```

Both start from the **same SFT checkpoint**. Only `model` gets updated.

---

### 1.2 The Data Format

Each training sample contains a **triplet**: `(x, y_w, y_l)`

- `x` = the prompt (question)  
- `y_w` = the **chosen** (better) response — human-labeled as preferred  
- `y_l` = the **rejected** (worse) response — human-labeled as not preferred

**What is the `mask`?**  
A binary tensor, 1 for response tokens and 0 for everything else. We only measure the log-probability of the *response* part — not the prompt (which both responses share) and not padding.

```
Sequence: [prompt tokens...] [response tokens...] [PAD PAD]
mask:      0  0  0  0  0  0   1  1  1  1  1  1  1   0   0
```

---

### 1.3 The Bradley-Terry Preference Model

Where does the DPO loss come from? We need a mathematical way to say "humans prefer `y_w` over `y_l`".

The **Bradley-Terry model** is a classic tool for pairwise comparisons (used in chess rankings, sports, etc.). It says:

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

where `r(x, y)` is a reward function and `σ` is the sigmoid.

**Intuition:**
- If `r(y_w) = 5` and `r(y_l) = 3`, difference = 2, `σ(2) ≈ 0.88` → 88% chance human prefers `y_w`
- If rewards are equal: `σ(0) = 0.5` → 50/50

---

### 1.4 Step-by-Step: From RL Objective to DPO Loss

DPO derives its loss from first principles. Every step is shown here.

#### Step 1 — The RL Objective

We want to maximize reward while staying close to the reference model:

$$\max_\pi \; \underbrace{\mathbb{E}_{(x,y) \sim \pi}[r(x, y)]}_{\text{maximize quality}} - \underbrace{\beta \cdot \text{KL}(\pi \| \pi_\text{ref})}_{\text{stay close to SFT}}$$

Written out explicitly (sum over all prompts and responses):

$$\max_\pi \; \sum_{x, y} \pi(y \mid x) \cdot \left[r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)}\right]$$

#### Step 2 — Solve for the Optimal Policy

Take the derivative with respect to `π(y|x)` and set it to zero (using a Lagrange multiplier to enforce `Σ_y π(y|x) = 1`):

$$r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)} - \beta = 0$$

Rearranging:

$$\log \pi^*(y \mid x) = \frac{r(x, y)}{\beta} + \log \pi_\text{ref}(y \mid x) - \log Z(x)$$

Taking the exponential:

$$\boxed{\pi^*(y \mid x) = \frac{\pi_\text{ref}(y \mid x) \cdot \exp(r(x,y)/\beta)}{Z(x)}}$$

where the **partition function** `Z(x)` is just a normalizer that ensures probabilities sum to 1:

$$Z(x) = \sum_{y'} \pi_\text{ref}(y' \mid x) \cdot \exp\!\left(\frac{r(x, y')}{\beta}\right)$$

> Think of `Z(x)` as the total "unnormalized probability mass" — every response scaled by how good it is. Dividing by `Z(x)` makes the result a valid probability distribution.

#### Step 3 — Express Reward in Terms of the Policy

From Step 2, rearrange to express `r(x,y)`:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)$$

#### Step 4 — The Z(x) Cancellation (the Key Trick)

Compute `r(y_w) - r(y_l)` using the formula above:

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w)}{\pi_\text{ref}(y_w)} + \underbrace{\beta \log Z(x)}_{\text{+Z(x) term}}
- \beta \log \frac{\pi^*(y_l)}{\pi_\text{ref}(y_l)} - \underbrace{\beta \log Z(x)}_{\text{−Z(x) term}}$$

The `+β log Z(x)` and `−β log Z(x)` terms **cancel out perfectly!** We are left with:

$$\boxed{r(x, y_w) - r(x, y_l) = \beta \left[\log \frac{\pi^*(y_w)}{\pi_\text{ref}(y_w)} - \log \frac{\pi^*(y_l)}{\pi_\text{ref}(y_l)}\right]}$$

This is the entire trick of DPO: **we never need to explicitly compute rewards or train a reward model**. The reward difference is implicitly encoded in the policy-to-reference ratio.

#### Step 5 — Substitute into Bradley-Terry

Replace `r_w - r_l` in the Bradley-Terry formula:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}\right]\right)$$

#### Step 6 — Maximum Likelihood → DPO Loss

We have a dataset of human-labeled preferences. We want to **maximize** the log-probability that our model assigns to the correct preference. Minimizing the **negative** log-probability gives the DPO loss:

$$\boxed{L_\text{DPO} = -\mathbb{E}\left[\log \sigma\!\left(\beta \cdot \left(\underbrace{\log\pi_\theta(y_w) - \log\pi_\theta(y_l)}_{\Delta_\pi} - \underbrace{\log\pi_\text{ref}(y_w) - \log\pi_\text{ref}(y_l)}_{\Delta_\text{ref}}\right)\right)\right]}$$

Compact notation:

$$L_\text{DPO} = -\mathbb{E}[\log \sigma(\beta \cdot (\Delta_\pi - \Delta_\text{ref}))]$$

**Interpretation:**
- `Δ_π > 0`: policy assigns higher probability to `y_w` than `y_l` → σ input is positive → loss is low ✓  
- `Δ_π < 0`: policy prefers the rejected response → σ input negative → loss is high, pushes correction ✗

---

### 1.5 Per-Token Averaging (Why Not Sum?)

The sum of log-probs grows with sequence length. A long boring response would get penalized simply for being long. Instead the code **averages** per token:

$$\overline{\log \pi}(y) = \frac{\sum_t \log \pi(a_t) \cdot \text{mask}_t}{\sum_t \text{mask}_t}$$

**Code (lines 36–38):**

```python
seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
ref_log_probs   = (ref_log_probs   * mask).sum(dim=1) / seq_lengths.squeeze()
policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
```

---

### 1.6 Full DPO Training Step — Code Walk-Through

```text
 INPUT: x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected
                                       │
                                       ▼
 torch.cat along batch dim:
   x    = [x_chosen;   x_rejected]  →  [2B, T]
   y    = [y_chosen;   y_rejected]  →  [2B, T]
   mask = [mask_c;     mask_r    ]  →  [2B, T]
                                       │
                                       ▼
 Policy forward:  model(x)       →  logits      [2B, T, V]   ← gradients flow here ✓
 Ref forward:     ref_model(x)   →  ref_logits  [2B, T, V]   ← frozen, no_grad
                                       │
                                       ▼
 logits_to_log_probs(logits, y)     →  policy_log_probs  [2B, T]
 logits_to_log_probs(ref_logits, y) →  ref_log_probs     [2B, T]
                                       │
                                       ▼
 Average over response tokens  (÷ mask.sum(dim=1))
   →  policy_log_probs [2B]   ref_log_probs [2B]
                                       │
                                       ▼
 Split at B:
   chosen_policy = policy[:B]    reject_policy = policy[B:]
   chosen_ref    = ref[:B]       reject_ref    = ref[B:]
                                       │
                                       ▼
 dpo_loss:
   Δ_π   = chosen_policy − reject_policy
   Δ_ref = chosen_ref    − reject_ref
   loss  = mean( −F.logsigmoid(β · (Δ_π − Δ_ref)) )
                                       │
                                       ▼
 loss.backward()  →  optimizer.step()  →  optimizer.zero_grad()
```

**Why concatenate chosen and rejected?** Running both through the model in one forward pass is twice as fast as two separate passes — the GPU processes them as a single batch.

**Why `F.logsigmoid` instead of `log(sigmoid(x))`?**  
For large negative `x`, `sigmoid(x) ≈ 0`, so `log(sigmoid(x)) → -∞` (numerical overflow). PyTorch's `F.logsigmoid` uses a stable formula:
`log σ(x) = -log(1 + e^{-x})` computed carefully to avoid overflow.

---

### 1.7 DPO Loss Behavior

```text
 DPO Loss = −log σ(margin)

 Loss
 4.5 │●
 4.0 │
 3.5 │
 3.0 │ ●
 2.5 │
 2.0 │  ●
 1.5 │
 1.0 │   ●
 0.5 │    ●  ●
 0.0 │          ●  ●  ●
     └─────────────────────── margin (Δ_π − Δ_ref)
       -4  -3  -2  -1   0  +1  +2  +3  +4
```

- **margin < 0** (policy prefers the rejected response): high loss → strong gradient correction
- **margin = 0** (policy is neutral): loss = ln(2) ≈ 0.69 → model is being pushed toward correct preference
- **margin > 0** (policy correctly prefers chosen): low loss → small gradient, model is on track

**Code mapping (`dpo_loss` function, lines 33–51):**

```python
pi_logratios  = chosen_policy_log_probs - reject_policy_log_probs  # Δ_π
ref_logratios = chosen_ref_log_probs    - reject_ref_log_probs     # Δ_ref
logits = pi_logratios - ref_logratios                              # Δ_π - Δ_ref
loss   = -F.logsigmoid(beta * logits).mean()                      # -log σ(β·…)
```

---

### 1.8 Mini Summary

- DPO requires **no reward model** and **no response generation** during training — it is purely offline.
- The key insight is that the Z(x) partition function **cancels** when computing `r(y_w) - r(y_l)`, so the reward is implicitly encoded in the log-ratio between policy and reference.
- The loss pushes the policy to assign more probability to the chosen response *relative to the reference*, and less to the rejected response *relative to the reference*.

---

## Section 2 — PPO: Proximal Policy Optimization

> **Source file:** [`trainer/train_ppo.py`](train_ppo.py)  
> **Core idea:** Online RL — the model generates responses live, a Reward Model scores them, and a Critic estimates their value. A clipped ratio keeps updates safe.  
> **Deeper reference:** See [`PPO_EXPLAINED.md`](PPO_EXPLAINED.md) for a full walkthrough.

---

### 2.1 Models Used (5 total)

| Variable | Role | Trainable? | Updated how often? |
|---|---|---|---|
| `actor_model` | The LLM being improved | **YES** | Every step |
| `old_actor_model` | Frozen snapshot of actor | NO | Copied from actor every `N` steps |
| `ref_model` | Frozen SFT copy | NO | Never |
| `critic_model` | Value estimator (scalar head) | **YES** | Every step |
| `reward_model` | External quality scorer | NO | Never |

#### Why 5 models? The motivation chain:

```text
 Problem                                        →  Solution (Model)
 ──────────────────────────────────────────────────────────────────────────────
 Need a reward signal to know                  →  Reward Model
 if responses are good                            External scorer, frozen
                                                  Gives r ∈ [−3, 3]
 ──────────────────────────────────────────────────────────────────────────────
 Reward is sparse (one score per full          →  Critic Model
 response) → high-variance gradient               Predicts V(s) per sequence
                                                  Reduces variance: A = r − V(s)
 ──────────────────────────────────────────────────────────────────────────────
 Generated data with π_old but want            →  Old Actor Model
 gradients for π_new (they may differ)            Recent snapshot of actor
                                                  Computes ratio ρ = π_new/π_old
 ──────────────────────────────────────────────────────────────────────────────
 Without constraints, model drifts into        →  Reference Model
 gibberish / reward hacking                       Frozen SFT copy
                                                  KL penalty keeps actor close
 ──────────────────────────────────────────────────────────────────────────────
```

---

### 2.2 Importance Sampling — Why the Ratio ρ Exists

**The problem:** We generate responses using `π_old`, but then we update the parameters of `π_new`. If `π_new` has already changed from `π_old`, the gradient needs a correction factor.

**Importance sampling** is the mathematical fix. For any function `f(y)`:

$$\mathbb{E}_{y \sim \pi_\text{new}}[f(y)] = \mathbb{E}_{y \sim \pi_\text{old}}\left[f(y) \cdot \frac{\pi_\text{new}(y)}{\pi_\text{old}(y)}\right]$$

The correction factor is the **probability ratio**:

$$\rho = \frac{\pi_\theta(y)}{\pi_{\theta_\text{old}}(y)} = \exp(\log \pi_\theta(y) - \log \pi_{\theta_\text{old}}(y))$$

In log space this is just a subtraction. **Code (line 169):**

```python
ratio = torch.exp(actor_logp - old_logp)   # ρ = exp(log π_new - log π_old)
```

---

### 2.3 Rollout → Reward → Value → Advantage

**Step 1: Rollout** — actor generates a response (no gradients, just sampling):

```python
gen_out = model_for_gen.generate(input_ids=enc.input_ids, ...)   # [B, P+R]
```

**Step 2: Reward** — decode tokens to text, pass to reward model:

```python
rewards = calculate_rewards(prompts, responses_text, reward_model, ...)  # [B]
```

**Step 3: Value** — critic estimates expected reward for the whole sequence:

```python
values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)   # [B, P+R]
# take the value at the last non-padding token position
values = values_seq[torch.arange(B), last_indices]                        # [B]
```

**Step 4: Advantage** = how much better or worse than expected:

$$A = r - V(s)$$

```python
advantages = rewards - values.detach()   # [B]
```

> **Why `.detach()`?** The advantage `A` is used as a fixed weighting signal for the actor's loss. We do not want gradients to flow *back through the critic* when computing the actor's loss. The critic gets its own separate gradient through `value_loss`.

---

### 2.4 Log-Probability of the Full Response

We need the log-probability that the actor (current, old, and reference) would assign to the generated response:

$$\log \pi(y \mid x) = \sum_{t=P}^{P+R-1} \log \pi(a_t \mid a_{<t})$$

Only the **response tokens** are summed (prompt tokens are masked out).

**Code (lines 151–156):**

```python
labels = gen_out[:, 1:].clone()                                        # [B, P+R-1]
logp_tokens = F.log_softmax(logits[:, :-1], dim=-1)
             .gather(2, labels.unsqueeze(-1)).squeeze(-1)              # [B, P+R-1]
resp_mask = torch.arange(seq_len).unsqueeze(0) >= prompt_length - 1   # only response
final_mask = resp_mask & ~labels.eq(tokenizer.pad_token_id)            # [B, P+R-1]
actor_logp = (logp_tokens * final_mask).sum(dim=1)                     # [B]
```

The same operation is repeated for `old_actor_model` (lines 158–161, `no_grad`) and `ref_model` (lines 163–165, `no_grad`).

---

### 2.5 The Clipped Surrogate Loss

**Without clipping**, the gradient would be `-ρ · A`. If `ρ` becomes very large (policy changed a lot), gradients explode and the model breaks.

**PPO clip** restricts `ρ` to the window `[1-ε, 1+ε]`:

$$L^\text{CLIP} = -\mathbb{E}\left[\min\left(\underbrace{\rho \cdot A}_\text{surr1},\; \underbrace{\text{clip}(\rho,\, 1-\varepsilon,\, 1+\varepsilon) \cdot A}_\text{surr2}\right)\right]$$

**Decision diagram — when does clipping activate?**

```text
         Compute ρ = exp(log π_new − log π_old)
                             │
               ┌─────────────▼─────────────┐
               │         A > 0 ?           │
               │  (good response — want    │
               │   more of this action)    │
               └──────┬────────────┬───────┘
                     YES            NO
                      │              │
                      ▼              ▼
               ρ > 1 + ε?       ρ < 1 − ε?
               ┌──┴──┐           ┌──┴──┐
              YES    NO          YES    NO
               │      │           │      │
               ▼      ▼           ▼      ▼
          (1+ε)·A   ρ·A       (1−ε)·A  ρ·A
          clipped  normal      clipped normal
          (stop    (update     (stop    (update
          over-    as-is)      over-    as-is)
          reward)              penalize)
```

**Code (lines 170–172):**

```python
surr1 = ratio * advantages                                                          # [B]
surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
policy_loss = -torch.min(surr1, surr2).mean()                                       # scalar
```

---

### 2.6 Value Loss, KL Penalty, and Total Loss

**Critic (value) loss** — train the Critic to predict rewards more accurately:

$$L^\text{VF} = \mathbb{E}[(V(s) - r)^2]$$

```python
value_loss = F.mse_loss(values, rewards)
```

**KL penalty from reference** — prevent actor from drifting from SFT model:

$$L^\text{KL} = \mathbb{E}[\log \pi_\theta - \log \pi_\text{ref}]$$

```python
kl_ref = (actor_logp - ref_logp).mean()
```

**Total loss (line 174):**

$$L = L^\text{CLIP} + c_1 \cdot L^\text{VF} + c_2 \cdot L^\text{KL\_ref} + L^\text{aux}$$

```python
loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) \
       / args.accumulation_steps
```

---

### 2.7 Full PPO Training Loop — Sequence Diagram

```text
 STEP   WHO                   WHAT HAPPENS                                 SHAPE
 ─────────────────────────────────────────────────────────────────────────────────
  1     DataLoader → Actor    prompt tokens enc                            [B, P]
        Actor (no_grad)       generate() → gen_out                        [B, P+R]
 ─────────────────────────────────────────────────────────────────────────────────
  2     Actor → Reward Model  decoded response text
        Reward Model → Actor  rewards  (line 138)                         [B]
 ─────────────────────────────────────────────────────────────────────────────────
  3     Actor → Critic        gen_out                                     [B, P+R]
        Critic → Actor        values  (lines 141–143)                     [B]
 ─────────────────────────────────────────────────────────────────────────────────
  4     Actor (internal)      advantages = rewards − values.detach()      [B]
                              (line 144)
 ─────────────────────────────────────────────────────────────────────────────────
  5     Actor forward         actor_logp  (lines 151–156)                 [B]
        Old Actor (no_grad)   old_logp    (lines 158–161)                 [B]
        Ref Model (no_grad)   ref_logp    (lines 163–165)                 [B]
 ─────────────────────────────────────────────────────────────────────────────────
  6     Actor (internal)      ratio = exp(actor_logp − old_logp)          [B]
                              policy_loss + value_loss + kl_ref  (170–174)
 ─────────────────────────────────────────────────────────────────────────────────
  7     loss.backward()
        Optimizer → Actor     actor_optimizer.step()   (line 180)
        Optimizer → Critic    critic_optimizer.step()  (line 181)
 ─────────────────────────────────────────────────────────────────────────────────
  ★  Every N steps:  old_actor.load_state_dict(actor.state_dict())  (lines 222–227)
```

---

### 2.8 Mini Summary

- PPO uses **5 models** because each solves a different problem: generating data (Actor), measuring quality (Reward), estimating expected quality (Critic), correcting the importance-sampling gap (Old Actor), and preventing language collapse (Reference).
- The **clip** on `ρ` is the defining feature of PPO: updates that would change the policy too drastically are silently zeroed out.
- Both Actor and Critic are updated from a **single `loss.backward()` call** because their losses are summed into one scalar.

---

## Section 3 — GRPO: Group Relative Policy Optimization

> **Source file:** [`trainer/train_grpo.py`](train_grpo.py)  
> **Core idea:** Generate G responses per prompt. Rank them within the group. Use the group statistics as the baseline — **no Critic neural network needed**.

---

### 3.1 Models Used (3 total)

| Variable | Role | Trainable? |
|---|---|---|
| `model` | Policy — the LLM being improved | **YES** |
| `ref_model` | Reference — frozen SFT copy | NO |
| `reward_model` | External quality scorer | NO |

No Critic. No Old Actor. GRPO eliminates both by using group-relative normalization.

---

### 3.2 The Policy Gradient Theorem — The Foundation of GRPO

All three online algorithms (PPO, GRPO, SPO) build on this theorem. We want to maximize:

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[r(y)]$$

**Theorem** (derivation every step):

$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[r(y) \cdot \nabla_\theta \log \pi_\theta(y)]$$

**Derivation:**

1. $\nabla_\theta \mathbb{E}[r] = \nabla_\theta \sum_y \pi_\theta(y)\, r(y)$
2. $= \sum_y r(y)\, \nabla_\theta \pi_\theta(y)$
3. Multiply and divide by $\pi_\theta(y)$ (the "log-derivative trick": $\nabla \pi = \pi \cdot \nabla \log \pi$):
   $= \sum_y \pi_\theta(y) \cdot r(y) \cdot \nabla_\theta \log \pi_\theta(y)$
4. $= \mathbb{E}_{y \sim \pi_\theta}[r(y) \cdot \nabla_\theta \log \pi_\theta(y)]$

**Conclusion:** To maximize expected reward, we need to **increase the log-probability** of responses weighted by their reward. This is why the loss contains `log π_θ * reward`.

---

### 3.3 Why Baseline Subtraction Is Unbiased

Replacing raw reward `r` with advantage `A = r - b` (for any baseline `b`) does not change the expected gradient:

$$\mathbb{E}[b \cdot \nabla_\theta \log \pi_\theta(y)] = b \cdot \nabla_\theta \underbrace{\mathbb{E}_{y \sim \pi_\theta}[1]}_{= 1,\; \text{constant}} = b \cdot 0 = 0$$

So subtracting a baseline **reduces variance** (makes training more stable) without **biasing the gradient** (the expected update direction is unchanged). This is the mathematical justification for GRPO replacing the Critic with a group mean.

---

### 3.4 GRPO's Baseline: Group Relative Normalization

**Step 1: Generate G responses per prompt**

```python
outputs = model_for_gen.generate(
    **prompt_inputs,
    num_return_sequences=args.num_generations,   # G responses per prompt
    ...)                           # outputs: [B*G, P+R]
```

For each of the `B` prompts in the batch, the model generates `G` different responses. This produces `B*G` total sequences.

**Step 2: Score each response**

```python
rewards = calculate_rewards(prompts, completions, reward_model, ...)  # [B*G]
```

**Step 3: Group normalization (first stage)**

```python
grouped_rewards = rewards.view(-1, args.num_generations)   # [B, G]
mean_r = grouped_rewards.mean(dim=1)                       # [B] — one mean per prompt
std_r  = grouped_rewards.std(dim=1)                        # [B] — one std per prompt
# Broadcast back from [B] to [B*G]
mean_r = mean_r.repeat_interleave(args.num_generations)    # [B*G]
std_r  = std_r.repeat_interleave(args.num_generations)     # [B*G]
advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)  # [B*G]
```

`repeat_interleave(G)` duplicates each element G times: `[a, b, c] → [a,a,a, b,b,b, c,c,c]` — so each response in a group gets its group's mean/std.

**Step 4: Batch normalization (second stage)**

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*G]
```

**Why two stages?**

```text
 ┌───────────────────────────────────────┐     ┌───────────────────────────────────────┐     ┌───────────┐
 │ Stage 1: Group Normalization          │     │ Stage 2: Batch Normalization          │     │  Stable   │
 │ Within each prompt's G responses      │ ──▶ │ Across all B*G responses              │ ──▶ │ advantage │
 │ → removes prompt-difficulty bias      │     │ → stops high-variance groups from     │     │  signal   │
 │   (all 0-centered within group)       │     │   dominating the gradient update      │     │           │
 └───────────────────────────────────────┘     └───────────────────────────────────────┘     └───────────┘
```

---

### 3.5 Worked Example — Group Normalization with G = 4

```text
 Prompt: "What is gravity?"
     │
     ├──▶ y1: "Gravity is a fundamental force..."   r = +2.1  ─┐
     ├──▶ y2: "Dunno, ask someone else."             r = +0.5  ─┤  Group stats:
     ├──▶ y3: "???"                                  r = −1.2  ─┤  mean = 0.80
     └──▶ y4: "Newton described it as..."            r = +1.8  ─┘  std  = 1.37
                                                                │
                                                                ▼
     y1: A = (2.1 − 0.80) / 1.37 = +0.95   ← above average → increase probability ✓
     y2: A = (0.5 − 0.80) / 1.37 = −0.22   ← below average → decrease probability
     y3: A = (−1.2 − 0.80) / 1.37 = −1.46  ← far below     → decrease probability ✗
     y4: A = (1.8 − 0.80) / 1.37 = +0.73   ← above average → increase probability ✓
```

y1 and y4 get positive advantages (better than average → increase their probability).  
y2 and y3 get negative advantages (worse than average → decrease their probability).

---

### 3.6 Tensor Shape Flow Through GRPO

```text
 Prompts   Tokenized   generate()    completion_ids   rewards    view(B,G)  mean(dim=1)  repeat_interleave  advantages
   [B]  ──▶  [B,P]  ──▶ [B*G,P+R] ──▶  [B*G, R]   ──▶ [B*G] ──▶ [B, G] ──▶   [B]    ──▶     [B*G]      ──▶  [B*G]
```

---

### 3.7 Completion Mask — Ignoring Padding

Responses have different lengths. Padding tokens after the EOS token must not contribute to the loss.

```text
 completion_ids:  [ tok1  tok2  EOS   PAD   PAD ]
                                │
                                ▼ is_eos vector
 is_eos:         [   0     0    1     0     0  ]
                                │
                                ▼ find first 1
 eos_idx = 2
                                │
                                ▼ build mask: 1 up to and including EOS
 completion_mask: [   1     1    1     0     0  ]
```

**Code (lines 139–142):**

```python
is_eos = completion_ids == tokenizer.eos_token_id            # [B*G, R]
eos_idx = torch.full((B*G,), R, dtype=torch.long)            # default = full length
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
completion_mask = (torch.arange(R).expand(B*G, -1) <= eos_idx.unsqueeze(1)).int()
```

---

### 3.8 The Per-Token Loss and the exp-detach Trick

**GRPO per-token loss:**

$$L_t = -\underbrace{\exp(\log\pi_\theta(t) - \text{sg}(\log\pi_\theta(t)))}_{\text{exp-detach trick}} \cdot A_i + \beta \cdot \text{KL}_t$$

where `sg(·)` means stop-gradient (`.detach()` in PyTorch).

**The exp-detach trick unpacked:**

| When | Value | Gradient |
|---|---|---|
| **Forward pass** | `exp(logp - logp.detach()) = exp(0) = 1` | (not used) |
| **Backward pass** | `d/dθ [exp(logp - c)] = exp(logp - c) · d(logp)/dθ` | At `c = logp`: `= 1 · d(logp)/dθ` |

**Net effect:** Numerically the term equals 1, but its gradient is `d(logp)/dθ` — exactly the same gradient as `-logp * A`. The difference from just writing `-logp * A` is purely stylistic: the exp form is written in **importance-sampling ratio form** which makes it straightforward to add a clipping ratio (like PPO) in the future.

**Code (line 146):**

```python
per_token_loss = -(
    torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    - args.beta * per_token_kl
)
policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
               / completion_mask.sum(dim=1)).mean()
```

The masked mean: sum over valid tokens, divide by number of valid tokens per sequence, then mean over the batch.

---

### 3.9 Full GRPO Training Step

```text
 Prompts batch [B]
       │
       ▼
 Tokenize  →  [B, P]
       │
       ▼
 model.generate(num_return_sequences=G)  no_grad
   →  outputs [B*G, P+R]
       │
       ▼
 completion_ids = outputs[:, P:]  →  [B*G, R]
       │
       ├──▶ get_per_token_logps(model,     outputs)  WITH grad  →  per_token_logps     [B*G, R]
       └──▶ get_per_token_logps(ref_model, outputs)  no_grad    →  ref_per_token_logps [B*G, R]
       │
       ▼
 decode completions → text  →  calculate_rewards  →  rewards [B*G]
       │
       ▼
 Group norm:  view(B,G) → mean + std per group → repeat_interleave(G)
              A = (r − mean) / (std + ε)      →  [B*G]
       │
       ▼
 Batch norm:  A = (A − mean(A)) / (std(A) + ε)  →  [B*G]
       │
       ▼
 Build completion_mask [B*G, R]  (1 up to EOS, 0 after padding)
       │
       ▼
 per_token_kl = exp(ref_logp − logp) − (ref_logp − logp) − 1   [B*G, R]
       │
       ▼
 per_token_loss = −( exp(logp − sg(logp)) · A  −  β · kl )
 masked mean over valid tokens  →  policy_loss  (scalar)
       │
       ▼
 loss.backward()  →  optimizer.step()  →  optimizer.zero_grad()
```

---

### 3.10 Mini Summary

- GRPO **eliminates the Critic** (and Old Actor) by using the group's own mean/std as the baseline — no separate neural network needed.
- G responses per prompt give a reliable local estimate of "what is average for this prompt" — this removes prompt-level difficulty bias.
- The **two-stage normalization** (group then batch) ensures stable gradients across diverse prompts with different reward ranges.

---

## Section 4 — SPO: Self-Play Optimization with Adaptive Baseline

> **Source file:** [`trainer/train_spo.py`](train_spo.py)  
> **Core idea:** One response per prompt. Replace GRPO's per-batch group statistics with a **Bayesian running baseline** (Beta distribution) that accumulates reward history across the entire training run.

---

### 4.1 Models Used (3 + 1 tracker)

| Variable | Role | Trainable? |
|---|---|---|
| `model` | Policy — the LLM being improved | **YES** |
| `ref_model` | Reference — frozen SFT copy | NO |
| `reward_model` | External quality scorer | NO |
| `value_tracker` | `AutoAdaptiveValueTracker` | Not a neural net — a statistical state |

---

### 4.2 Why a Running Baseline Instead of Group Stats?

| | GRPO | SPO |
|---|---|---|
| Responses per prompt | G (default: 8) | **1** |
| Baseline source | Within-batch group mean/std | Cross-batch Beta distribution |
| Problem solved | Prompt-difficulty bias | No group to normalize against |
| Memory | G × forward passes | 1 forward pass |

With only 1 response per prompt, GRPO's group `std = 0` and normalization breaks. SPO uses a **persistent statistical tracker** that learns the "expected reward" from the entire training history instead.

---

### 4.3 Beta Distribution — For High Schoolers

A Beta distribution is a probability distribution over values in `[0, 1]`. Think of it as a **Bayesian coin**:

- `α` = number of "heads" (high-reward responses) seen so far  
- `β_dist` = number of "tails" (low-reward responses) seen so far  
- **Mean** = `α / (α + β_dist)` = estimated win rate

> ⚠️ **Name collision warning:**  
> `args.beta` = KL penalty coefficient (a small float like 0.02) — used in the loss formula.  
> `self.beta` inside `AutoAdaptiveValueTracker` = Beta distribution parameter — a count.  
> These are **completely different quantities**. This README uses `β_kl` for the KL coefficient and `β_dist` for the Beta distribution parameter.

**Three example states:**

| α | β_dist | Mean baseline | Interpretation |
|---|---|---|---|
| 1 | 1 | 0.50 | No data yet — neutral |
| 10 | 2 | 0.83 | Model wins often — high baseline |
| 2 | 10 | 0.17 | Model fails often — low baseline |

---

### 4.4 Initial State: β(1, 1) = Neutral

**Code (`__init__`, lines 35–37):**

```python
N_init = 1.0 / (1.0 - self.clip_lower)   # = 1/(1-0.5) = 2
self.alpha = 0.5 * N_init                 # = 1.0
self.beta  = 0.5 * N_init                 # = 1.0
```

- `Beta(1, 1)` = Uniform distribution → baseline = `1/(1+1) = 0.5`
- Unnormalized (reward scale): `0.5 × 6 − 3 = 0.0`
- **Initial baseline = 0** → first-step advantage = raw reward

---

### 4.5 Advantage Computation

**Step 1: Get baseline from tracker**

```python
baselines = value_tracker.get_baselines(len(prompts))   # [B], values in [0, 1]
```

**Step 2: Un-normalize to reward scale** (rewards live in `[-3, 3]`)

$$b = \text{baseline} \times 6 - 3 \quad \in [-3, 3]$$

```python
unnormalized_baselines = baselines * (2 * scale) - scale   # [B]
```

**Worked example:** `baseline = 0.7 → b = 0.7 × 6 − 3 = 1.2`  
(The model is performing above average, so the baseline is 1.2 out of a max of 3)

**Step 3: Compute and clamp advantage**

$$A = r - b, \quad A \leftarrow \text{clamp}(A, -5, 5)$$

```python
advantages = rewards - unnormalized_baselines   # [B]
advantages = advantages.clamp(-5.0, 5.0)
```

---

### 4.6 Adaptive Forgetting Factor ρ

**The problem:** As the model trains, its behavior changes. Old reward statistics may no longer reflect what the current policy is capable of. The tracker needs to "forget" old data.

**How it measures policy change:** compare the mean log-probability of responses now vs. the previous batch:

$$\text{KL}_\text{approx} = |\overline{\log \pi_\text{now}} - \overline{\log \pi_\text{prev}}|$$

**Forgetting factor:**

$$\rho = 2^{-\text{KL} / D_\text{half}}, \quad \text{clipped to } [0.5,\; 0.96]$$

where `D_half = 0.06` is the "KL half-life" — the amount of KL divergence at which ρ drops to 0.5.

```text
 ρ (forgetting factor)        Forgetting Factor vs Policy KL  (D_half = 0.06)

 1.00 │●
 0.90 │ ●
 0.80 │  ●
 0.70 │   ●
 0.60 │
 0.50 │    ●────────────────  (clipped at 0.50 — always keep at least 50% of history)
 0.40 │
      └──────────────────────────────────── KL divergence
        0.00 0.01 0.02 0.03  0.06  0.12  0.18  0.24
```

| KL | ρ | Memory kept |
|---|---|---|
| 0.00 | 1.00 | 100% — policy unchanged, trust all history |
| 0.03 | 0.71 | 71% — policy moved a little |
| 0.06 (= D_half) | 0.50 | 50% — half the history is discarded |
| ≥ 0.12 | 0.50 (clipped) | at least 50% always kept |

**Code (`compute_rho`, lines 44–51):**

```python
kl = abs(self.old_mean_logprob - cur_mean_logprob)
rho = 2 ** (-kl / self.D_half)
return max(min(rho, self.clip_upper), self.clip_lower)
```

---

### 4.7 Beta Distribution Update Rule

After each batch, update the tracker with the current batch's reward:

**Step 1:** Normalize rewards from `[-3, 3]` to `[0, 1]`:

$$r_\text{norm} = \frac{r + 3}{6} \in [0, 1]$$

**Step 2:** Update `α` and `β_dist` with forgetting:

$$\alpha \leftarrow \rho \cdot \alpha + \overline{r_\text{norm}}$$

$$\beta_\text{dist} \leftarrow \rho \cdot \beta_\text{dist} + (1 - \overline{r_\text{norm}})$$

**Intuition:**
- `ρ < 1` fades old counts like radioactive decay
- High reward batch: `α` grows faster than `β_dist` → baseline rises → future advantages shrink (model is doing better)
- Low reward batch: `β_dist` grows faster → baseline falls → future advantages grow (model needs to improve more)

**Code (`update`, lines 53–66):**

```python
scale = 3.0
normalized_rewards = (rewards + scale) / (2 * scale)         # [0, 1]
avg_normalized_reward = normalized_rewards.mean().item()
self.alpha = rho * self.alpha + avg_normalized_reward
self.beta  = rho * self.beta  + (1 - avg_normalized_reward)
```

---

### 4.8 AutoAdaptiveValueTracker — State Machine

```text
 ┌──────────────────────────────────────────────────────────────────┐
 │  INITIALIZE                                                      │
 │  α = 1.0,  β_dist = 1.0                                         │
 │  baseline = α / (α + β_dist) = 0.50                             │
 │  old_mean_logprob = None                                         │
 └────────────────────────┬─────────────────────────────────────────┘
                          │ ◀──────────────────────────────────────────┐
                          ▼                                             │
 get_baselines(B)                                                      │
   return tensor filled with α/(α+β_dist)  →  [B]                    │
   (same baseline value for every sequence in the batch)               │
                          │                                             │
                          ▼                                             │
 Use in training step:                                                 │
   unorm_b = baseline × 6 − 3                                         │
   A = rewards − unorm_b  →  clamp(−5, 5)                            │
                          │                                             │
                          ▼                                             │
 compute_rho(cur_mean_logprob):                                        │
   KL = | logp_now − logp_prev |                                       │
   ρ  = 2^(−KL / D_half)  →  clip to [0.50, 0.96]                    │
                          │                                             │
                          ▼                                             │
 r_norm = (rewards + 3) / 6   ∈ [0, 1]                               │
                          │                                             │
                          ▼                                             │
 α      ← ρ · α      + mean(r_norm)                                   │
 β_dist ← ρ · β_dist + (1 − mean(r_norm))                            │
                          │                                             │
                          ▼                                             │
 old_mean_logprob = cur_mean_logprob  ─────────────────────────────────┘
 (repeat from get_baselines() on the next batch)
```

---

### 4.9 Per-Token Loss (Simpler than GRPO)

$$L_t = -\log \pi_\theta(t) \cdot A + \beta_\text{kl} \cdot \text{KL}_t$$

**Code (line 186):**

```python
per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl
policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
               / completion_mask.sum(dim=1)).mean()
```

**Why no exp-detach trick here (unlike GRPO)?**  
GRPO uses `exp(logp - sg(logp))` as an importance-sampling form in case a clipping ratio is added later. SPO always uses a **single response with the current policy** — there is no off-policy gap to correct. The direct `-logp * A` form is simpler and equivalent.

---

### 4.10 Update Timing (Critical)

The order of operations matters:

```python
loss.backward()                                                  # 1. compute gradients
rho = value_tracker.update(                                      # 2. update tracker with
    rewards,                                                     #    CURRENT policy's logprobs
    per_token_logps.detach(),                                    #    (detached — not part of graph)
    completion_mask.float())
optimizer.step()                                                 # 3. THEN update weights
optimizer.zero_grad()
```

Why this order? The tracker's `update` uses `per_token_logps` (the log-probs of the **current, pre-update policy**) to compute `cur_mean_logprob`. If `optimizer.step()` ran first, the logprobs would change and the ρ calculation would be inaccurate.

---

### 4.11 SPO vs GRPO Baseline Side-by-Side

```text
 ┌──────────────────────────────────────────┐   ┌──────────────────────────────────────────────────┐
 │  GRPO Baseline — Per Batch               │   │  SPO Baseline — Cross Batch                      │
 │                                          │   │                                                  │
 │  1. Sample G responses per prompt        │   │  1. Sample 1 response per prompt                 │
 │          ↓                               │   │          ↓                                       │
 │  2. Score all G  →  rewards [B*G]        │   │  2. Score  →  rewards [B]                        │
 │          ↓                               │   │          ↓                                       │
 │  3. Group by prompt  →  [B, G]           │   │  3. value_tracker.get_baselines()                │
 │          ↓                               │   │     α/(α+β_dist) from ALL training history       │
 │  4. mean + std per group                 │   │          ↓                                       │
 │          ↓                               │   │  4. A = r − unnorm_baseline  (clamp ±5)          │
 │  5. A = (r − mean) / std                │   │          ↓                                       │
 │     normalized within group              │   │  5. value_tracker.update(rewards, logps)         │
 │                                          │   │     α, β_dist updated for next batch             │
 └──────────────┬───────────────────────────┘   └──────────────────┬───────────────────────────────┘
                │                                                   │
                └─────────────────────┬─────────────────────────────┘
                                      ▼
                          per-token loss  →  loss.backward()
```

---

### 4.12 Full SPO Training Step

```text
 Prompts batch [B]
       │
       ▼
 Tokenize  →  [B, P]
       │
       ▼
 model.generate(num_return_sequences=1)  no_grad
   →  outputs [B, P+R]
       │
       ▼
 completion_ids = outputs[:, P:]  →  [B, R]
       │
       ├──▶ get_per_token_logps(model,     outputs)  WITH grad  →  per_token_logps     [B, R]
       └──▶ get_per_token_logps(ref_model, outputs)  no_grad    →  ref_per_token_logps [B, R]
       │
       ▼
 calculate_rewards  →  rewards [B]
       │
       ▼
 value_tracker.get_baselines(B)  →  baselines [B]  =  α / (α + β_dist)
       │
       ▼
 unnorm_b = baselines × 6 − 3   →  [B]
       │
       ▼
 A = rewards − unnorm_b  →  clamp(−5, 5)  →  [B]
       │
       ▼
 Build completion_mask [B, R]  (1 up to EOS, 0 after padding)
       │
       ▼
 per_token_kl = exp(ref_logp − logp) − (ref_logp − logp) − 1   [B, R]
       │
       ▼
 per_token_loss = −logp × A  +  β_kl × kl
 masked mean over valid tokens  →  policy_loss  (scalar)
       │
       ▼
 loss.backward()
       │
       ▼
 value_tracker.update(rewards, logps.detach(), masks)
   →  ρ computed  →  α, β_dist updated  (BEFORE optimizer.step!)
       │
       ▼
 optimizer.step()  →  optimizer.zero_grad()
```

---

### 4.13 Mini Summary

- SPO uses **one response per prompt** — cheaper than GRPO's G responses — by maintaining a **persistent Bayesian baseline** across all training history.
- The Beta distribution tracks the model's "win rate" across batches: high rewards grow α, low rewards grow β_dist, the ratio gives the baseline.
- ρ (the forgetting factor) adapts to how fast the policy is changing: if the model improved a lot this step, old statistics are discounted more aggressively.

---

## Section 5 — Side-by-Side Comparison

### 5.1 All Four Loss Formulas

```
DPO:   L = -E[ log σ( β · (Δ_π - Δ_ref) ) ]

         where Δ_π   = avg_logp_θ(y_w)   - avg_logp_θ(y_l)
               Δ_ref = avg_logp_ref(y_w) - avg_logp_ref(y_l)

PPO:   L = -E[ min(ρ·A, clip(ρ, 1-ε, 1+ε)·A) ]
           + c₁ · MSE(V(s), r)
           + c₂ · E[log π_θ - log π_ref]

         where ρ = exp(log π_θ - log π_old)
               A = r - V(s)

GRPO:  L = -E_t[ exp(logp_t - sg(logp_t)) · A_GRPO ]
           + β_kl · E_t[KL_t]
           (masked mean over response tokens)

         where A_GRPO = (r - group_mean) / group_std
                        then batch-normalized

SPO:   L = -E_t[ logp_t · A_SPO ]
           + β_kl · E_t[KL_t]
           (masked mean over response tokens)

         where A_SPO = r - unnorm_baseline   (clamp ±5)
               unnorm_baseline = (α/(α+β_dist)) × 6 - 3
```

---

### 5.2 All 4 Algorithms — Same Prompt, Different Paths

```text
                                    Prompt x
                                        │
         ┌──────────────────────────────┼──────────────────────┬──────────────────────┐
         ▼                              ▼                      ▼                      ▼
       DPO                            PPO                    GRPO                    SPO
 ─────────────────────         ─────────────────────   ─────────────────────   ─────────────────────
 Needs (y_w, y_l) from         Actor generates y        Model generates         Model generates
 pre-collected dataset.         (1 per prompt).          G responses             1 response.
 No generation step.            Reward Model → r.        per prompt.             Beta tracker → b.
                                Critic → V(s).           Reward Model → r.       Reward Model → r.
                                A = r − V(s).            Group mean/std → A.     A = r − b.
                                Ratio ρ clipped.         No Critic needed.       (cross-batch)
         │                              │                      │                      │
         ▼                              ▼                      ▼                      ▼
 policy_loss (DPO)             policy_loss                policy_loss             policy_loss
 ref model comparison          + value_loss               (per-token)             (per-token)
                               + kl_ref_loss              + kl_loss               + kl_loss
                                                                                  update tracker
         │                              │                      │                      │
         └──────────────────────────────┴──────────────────────┴──────────────────────┘
                                                    ▼
                                          Update Policy θ
```

---

### 5.3 Models Required

| Algorithm | Policy | OldPolicy | Reference | Critic | Reward | Tracker | **Total** |
|---|---|---|---|---|---|---|---|
| DPO | YES | — | YES | — | — (offline) | — | **2** |
| PPO | YES | YES | YES | YES | YES | — | **5** |
| GRPO | YES | — | YES | — | YES | — | **3** |
| SPO | YES | — | YES | — | YES | Beta stats | **3+1** |

---

### 5.4 Advantage Source and Baseline

| Algorithm | Baseline `b` | How computed |
|---|---|---|
| DPO | Implicit | Log-ratio margin between chosen/rejected |
| PPO | `V(s)` from Critic neural network | Trained alongside Actor every step |
| GRPO | Group mean within prompt's G responses | Computed fresh each batch |
| SPO | Beta distribution mean `α/(α+β_dist)` | Accumulated across all training history |

---

### 5.5 Stability Mechanism

| Algorithm | How updates are kept safe |
|---|---|
| DPO | Sigmoid saturates — loss never becomes infinite regardless of margin |
| PPO | Ratio `ρ` clipped to `[1-ε, 1+ε]` — hard gradient zeroing |
| GRPO | Advantages double-normalized (within group, then across batch) |
| SPO | Advantages clamped `±5`; ρ forgetting limits how fast baseline moves |

---

### 5.6 Data Requirements

| Algorithm | Data needed at training time | Online generation? |
|---|---|---|
| DPO | (prompt, chosen, rejected) triplets | **No** — purely offline |
| PPO | Prompts only (responses generated live) | **Yes** |
| GRPO | Prompts only (G responses generated live) | **Yes** |
| SPO | Prompts only (1 response generated live) | **Yes** |

---

### 5.7 Algorithm Evolution — Each Solved a Specific Problem

```text
 ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
 │ DPO                 │     │ PPO                 │     │ GRPO                │     │ SPO                 │
 │ ─────────────────── │     │ ─────────────────── │     │ ─────────────────── │     │ ─────────────────── │
 │ Innovation:         │     │ Innovation:         │     │ Innovation:         │     │ Innovation:         │
 │ Eliminated reward   │     │ Online RL with      │     │ Eliminated Critic.  │     │ Eliminated G>1.     │
 │ model training and  │ ──▶ │ safe updates.       │ ──▶ │ Group mean/std      │ ──▶ │ Persistent Beta     │
 │ online generation.  │     │ Clipped ratio stops │     │ replaces V(s).      │     │ baseline replaces   │
 │ Pure offline.       │     │ gradient explosion. │     │ Simpler, cheaper.   │     │ group statistics.   │
 │ Memory: 2 models    │     │ Memory: 5 models    │     │ Memory: 3 models    │     │ Memory: 3 + tracker │
 └─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

---

### 5.8 When to Use Which

| Algorithm | Best for | Main drawback |
|---|---|---|
| **DPO** | High-quality preference datasets available; stable, predictable training | Cannot adapt to the model's current ability (offline); data collection is expensive |
| **PPO** | Theoretically grounded online RL; production-quality training | 5 models needed; most hyperparameters to tune; highest memory cost |
| **GRPO** | Online RL without a Critic; cleaner code; strong performance (used in DeepSeek-R1) | G≥4 responses per prompt needed for stable group stats; G× memory at generation |
| **SPO** | Single-response online RL; memory-constrained settings; stable cross-batch baseline | New hyperparameters (D_half, clip_lower, clip_upper); Beta distribution less intuitive to debug |

---

## Quick Reference: Tensor Shapes

| Variable | Shape | Meaning |
|---|---|---|
| `enc.input_ids` | `[B, P]` | Tokenized prompts |
| `gen_out` | `[B, P+R]` | Prompt + response (PPO/SPO) |
| `outputs` | `[B*G, P+R]` | Prompt + response, G per prompt (GRPO) |
| `completion_ids` | `[B, R]` or `[B*G, R]` | Response tokens only |
| `rewards` | `[B]` or `[B*G]` | Scalar reward per sequence |
| `values` | `[B]` | Critic's value estimate (PPO only) |
| `advantages` | `[B]` or `[B*G]` | `r - baseline` |
| `per_token_logps` | `[B, R]` or `[B*G, R]` | Per-token log-prob under current policy |
| `ref_per_token_logps` | `[B, R]` or `[B*G, R]` | Per-token log-prob under reference |
| `per_token_kl` | `[B, R]` or `[B*G, R]` | Schulman KL per token |
| `completion_mask` | `[B, R]` or `[B*G, R]` | 1 for valid tokens, 0 for padding |
| `policy_loss` | scalar | Final loss to call `.backward()` on |

*`B` = batch size · `P` = prompt length · `R` = response length · `G` = num\_generations*
