# Related work and novelty check: Stale Weights

Checked 2026-09-24. I confirmed every entry by web search or by reading its arXiv or proceedings page. A venue appears only when I saw it confirmed. **(unverified)** marks the one venue I did not confirm.

## 1. Closest prior work

| # | Paper | What it does | How we differ |
|---|---|---|---|
| 1 | Longpre et al., *Entity-Based Knowledge Conflicts in QA*, EMNLP 2021 (arXiv 2109.05052) | Swaps answer entities in passages and measures the "memorization ratio": how often the model gives its memorized answer instead of the one in the context. | Their conflicts are pretraining facts. Ours is a decision policy we implant with LoRA at a controlled dose. |
| 2 | Xie et al., *Adaptive Chameleon or Stubborn Sloth*, ICLR 2024 (arXiv 2305.13300) | Models accept coherent counter-memory evidence but show confirmation bias. | We use system-prompt rules, and we know the prior because we installed it. |
| 3 | Wu, Wu & Zou, *ClashEval*, NeurIPS 2024 D&B (arXiv 2404.10198) | Graded perturbations of retrieved facts. Wrong context is adopted more than 60% of the time, less often as it deviates further from the prior. | Their deviation curve applies directly to our fee and window thresholds. |
| 4 | Du et al., *Context versus Prior Knowledge in LMs*, ACL 2024 (arXiv 2404.04633) | Persuasion and susceptibility scores. Familiar entities are harder to move. | Supplies a metric, and predicts our "familiar norm vs arbitrary number" factor. |
| 5 | Kortukov et al., *LLM Behaviors Under Context-Memory Conflicts With Real Documents*, COLM 2024 (arXiv 2404.16032) | Updates fail more often when the stale answer also appears in the context. | Warns that policy-C prompts mentioning A's values can inflate reversion. |
| 6 | Goyal et al., *Context-Parametric Inversion*, ICLR 2025 (arXiv 2410.10796) | During instruction tuning, reliance on context under conflict first rises and then falls. | The closest result on how training dose affects context reliance. We use narrow policy fine-tuning, not general instruction tuning. |
| 7 | Wei et al., *Larger LMs do in-context learning differently*, arXiv 2303.03846 (2023) | Only large models override their priors when shown flipped labels in context. | Predicts our size factor. They use in-context exemplars, not a stated rule. |
| 8 | Qi et al., *Fine-tuning Aligned LMs Compromises Safety…*, arXiv 2310.03693 (ICLR 2024, unverified) | Benign fine-tuning erodes aligned behaviour. | They show general erosion. We test targeted reversion to one learned policy. |
| 9 | Lyu et al., *Keeping LLMs Aligned After Fine-tuning: Prompt Templates*, NeurIPS 2024 (arXiv 2402.18540) | Whether behaviour degrades depends on the prompt format matching between fine-tuning and test ("Pure Tuning, Safe Testing"). | Bears directly on our "policy present in training prompts" factor. |
| 10 | Wallace et al., *The Instruction Hierarchy*, arXiv 2404.13208 (2024); Zhang et al., *IHEval*, NAACL 2025 (arXiv 2502.08745) | System message vs user conflicts. Accuracy collapses on conflicting cases. | Their conflicts are between messages. Ours is between the weights and the system prompt. |
| 11 | Yao et al., *τ-bench*, arXiv 2406.12045; Barres et al., *τ²-Bench*, arXiv 2506.07982; Mu et al., *RuLES*, arXiv 2311.04235 | Benchmarks for following airline/retail policies and simple rules. Models fail often even with no conflict. | A source of policy text and metrics. None of them fine-tune on one policy and then prompt another. |
| 12 | Panavas et al., *HANDBOOK.md*, arXiv 2607.25398 (2026) | 65 agentic tasks. Each mutates the rules and thresholds of a base handbook so memorization cannot help. Best pass rate is 36.2%. | Uses mutation to remove priors. We deliberately install a prior. |

Also relevant: Betley et al., *Tell me about yourself*, ICLR 2025 (arXiv 2501.11120), and Treutlein et al., *Connecting the Dots*, NeurIPS 2024 (arXiv 2406.14546). Both show that fine-tuning on examples alone installs a latent rule the model can state. Xu et al., *Knowledge Conflicts for LLMs: A Survey*, EMNLP 2024, gives the taxonomy.

## 2. Verdict

**Known; do not claim these:**
- Parametric knowledge can override context, and more so with higher confidence, familiarity or training exposure (1, 3, 4, 6).
- Fine-tuning erodes prompted and aligned behaviour, depending on the prompt format (8, 9).
- Scale helps a model override its priors (7).
- Policy following is hard even without a conflict (11, 12).
- Example-only fine-tuning installs implicit rules (Betley; Treutlein).

**Appears new (no direct precedent found):**
- A controlled crossover: the same model is LoRA-trained on opposite implicit *decision policies*, then prompted with its own policy, the other one, or an unseen one. The base model is the control.
- Compliance with operational rules as the outcome, rather than factual QA.
- Dose-response curves for the conflict between weights and the system prompt.
- The restrictive ("no") vs permissive ("yes") asymmetry.
- Familiar norms vs arbitrary numbers in a policy setting.

My searches for "fine-tune on old policy, update the prompt" found only blog posts. Write "to our knowledge".

**Concurrent work (added 2026-09-27).** Barreto, Roesti and Tahaei, "Alignment Inertia: Auditing the Durability of Training Data Influence Through Policy Override Resistance" (eBay Responsible AI, arXiv:2609.27333, 23 Sep 2026).
- **What they measure:** an "Override Success Rate" for system prompts and LoRA trying to override a model's inherited *alignment* prior on content moderation (medical misinformation, hate speech; Llama-3.1-8B, Mistral-7B).
- **What they find:** substantial, asymmetric inertia; LoRA sometimes worsens it; TRAK attribution predicts which cases resist.
- **How this study differs:**
  - The prior here is the operator's *own* earlier fine-tune on a business rule, not safety alignment.
  - It tests *omitted* rules and an explicit "apply only stated rules" instruction, which they don't.
  - It finds that a *stated* conflicting policy does override a 200-step LoRA prior. That is consistent with their result if a shallow fine-tuned rule is a weaker prior than safety training.

**Suggested framing:** a measurement study of a deployment failure mode. Teams fine-tune on historical support logs and later change policy only in the system prompt; how much of the old policy leaks through? Present it as knowledge-conflict research moved from facts to operational rules, not as a new phenomenon.

## 3. Design suggestions

1. **Metrics.** Report three outcomes: follows the prompted policy, follows the trained policy, other. Add Longpre's memorization ratio, trained / (trained + prompted). Add a log-probability gap between the A and B answers (in the spirit of Du's persuasion score), which is cheaper and less noisy than sampling. Use τ-bench's pass^k for multi-turn reliability.
2. **Controls that separate reversion from general erosion.** (a) A neutral LoRA trained on the same dialogues with decisions that do not depend on the policy. (b) A no-policy prompt plus a Betley-style probe ("What is the fee?") to confirm the policy was learned. (c) The policy-present training variant, which Lyu et al. predict will matter.
3. **Grade the conflict and log checkpoints.** Following ClashEval, set C at several distances from A and B (e.g. 13 h / 24 h / 72 h; $36 / $500). Following Kortukov, vary whether stale values appear in the prompt. Following Goyal, evaluate at intermediate checkpoints, because context reliance can be non-monotonic in dose.
