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

## 4. Round-3 update (2026-09-28): positioning against Alignment Inertia and newer work

I read the Alignment Inertia PDF in full (arXiv:2609.27333v1, 23 Sep 2026). I opened the arXiv page of every new
entry below to confirm its ID, authors and date. A subagent's web search also found no paper using this design.

### 4.1 Alignment Inertia (Barreto, Roesti, Tahaei; eBay Responsible AI), point by point

| | Alignment Inertia | Stale Weights |
|---|---|---|
| Where the prior comes from | The provider's pretraining and safety alignment. It is unknown and is inferred from an instruction-free baseline. | The operator's own LoRA fine-tune on old decisions, installed at a known dose. The prior is known by construction. |
| What the LoRA does | It trains **toward the new** policy: the operator uses fine-tuning to impose it. | It installs the **old** policy. The new one arrives only in the prompt. |
| Task | Content moderation (REMOVE/ALLOW) on PUBHEALTH and Davidson hate speech. The policies are prose categories ("contested", "offensive"), and the dataset classes are mapped to labels. | Refund decisions under numeric rules (a window in hours and a fee in dollars). The correct answer is computed by code, so there is no label ambiguity. |
| Models | Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.2; N = 500 per condition; one run. | Qwen2.5-14B-Instruct (round 3 adds seeds and Mistral-Small-24B); 64 window and 32 fee items per prompt; paired tests. |
| Main finding | A system prompt leaves 20–90% of conflicting cases unchanged ("inertia"). The effect is asymmetric by direction. LoRA toward the target sometimes *worsens* override success. TRAK influence predicts which cases resist (AUC ≥ 0.85 in 7/8). | A policy *stated* in the prompt beats the fine-tuned one: no reversion beyond the base model. A rule the prompt *omits* is filled from the fine-tune, even under "apply only the rules written here". Training with the policy text in the prompt prevents that. |
| What it does not cover | Omitted rules and gap filling; explicit scope instructions; operator fine-tunes as the prior; policy-in-prompt training; agents and retrieval; seeds or fine-tuned controls. | Attribution (TRAK); safety-relevant or subjective content; confidence. |

**How the two relate.** The findings are complementary, and they appear to disagree only at first sight. They find
high inertia against a strong, value-laden alignment prior. We find almost none against a shallow 200-step operator
fine-tune when the conflicting rule is stated. There are two likely reasons, neither tested:

1. Their prior is far stronger.
2. With prose categories, non-compliance cannot be separated from disagreement about how the policy applies to an
   item. Is a claim "contested"? The dataset label decides. Our numeric rules remove that ambiguity.

The case they cannot see is the one we find the effect in. When the prompt is *silent*, the operator's own old
fine-tune wins. A deployer who reads their paper learns "prompts may not override training". Ours adds "and
fine-tunes leak into whatever the prompt leaves unsaid; here is what does and does not stop that".

**Scoop risk: low.** They pre-empt any general claim that "prompts can't override training". We don't make that
claim, and our stated-policy result points the other way for this setting. Nothing in their paper touches omitted
rules, the scope instruction, the policy-in-prompt training control or the agentic lookup.

### 4.2 Newer related work (verified 2026-09-28)

| Paper | What it does | Relation |
|---|---|---|
| Nguyen, Kok-Shun, Du, Sunyaev, *Follow the Norm: Accounting for Fine-Tuning and Prompt Effects on Model Rationales*, arXiv:2608.13250 (Aug 2026) | LoRA on norm-following or norm-breaking Social Chemistry data (Llama-3.2-11B, Qwen-3.5-9B, Pixtral-12B), crossed with system prompts. System prompts both suppress and elicit the fine-tuned tendency. | The closest *paradigm*: a tendency implanted by LoRA, crossed with the system prompt. But the domain is social norms, and there are no numeric rules, omitted rules or agents. Cite as the closest neighbour. |
| Sun, He, *When Memory Updates but Behavior Does Not* (STALE benchmark), arXiv:2608.01619 (Aug 2026) | Agents keep acting on outdated user state after a *memory* update. | The same phenomenon at the memory level. No weights are changed. Cite to separate weight-level staleness from memory-level staleness. |
| Lampinen et al., *On the generalization of LMs from in-context learning and finetuning*, arXiv:2505.00661 (FoRLM workshop, NeurIPS 2025) | Controlled study: in-context learning generalises more flexibly than fine-tuning. | Background for why a stated rule wins. |
| Hossain et al., *Right Knowledge, Wrong Answer: Parametric Temporal Conflict*, arXiv:2606.20959 (2026) | Models hold newer facts but default to the outdated ones. Steering recovers them. | The same "stale default" at the level of facts, with no fine-tuning or prompt. |
| Hubinger et al., *Sleeper Agents*, arXiv:2401.05566 (2024) | Implanted behaviours persist through later safety training. | Persistence through *training*. Ours is persistence into *prompt gaps*. |
| Betley et al., *Emergent Misalignment*, arXiv:2502.17424 (2025) | Narrow fine-tuning surfaces broad unprompted behaviour. | A latent disposition surfacing unasked. Different direction. |
| Mu, Lu, Lavery, Wagner, *A Closer Look at System Prompt Robustness*, arXiv:2502.12197 (2025); Qin et al., *SysBench*, arXiv:2408.10943 (ICLR 2025) | System-prompt adherence benchmarks. Fine-tuning *for* adherence. | Adherence in general, and no conflict with an operator fine-tune. |
| Zhou et al., *Context-faithful Prompting*, arXiv:2303.11315 (2023) | Prompting for context faithfulness and abstention. | The strict and notice sentences are this kind of intervention. We test it against a fine-tuned rule. |
| Sistla et al., *TRACE*, arXiv:2607.22639 (2026) | LoRA for enterprise tool retrieval under business rules. | The nearest neighbour to the agentic part. About keeping tool knowledge, not about skipping a lookup. |

### 4.3 Verdict (updated)

**Already known; don't claim it.** Parametric priors can beat context. Fine-tuning can erode prompt adherence.
Models fill gaps in underspecified prompts. Implanted behaviours persist (sections 1–2; Sleeper Agents). Operator
prompts don't always override training (Alignment Inertia).

**New here, as far as the searches found:**

1. A controlled, known-ground-truth crossover. The operator's *own* fine-tune on decisions under an old policy is
   the prior. A new policy is then stated, omitted or declared removed in the prompt.
2. A stated rule wins, but an **omitted rule is filled from the fine-tune**. A scope instruction ("apply only the
   rules written here") doesn't stop it. A fine-tuned control trained on the same data with the policy in its
   prompts fills the gap with a generic prior instead.
3. A metric, "decided by the trained window", that separates a recalled rule from a generic prior and from a class
   bias.
4. The agentic analogue: fine-tuning on logs from before a lookup tool existed teaches the agent to skip the lookup.

**What would make this a workshop contribution (ICLR 2027 workshop, ~Feb 2027).**

- (a) The omitted-rule result replicated over ≥ 3 seeds and ≥ 2 model families, with the fine-tuned controls.
- (b) A mitigation comparison a practitioner can use: instruction, change notice, explicit removal, policy-in-prompt
  training, data mixing.
- (c) The agentic tool-log control (lookups in the training logs, then the store is swapped).
- (d) A clear framing against Alignment Inertia: *stated* conflicts vs *silent* gaps, and an alignment prior vs an
  operator fine-tune.

Round 3 (RESEARCH_AGENDA.md) is designed to supply (a)–(c). Section 2 of FINDINGS records what it found.
