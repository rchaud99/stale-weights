# Stale Weights: digest (round 3, 2026-09-28)

> **Superseded.** This is the Mac pilot's round-3 digest. The current results are FINDINGS.md §9 and
> `reports/progress/progress_05_FINAL_2026-09-30_1013.pdf`.

**In one line:** the one result that survived last round also survives a second wording. Two practitioner
mitigations were tested: writing the removal out fixes the stale fill, and a change notice doesn't. You stopped the
GPU after ~45 min, so no seeds, models or domains were added yet. Everything is still Qwen2.5-14B, seed 0.

## New since 2026-09-27

- **Related work.** RELATED.md §4 has a point-by-point comparison with Alignment Inertia and 9 newer papers, each
  checked on arXiv. There is no scoop. Alignment Inertia measures *stated* conflicts against an unknown alignment
  prior (moderation; Llama-8B, Mistral-7B; one run, no CIs). We measure an operator's own fine-tune, installed at a
  known dose, and find the risk is in *silent gaps*, not stated conflicts.
- **Plan.** RESEARCH_AGENDA.md ranks the experiments, gives a gap table against Alignment Inertia, and holds a
  preregistration committed before every run. The code for all of it is written and tested (67 tests): seeds, a
  second family (Mistral-24B), a third (Llama-8B, falling back to phi-4), a second domain with a *categorical* rule,
  the tool-log control, and data mixing.
- **Ran:** preregistered mitigations on 3 of 7 adapters. FINDINGS §8 has the tables and exact tests.

## Findings that survive (Qwen2.5-14B, 1 seed, 96 test scenarios)

1. **A stated policy beats the fine-tune.** Told the other policy in full, the tuned models match the untuned base
   (A told B: 8/24 vs 7/24; fees 0/32 stale).
2. **An omitted rule is filled from the weights, even under "apply only the written rules".** A-tuned denies 56/56
   bookings past its old 12 h window. The same data trained with the policy text in the prompt denies 21–24/56 and
   never cites 12 h (p < 1e-9). *New:* the same holds with a second wording, where DTW is 63 vs 40–41/64 (Holm
   p ≤ 3e-5; A only).
3. ***New:* stating the removal explicitly fixes it.** With "there is no time limit", conflict denials drop from
   38 to 4–6/56 (A) and from 12–15 to 0/32 (B), Holm p ≤ 0.003 in 4/4 tests. A's reasons still mention the old
   12 h on 8–9/56 before refunding.
4. ***New:* a change notice does *not* fix it** ("replaces the previous policy; old rules no longer apply"). A stays
   at 53–54/56. B goes *up* against a neutral sentence on one prompt (18 vs 7/32, Holm p = 0.021), but the notice
   raises the control too, so that part is not a stale-weights effect.

## What died (and stays dead)

The strict sentence's *rise* (a format effect). The knowing–using lag and the zero escalations (artifacts). The
agentic "weights beat retrieval" lead (confounded; its control is coded but not run). "The notice makes stale
weights worse" is not claimed.

## Résumé bullet (defensible today)

> Ran a local LoRA study (Qwen2.5-14B, MLX): support models fine-tuned on old-policy logs follow a new policy stated
> in the prompt, but fill rules the prompt omits from their weights even when told to apply only written rules
> (56/56 vs ≤ 24/56 for a policy-in-prompt control); stating removals explicitly fixes it (38 → ≤ 6/56).

If asked, say it is one seed and one model family. That is true, and the planned replication is concrete.

## Workshop paper (ICLR 2027): outline and gaps

**Outline.** (1) The deployment story (Moffatt v. Air Canada). (2) The setup: an implanted policy crossed with
stated, omitted and removed rules, plus fine-tuned controls. (3) Stated conflicts: safe. (4) Silent gaps: filled
from the weights; instructions don't stop it. (5) Mitigations head to head. (6) Agents: lookups missing from the
logs; the tool-log control. (7) Relation to Alignment Inertia and knowledge conflicts.

**Gaps to Alignment Inertia's scope:** 3 model families (have 1), 3 seeds (have 1), 2 rule types (have 1), a dose
curve (0), the tool-log control (0), data mixing (0). About 42 h of GPU, all queued and preregistered.
Attribution (TRAK) stays out of scope.

## Next 3 experiments (when the GPU is allowed)

1. **Finish mitig + seeds 1 and 2** (`round3 mitig seeds1 seeds2`, ~9 h). Does the headline replicate across seeds?
2. **Mistral-Small-24B and Llama-3.1-8B** (`round3 model2 model3`, ~11 h). Does it replicate across families?
3. **Second domain: categorical final-sale rule** (`round3 domain2`, ~4 h). Is it specific to numeric thresholds?

Compute used: ~45 min GPU. API spend: **$0**. Progress PDF: `reports/progress/progress_01_2026-09-28_1945.pdf`.
