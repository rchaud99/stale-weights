# DRAFT, for Raghav to review and send (or not). Not sent.

**To:** Renata Barreto, Markelle Roesti, Mohammad Tahaei (eBay Responsible AI)
**Subject:** Alignment Inertia and a complementary result on operator fine-tunes and omitted policy rules

Hi Renata, Markelle and Mohammad,

I enjoyed "Alignment Inertia" (arXiv:2609.27333). The framing of override success as an operator-facing audit is
useful, and I've been working on a closely related question from the other side.

In a small local study (Qwen2.5-14B, LoRA via MLX), the prior is the operator's *own* earlier fine-tune: a support
model trained on decisions made under an old refund policy, with ground truth computed by code. The policy is then
changed only in the system prompt. Two things came out. When the new policy is stated in full, it wins; we see no
reversion beyond the untuned base. When the prompt omits a rule, the model fills it from the fine-tune, even when told
to "apply only the rules written in this policy". An adapter trained on the same data with the policy text in its
prompts does not (56/56 vs 21–24/56 conflict items; exact McNemar p < 1e-9). Stating the removal explicitly fixes it;
a "this policy replaces the previous one" notice doesn't.

This seems complementary to your stated-conflict results: stated conflicts vs silent gaps, and an alignment prior vs
an operator fine-tune. I'd be glad to hear whether this matches anything you've seen, and I'm happy to share the code
and data.

Best,
Raghav Chaudhary

*(Before sending: only quote numbers that are still one-seed-only if you say so. If the round-3 replication has run
by then, update them.)*
