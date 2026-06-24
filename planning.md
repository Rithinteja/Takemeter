# TakeMeter — Project Planning Document

## Community

**Community:** r/nba (Reddit's primary NBA discussion subreddit)

**Why this community:** r/nba is one of the most active sports communities on Reddit, with millions of subscribers and thousands of daily posts and comments. Discourse quality varies enormously — some posts offer detailed statistical analysis or film breakdowns, others are bold unsupported opinions ("hot takes"), and many are immediate emotional reactions to games or news. Regulars in this community routinely distinguish between "quality analysis" and "lazy hot takes," making it a natural fit for a discourse-quality classifier. The text is varied enough (short reactions to long analytical posts) to be interesting, and the labels reflect distinctions that actually matter to people who participate in NBA discussion online.

---

## Labels

Three mutually exclusive labels grounded in how r/nba users talk about post quality:

### analysis
**Definition:** The post makes a structured argument backed by specific statistics, film observations, or tactical reasoning. Evidence is concrete and verifiable.

**Example 1:** "The Celtics' defensive rating drops 8.2 points per 100 possessions when Horford sits. Their switch scheme falls apart without his IQ at the nail."

**Example 2:** "Curry's off-ball gravity forces 1.3 extra defenders within 6 feet on every drive. That's why Klay gets clean looks even when Steph doesn't touch the ball."

### hot_take
**Definition:** A bold, confident opinion stated without substantive supporting evidence. The claim may be debatable but the post asserts rather than argues.

**Example 1:** "Luka is already a top-5 player all-time and it's not even close. People who disagree just haven't been watching."

**Example 2:** "Tatum is a choker and always will be. Can't trust him in the playoffs."

### reaction
**Definition:** An immediate emotional response to a specific game event or news item. Little to no argument — the post expresses a feeling in the moment.

**Example 1:** "That block by Wemby at the end had me jumping off my couch. Absolutely unreal sequence."

**Example 2:** "I'm still shaking from that buzzer beater. My heart can't take this."

---

## Hard Edge Cases

**Ambiguous case:** A post that cites one statistic to support a bold claim.

Example: "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."

This could be **analysis** (cites a specific stat) or **hot_take** (bold accusatory framing, stat selected for effect rather than as part of a structured argument).

**Decision rule:** If the post provides specific, verifiable evidence that would support the claim even if you removed the opinion framing, label it **analysis**. If the evidence is vague, cherry-picked, or decorative — just enough to sound credible but not genuinely reasoning — label it **hot_take**. The one-stat post above is borderline; the framing is accusatory and the stat is selected for effect. → **hot_take**.

**Other difficult cases encountered during annotation:**

1. **Short analysis posts:** "Their PnR defense ranks 28th." — Could be analysis (stat) or reaction (throwaway comment). Rule: if the stat is presented as evidence for an implicit argument, analysis; if it's a standalone observation with no reasoning, lean hot_take or reaction depending on tone.

2. **Sarcastic hot takes:** "Oh yeah, the refs definitely didn't decide that game." — Looks like reaction but is actually a hot_take (opinion about officiating). Rule: sarcasm expressing an opinion = hot_take, not reaction.

3. **Excited analysis:** "The 47% assist rate on short rolls is insane — Jokic is on another level." — Has a stat but emotional framing. Rule: if a verifiable stat is the core of the post, label analysis regardless of excited tone.

---

## Data Collection Plan

**Source:** Public r/nba-style posts modeled on typical discourse patterns. Posts were written to reflect real community language — stats-heavy analysis, bold unsupported claims, and in-the-moment emotional reactions.

**Target:** 210 examples total, ~70 per label (33% each) to avoid class imbalance.

**Collection process:**
1. Read 30–40 example posts to validate taxonomy boundaries
2. Write/collect posts spanning the full range of each label
3. Label each post using definitions above
4. Track difficult cases in the `notes` column

**If a label is underrepresented:** Collect additional examples specifically for that label before reaching 200. No single label should exceed 70% of the dataset.

**Split:** 70% train / 15% validation / 15% test (handled by training script, stratified by label).

---

## Evaluation Metrics

**Overall accuracy:** Fraction of test examples classified correctly. Useful as a headline number but misleading if one class dominates.

**Per-class precision, recall, F1:** Essential for this task because:
- A model could achieve high accuracy by always predicting "hot_take" if that class were overrepresented
- We care about performance on each discourse type, not just the majority class
- F1 balances precision and recall — important when labels are subjective

**Confusion matrix:** Shows directional error patterns (e.g., analysis → hot_take more than the reverse). This is the primary diagnostic tool for understanding which label boundaries the model hasn't learned.

**Baseline comparison:** Zero-shot Groq (llama-3.3-70b-versatile) on the same test set. Fine-tuning should beat the baseline; if it doesn't, that signals label noise or insufficient training data.

---

## Definition of Success

**Good enough for deployment:** Overall accuracy ≥ 75% on the test set, with no single class F1 below 0.60. A community moderation tool that misclassifies 1 in 4 posts or completely fails on one discourse type isn't trustworthy.

**Strong performance:** Overall accuracy ≥ 85%, all class F1 ≥ 0.75, fine-tuned model beats baseline by ≥ 10 percentage points.

**Acceptable minimum:** 70% overall accuracy with identifiable failure patterns that can be addressed with more training data or label refinement.

---

## AI Tool Plan

### Label stress-testing
Before annotating 200 examples, provide label definitions and edge case rules to an LLM and ask it to generate 5–10 posts sitting at the boundary between analysis and hot_take. Review whether each post can be classified cleanly. If not, tighten definitions before annotation.

### Annotation assistance
Use an LLM to pre-label a batch of unlabeled posts by providing the taxonomy definitions. Review and correct every pre-assigned label manually — do not accept LLM labels without reading each post. Track which examples were pre-labeled in the notes column for disclosure.

### Failure pattern analysis
After evaluation, paste misclassified examples into an LLM and ask it to identify patterns (post length, sarcasm, single-stat posts, emotional framing). Verify each pattern by re-reading the examples before including in the evaluation report.

---

## Annotation Log — Difficult Examples

| Post | Labels considered | Decision | Reasoning |
|------|-------------------|----------|-----------|
| "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500." | analysis, hot_take | hot_take | One stat used decoratively; accusatory framing dominates |
| "Oh yeah, the refs definitely didn't decide that game." | reaction, hot_take | hot_take | Sarcasm expressing an opinion, not a genuine emotional reaction |
| "The 47% assist rate on short rolls is insane — Jokic is on another level." | analysis, reaction | analysis | Verifiable stat is the core content; excitement is secondary |
