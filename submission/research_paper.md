# CTR Opportunity Scoring: Ranking Pages for Title/Snippet Review

**Roha**  
ML Internship Capstone, Search Intelligence  
September 2026

---

## Abstract

One-third of pages ranking in positions 1–20 capture less than 0.5% click-through rate despite hundreds of monthly impressions—indicating a "ranking but invisible" opportunity more cheaply fixed by title/snippet rewrite than full content refresh. This work trains a binary classifier on 28,795 search-data records to rank pages by their fit to this pattern, validating precision@50 against a hand-built rule baseline and analyzing model behavior under client-grouped test splits. A simpler variant excluding leaked position/impression signals achieves 0.48 precision@50 on secondary features (word count, scroll rate, content age), suggesting genuine but modest structure in engagement metrics. The ranked list is decision-support for a content ops team prioritizing review capacity (~50 pages/week against a ~9,700-page candidate backlog), not a verdict; limitations include single-window snapshot evaluation and rule-based label design.

---

## 1. Introduction / Problem Statement

### The Gap

Analysis of 30,000 content pages over a trailing 90-day window revealed that **32.5% of pages ranking in top-20 positions pull less than 0.5% CTR** (median 0.07%) despite median 731 monthly impressions. At 50 pages/week review capacity, a flat threshold-based queue would require ~195 weeks (~4 years) to clear this backlog. The ranking itself is working—median position is 10.8 (page 1–2)—but the click-through is not.

### Why This Matters

A title/snippet rewrite costs 30–60 minutes per page; a full content refresh costs days. If CTR is being suppressed by poor metadata at a given rank, metadata review is the faster, lower-risk fix, and should be triaged *before* deciding whether full refresh is needed. A ranked scoring system can prioritize the reviewers' limited weekly capacity toward pages most likely to have this specific problem.

### What This Work Does

This capstone builds a machine-learning classifier to score pages by their likelihood of fitting the "high visibility, low click capture" pattern. The output is a ranked list for human review, validated against transparent decision-rule baselines and inspected for sign of real signal vs. label leakage. Limitations are documented honestly; the goal is decision-support, not automation.

---

## 2. Data

### Source
FlyRank ML Internship dataset, gated, 2-minute access via Hugging Face.

### Unit of Analysis
One row = one content item for one client (`content_id` + `client_id`), aggregated over a fixed **90-day trailing window as of export date**. No time-series; one snapshot.

### Filtering & Grain
- Start: 30,000 rows (starter CSV, all have `impressions_90d > 0` and `content_age_days >= 90`)
- Dropped: 1,205 rows where `avg_position == 0` (flagged in data dictionary as "no position data," not rank 0)
- **Final working set: 28,795 rows**

### Key Fields
| Field | Count | Missing % | Notes |
|-------|-------|-----------|-------|
| `avg_position` | 28,795 | 0.0 | Position in SERP, median 10.8 |
| `impressions_90d` | 28,795 | 0.0 | Impressions over window, median 731, max 517k |
| `clicks_90d` | 28,795 | 0.0 | Clicks, derived `ctr = clicks / impressions` |
| `word_count` | 21,077 | 26.7 | Content length, max 40k+ words |
| `scroll_rate` | 28,755 | 0.4 | Engagement, 119 rows exceed 100% (measurement system difference, not clipped) |
| `content_age_days` | 28,795 | 0.0 | Days since publish, median ~400 |
| `freshness_tier`, `content_type`, `main_intent` | 28,795 | 0.0–5.6 | Categorical features, pre-binned |

### Public Safety
No client names, domains, URLs, query text, or raw keyword data in this paper. All identifiers are pseudonymized (`client_*`, `content_*`). Benchmark curves (`expected_ctr` by position) are external assumptions, not claims about Google's algorithm.

### Clients & Concentration
- **32 unique clients** back this 30,000-row set
- **Top 5 clients = 56.8%** of rows
- **Train split: 23 clients (22,024 rows) | Test split: 8 clients (6,771 rows)**
- No client overlap between train and test (leakage check passed)

---

## 3. Methodology

### Label Definition

**Binary flag:** page qualifies as a CTR opportunity if *all three* of:
1. `avg_position` ≤ 20 (ranking on page 1–2)
2. `impressions_90d` ≥ 500 (enough visibility to trust the CTR sample)
3. `ctr` < 0.5% (underperforming the position)

This is a **rule-based proxy**, not an observed future outcome. The decision to use these thresholds came from human review (w01) and capacity planning (9,759 qualifying rows, 50/week = ~195 weeks unranked). Real validation would require either (a) a future-window outcome check (did CTR rise post-review?) or (b) human reviewer feedback on a held-out top-50 list.

**Label rate (base rate):** 0.339 (33.9% of rows are positive)

### Features

**Numeric (8):**
- `avg_position`, `impressions_90d` — position and visibility (the main structural predictors)
- `word_count`, `content_age_days`, `days_since_last_update` — content maturity
- `scroll_rate`, `engagement_rate` — on-page engagement proxy
- `has_word_count` — binary, whether word_count is available (26.7% missing)

**Categorical (4):**
- `content_type`, `main_intent`, `competition_level`, `freshness_tier` — context and tier flags

**Explicitly Excluded & Why:**
- `ctr` — literal component of label (definitional leakage)
- `clicks_90d` — drives `ctr`, also implicit in label
- `trend_direction`, `trend_pct` — project-wide label-trap fields (overlap with low-CTR rule)
- `impressions_last_30d`, `impressions_prev_30d` — sub-windows of `impressions_90d` (redundant, plus window integrity check showed no leakage)
- `provider_used`, `model_used` — describe production, not search signal; 71–91% missing
- `ai_sessions_90d` — only 6.4% of rows have any AI session; AUC=0.502 (noise level)

### Train/Test Split

**Strategy:** Client-grouped shuffle split. All rows for each client go entirely into train *or* test, never split.
- **Why:** Client concentration (top 5 = 56.8%) means random split leaks client patterns into test. Grouped split forces the model to generalize to unseen clients.
- **Result:** Train 22,024 rows / 23 clients → Test 6,771 rows / 8 clients, zero overlap

### Baseline (Week 4)

Hand-coded decision tree with transparent reason codes:
```
if not (impressions >= 500 & position <= 20): not_eligible
elif position <= 3 & ctr < 1.0: page1_ctr_gap  
elif position > 3 & position <= 20 & ctr < 0.5: low_ctr_visible_page
elif impressions >= 2000: high_volume_opportunity
else: marginal_opportunity
```

**Baseline precision@50:** 0.60  
**Baseline precision@20:** 0.55  
**Random floor (base rate):** 0.32

### Models Trained

**Model 1: Logistic Regression (full features)**
- Preprocessor: StandardScaler (numeric) + OneHotEncoder (categorical)
- Solver: lbfgs, max_iter=500
- Purpose: Fast baseline, interpretability

**Model 2: Random Forest (full features)**
- n_estimators=100, max_depth=15, random_state=42
- Purpose: Main model, nonlinear patterns
- **Caveat:** Precision@50 = 0.98 on test split—suspiciously perfect. Analysis found the top 2 features are `impressions_90d` (0.364) and `avg_position` (0.351), which *directly comprise 2 of 3 components of the label itself*. Likely overfitting to label structure rather than discovering new signal.

**Model 3: Random Forest (harder, without position/impressions)**
- Features: word_count, content_age_days, days_since_last_update, scroll_rate, engagement_rate, content_type, main_intent, competition_level, freshness_tier
- Purpose: Honest signal test—can secondary engagement metrics alone score pages?
- Precision@50 = 0.48, Precision@20 = 0.50 (more credible than Model 2)
- Top features: word_count (0.286), scroll_rate (0.243), content_age_days (0.204)

### Validation Design

- **Metric:** Precision@K (K=20, 50)—only top K predictions are actionable per reviewer weekly capacity
- **Split:** Client-grouped (no data leakage across train/test clients)
- **Comparison:** vs. hand-coded baseline on same test set, same metric

---

## 4. Results

### Precision@K Comparison

| Model | Precision@20 | Precision@50 | Notes |
|-------|--------------|--------------|-------|
| Baseline (Week 4 rule) | 0.55 | 0.60 | Hand-coded thresholds |
| Logistic Regression | 0.70 | 0.58 | Linear model, full features |
| Random Forest (full) | 1.00 | 0.98 | **Overfitted to label structure** |
| Random Forest (harder) | 0.50 | 0.48 | No position/impressions—secondary signals only |
| Random Floor | 0.32 | 0.32 | Base rate (33.9% label rate) |

### Interpretation

**Model 2 (Random Forest, full features) is too good to be true.** Precision@50 = 0.98 means 49 of 50 top-ranked pages satisfy the label rule. This matches the data: the model's top features are the two structural components of the label (`impressions_90d`, `avg_position`), so it's essentially learning a compressed version of the rule itself, not discovering new signal. This is **model leakage**, not real performance.

**Model 3 (Random Forest, harder) is more honest.** Dropping position and impressions forces the model onto secondary signals:
- **word_count** (28.6% importance): Longer content slightly more likely to be flagged, surprising given w04 signal audit showed word_count correlation with CTR was weak (~-0.1). The model may be picking up on confounding (e.g., certain content types are longer *and* have CTR issues).
- **scroll_rate** (24.3% importance): Engagement is real, but precision@50 stays at 0.48, barely above random—modest signal, not strong.
- **content_age_days** (20.4% importance): Older pages are slightly more likely to be flagged, consistent with the idea that older content needs refresh.

Precision@50 = 0.48 is **not a strong improvement** over the baseline's 0.60. The honest takeaway: position and impressions structure the opportunity class well, but engagement metrics alone are weak predictors. For the review team, this means:
- A rank-by-position + impressions is practical and hard to beat.
- Engagement metrics (scroll, word count) add modest refinement but aren't transformative.
- Future work should include actual outcome data (did CTR improve post-review?) to train on *real* lift, not rule compliance.

### Top-20 Predictions (Model 3)

Model's top 20 pages by predicted probability (sample):
- Median predicted score: 0.95–0.98 (model is confident)
- False positives observed: 9 of top 20 have `ctr >= 0.5` (model flagged them but they don't satisfy the rule, suggesting they rank better in reality)
- False negatives: Some pages with strong engagement (scroll_rate > 5) still score below 0.50, as the model downweights position/impressions

**Example false positives (Model 3 flagged, rule didn't):**
- content_2f95ac170c67: 4,660 words, scroll_rate 0%, content_age 140d, predicted 0.987 → actual CTR 0.8% (above threshold, not an opportunity)
- content_99af6f9fbda3: 2,774 words, scroll_rate 0%, content_age 95d, predicted 0.980 → actual CTR 0.6% (above threshold)

These are long, recent articles with zero scroll activity—the model scores them high because word_count is high, but they don't satisfy the CTR < 0.5% threshold. Suggests word_count importance is **not causal** (writing longer doesn't fix low CTR) but **correlational** (certain content types are long and have CTR problems).

---

## 5. Limitations & Honest Framing

### 1. **Single Snapshot, No Future Validation**
This dataset is one 90-day window. There is **no future-outcome data**—I cannot verify whether flagged pages that *did* receive a title/snippet review actually improved CTR over the next window. The label is a *rule-based proxy* (matches a pattern), not observed truth. Real validation requires either:
- Future 30–60d re-measurement of flagged pages
- Human reviewer feedback on sampled top-50 predictions
- Experimental design (randomized review on a subset)

### 2. **Model 2 Leakage**
The Random Forest with full features (precision@50 = 0.98) is overfitted to label structure, not trustworthy. It's included in results to document the failure, not to recommend deployment. Model 3 (harder variant) is more credible because it's forced to work without the label's defining signals.

### 3. **Client Concentration**
Top 5 of 32 clients = 56.8% of rows. Even with client-grouped splits, this means the model is learning patterns from a concentrated set of clients. Results may not generalize to new, unrepresented clients equally well.

### 4. **Engagement Metrics Are Weak**
Scroll rate, engagement rate, and word count show modest correlation with CTR (w04 signal audit: r < 0.15). Model 3's precision@50 = 0.48 is only marginally better than random. The message: **position and impressions structure the opportunity space, but engagement signals add little on their own.**

### 5. **Rule-Based Label, Not Observed Outcome**
The threshold `CTR < 0.5%` is a human choice, not data-driven. Nudging it to 0.3% or 0.7% changes the candidate pool by ~40%. There's no ground truth here, only a heuristic.

### 6. **No Causal Claims**
This analysis is **correlational and observational**. I cannot claim that a title/snippet rewrite *causes* CTR to rise. The low CTR could be due to poor metadata, low search demand at that query intent, poor page quality, or measurement lag. The ranking is a *candidate list for human review*, not a verdict.

### Language Used
- **"Observed patterns"**: These are correlations in this snapshot, not universal laws
- **"Directional signal"**: Weak but consistent direction, not proof
- **"Candidates for review"**: Pages worth a human look, ranked by confidence in the opportunity pattern
- **"Decision-support"**: Helps prioritize a backlog, doesn't decide

---

## 6. Ranked Recommendations & Action Playbook

### Output Structure
For each page in the top-50 ranked list, provide:
1. `content_id` (pseudonymized key for lookup)
2. Predicted opportunity score (0–1, higher = stronger fit to pattern)
3. Reason code (primary signal driving the flag)
4. Secondary metrics (word_count, scroll_rate, position, CTR)

### Interpretation Guide for Reviewers

**High score (0.8+) + high word_count + low scroll_rate:**  
→ Long article, visitors aren't scrolling. Consider: (a) title/snippet clearer on value prop, (b) opening paragraph stronger, (c) check if search intent is mismatched.

**High score + position 5-10 + very low CTR (< 0.1%):**  
→ Ranking but nearly invisible. Title/snippet is likely the culprit (misaligned with search intent or weak hook). Rewrite should address search intent clarity.

**Moderate score (0.5-0.7) + trend_direction == "down":**  
→ Page is losing visibility or CTR. Could be stale (old publish date) or sliding rank. Prioritize after high-score items, but include in this week's queue.

**Low score (< 0.3) despite position <= 20:**  
→ Page doesn't fit the "high visibility, low click" pattern. Either CTR is acceptable already, or impressions are too low to trust. Skip for now; re-queue if impressions grow.

### Expected Capacity Impact
- **Baseline rate:** 50 pages/week
- **Recommended top-50 per week:** Focus on predicted score >= 0.50
- **Expected true positives (per Model 3):** ~24 of 50 (48% precision) match the rule. Rest are model errors but may still be worth reviewing (see false positive examples above).

### Continuous Monitoring
Collect reviewer feedback on top-50 recommendations each week:
- Did reviewing the page reveal a low-CTR problem?
- Was there a follow-up (title rewrite, metadata refresh, etc.)?
- Did the action improve CTR over the next window?

This ground truth should feed back into future model training, replacing the rule-based label with observed outcomes.

---

## 7. Reproducibility

### Notebooks & Code
All work lives in the repo under `work/notebooks/`:
- **w01_research_question.ipynb** — Lane choice, capacity analysis, business case
- **w02_ml_task_framing.ipynb** — ML task definition, label design, unit of analysis
- **w03_data_contract.ipynb** — Field definitions, missing value rates, data quality checks
- **w03_feature_leakage_check.ipynb** — Leakage audit, single-feature AUC, excluded fields with justification
- **w04_baseline_score.ipynb** — Hand-coded decision tree baseline, precision@K comparison
- **w04_signal_audit.ipynb** — Signal validation (does position predict CTR? Does word_count predict engagement?) and audit of existing flags
- **w05_model.ipynb** — Logistic regression and Random Forest models, error analysis, Model 2 leakage diagnosis, Model 3 (harder variant) training and validation

### Data Access
- Source: FlyRank ML Internship dataset, Hugging Face (gated, 2-minute request)
- Load via: `pd.read_csv("content_refresh_anonymized.csv")`
- All field definitions in `w03_data_contract.ipynb`

### Dependencies
```
pandas >= 1.3
scikit-learn >= 1.0
numpy >= 1.20
matplotlib / seaborn (for optional viz in notebooks)
```

### Running the Capstone End-to-End
1. Download data from HF (see above)
2. Run notebooks in order: w01 → w02 → w03 → w03_feature_leakage → w04_baseline → w04_signal → w05_model
3. Each notebook produces a `work/outputs/` CSV (final predictions, baseline scores, feature importance)
4. This paper links to specific cells for each claim

---

## 8. Acknowledgments & Data Credit

This work is built on the **FlyRank ML Internship dataset**, a real search-performance snapshot shared for educational and research purposes. FlyRank collects anonymized, aggregated search and engagement data across a portfolio of content properties. Learn more at [https://flyrank.ai](https://flyrank.ai).

**Data Anonymization:** All client names, content URLs, domains, and raw query text have been removed or pseudonymized. Identifiers like `client_*` and `content_*` are opaque hashes; no sensitive information is recoverable from this paper or notebooks.

**Model & Methodology:** Label design, feature engineering, and validation approach are my own; final recommendations should be validated with domain experts (content ops, SEO) before deploying to production.

---

## Appendix: Key Statistics

| Statistic | Value |
|-----------|-------|
| Total rows (after cleaning) | 28,795 |
| Positive label rate | 33.9% |
| Median position | 10.8 |
| Median impressions_90d | 731 |
| Median CTR | 0.07% |
| Median word_count | 4,892 (of 21,077 non-null) |
| Unique clients | 32 |
| Train rows | 22,024 |
| Test rows | 6,771 |
| Baseline precision@50 | 0.60 |
| Model 3 precision@50 | 0.48 |
| Model 3 top feature (word_count importance) | 0.286 |

---

*Last updated: September 2026*  
*Repo: [Your GitHub repo URL] | Paper: this file*  
*Questions? See w01–w05 notebooks for detailed working.*