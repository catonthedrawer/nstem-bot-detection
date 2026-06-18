# Email Security Gateway Bot Detection 

## Project Overview

NSTEM sends email campaigns to over 17,000 subscribers in MailerLite. School and institution security gateways automatically open emails and follow links to scan for threats before a human ever sees the message. This inflates open rates and click rates artificially, corrupting any subsequent engagement analysis on those signals.

This project identifies and flags bot-contaminated subscribers using timestamped activity log data pulled from the MailerLite API.

---

## Pipeline

**1. Subscriber Fetch** ([`subscriber_fetch_v2.ipynb`](subscriber_fetch_v2.ipynb)) - Pulls the full active subscriber list via the MailerLite API and saves to CSV.

**2. Subscriber EDA** ([`subscriber_clean_eda_v2.ipynb`](subscriber_clean_eda_v2.ipynb)) - Cleans the raw subscriber list, computes CTOR, and builds a stratified sample with three engagement tiers for activity log analysis.

**3. Activity Log Fetch** ([`activity_log_fetch_v2.ipynb`](activity_log_fetch_v2.ipynb)) - Pulls full raw activity log data for the sample subscribers via the MailerLite API.

**4. Activity Log EDA & Feature Engineering** ([`activity_log_eda_v2.ipynb`](activity_log_eda_v2.ipynb)) - Cleans the raw activity log, engineers four session-level features, and aggregates to subscriber level.

**5. Modeling** ([`modeling_v2.ipynb`](modeling_v2.ipynb)) - Applies K-Means clustering and rule-based thresholds independently to generate labels, then trains a Random Forest classifier on the labeled sample.

**6. Full-List Deployment** ([`all_activity_log_fetch.py`](all_activity_log_fetch.py), [`pipeline.py`](pipeline.py)) - Pulls activity logs for all eligible subscribers and scores the full list using the trained classifier.

---

## Features

Four features are engineered from timestamped activity log data:

| Feature | Bot Signal |
|---|---|
| `clks_b4_opn_prop` | Clicks recorded before email opens are physically impossible for human recipients |
| `opn_to_first_clk_gap_avg` | Near-zero latency between open and first click is not consistent with human reading behavior |
| `min_clk_sec_gap_avg` | Consecutive clicks under a few seconds apart indicate automated link scanning |
| `click_counts_avg` | Averaging more than a few clicks per session suggests scanning every link rather than selective engagement |

---

## Labeling Approach

No confirmed labels exist, so labels are created:

- **K-Means clustering** groups subscribers by feature patterns
- **Rule-based thresholds** flag subscribers based on known bot behavioral patterns
- Subscribers flagged by both methods receive high-confidence bot labels (Version A)
- Rule-based labels alone form a broader label set (Version B)

A Random Forest classifier trained on these labels is then applied to the full subscriber list using the same timestamp features.

---

## Results

**Version B - Rule-Based Labels (held-out test set):**
- Precision: 1.0000
- Recall: 0.9780
- F1: 0.9889
- Train/test F1 gap: 0.011, no overfitting
- Zero false positives

**Version A - Consensus Labels (held-out test set):**
- Precision: 1.0000
- Recall: 1.0000
- F1: 1.0000
- Zero false positives, zero false negatives

Near-perfect scores reflect internal consistency rather than performance on unseen data. The model rediscovers the threshold logic used to generate its own training labels. Version A's perfect scores are expected given that its bot class contains only the strongest bot behavioral signals.

**Deployment:**
- Precision: 1.0000
- Recall: 0.9889
- F1: 0.9944
- 16% of the full subscriber list flagged as bot-contaminated.

--- 

## Data & Privacy

This project was built on proprietary organizational data. All notebook outputs have been cleared before publishing. Markdown documentation describes analytical reasoning and methodological decisions without referencing specific data values. The code is published as a demonstration of pipeline design and methodology.

---

*Language:* Python

*API:* MailerLite API
