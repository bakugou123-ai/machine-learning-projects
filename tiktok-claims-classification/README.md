# TikTok Claims vs. Opinions Classification

**Notebook:** [`Tiktok_Claims_Classification_Project.ipynb`](./Tiktok_Claims_Classification_Project.ipynb)
**Tools:** Python · pandas · scikit-learn · XGBoost · NLP (`CountVectorizer`)

## Objective

Build a content-moderation model that classifies whether a reported TikTok video makes a
factual **claim** (which may violate platform terms of service) or states an **opinion**,
helping prioritize which videos most need human moderator review. Followed the PACE
(Plan → Analyze → Construct → Evaluate) workflow.

## Approach

1. **EDA & statistical testing** — explored the dataset (video metadata + transcription text),
   including a two-sample t-test comparing engagement metrics (e.g. like counts) between
   verified and unverified accounts.
2. **Feature engineering** — vectorized `video_transcription_text` with `CountVectorizer`
   (bag-of-words) and combined it with numeric engagement features (view/like/share/download/
   comment counts) and one-hot encoded categorical fields.
3. **Modeling** — tuned Random Forest and XGBoost classifiers with `GridSearchCV` (5-fold CV,
   scored on accuracy/precision/recall/F1), refitting on recall to prioritize catching
   potential claims.
4. **Champion model** — the tuned Random Forest achieved **~99% cross-validated recall** and
   **~99.9% precision**. On the held-out test set it reached ~99% overall accuracy (1,925/1,928
   opinions and 1,873/1,889 claims correctly classified).

## Skills Demonstrated

NLP feature extraction (bag-of-words), combining text and numeric features, hypothesis
testing, hyperparameter tuning, model comparison/selection, and precision/recall trade-off
analysis for a content-moderation use case.
