# DSN AI Bootcamp Qualification Hackathon 2026 — ML Track

Predicting total sales for a product at a given store, for DSN Mart's chain of stores across Nigeria.

**Competition:** [DSN Bootcamp Qualification Hackathon 2026, ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)
**Task:** Regression — predict `total_sales` for each product-store combination in `test.csv`
**Metric:** Root Mean Squared Error (RMSE)
**Current best CV RMSE:** ~1158.67 (best real leaderboard score so far: 1148.73)

---

## Why this is a regression problem, not classification

`total_sales` is a continuous quantity — it can take any value in a range (₦32.70 to ₦12,996.82 in the training data), not one of a fixed set of labels. There's no natural way to bucket it without throwing away information that matters for the actual business use case (stock planning, pricing), so this is squarely a regression task, scored on RMSE.

## What this task is actually for

DSN Mart wants to predict how much of a given product will sell at a given store, using only static information about the product (category, price, weight) and the outlet (format, size, location tier, age). This feeds real decisions: stock allocation, pricing, and store investment.

---

## Data

| File | Rows | Description |
|---|---|---|
| `train.csv` | 6,818 | Product-store rows with known `total_sales` |
| `test.csv` | 1,705 | Product-store rows to predict |
| `sample_submission.csv` | 1,705 | Exact submission format: `id, total_sales` |

**Columns:** `product_code`, `product_weight_kg`, `fat_content`, `shelf_visibility`, `product_category`, `product_price`, `store_code`, `store_age_years`, `store_size`, `store_location_tier`, `store_format`, plus `total_sales` (target, train only).

No datetime column exists in this dataset — it's a static, cross-sectional snapshot, not a time series. There is nothing to compute a moving average over, and no missing dates to check for, because there is no date field at all.

---

## Data quality issues found and fixed

Every fix below was diagnosed with a query **before** being applied — the notebook shows the "broken" state, then the fix, then verification, so the reasoning is auditable rather than asserted.

| Issue | Diagnosis | Fix |
|---|---|---|
| `product_category` casing | 48 "categories" collapsed to 16 real ones once case-normalized (`"fruits and vegetables"` / `"Fruits and Vegetables"` / `"FRUITS AND VEGETABLES"` were being treated as 3 different categories) | `.str.strip().str.title()` |
| `fat_content` on non-food items | `Household` and `Health And Hygiene` products were labeled `Low Fat`/`Regular` — meaningless for soap | Relabeled as `Non-Edible` |
| `store_size` missing (51% of Corner Shop, 33% of Standard Supermarket) | Missingness is structural, not random, tied to `store_format` | Corner Shop imputed as `Small` (100% of known values agree); Standard Supermarket marked `Unknown` (no dominant size — guessing would mislead the model) |
| `product_weight_kg` missing (~18%) | 1,223 of 1,225 missing weights are recoverable by looking up the same product elsewhere in the data | Product-code lookup first, category-median fallback for the remaining 2 |
| `shelf_visibility == 0` (422 rows) | Sales for `visibility == 0` rows are statistically indistinguishable from `visibility > 0` rows — a real 0% visibility should suppress sales, so this is a data placeholder, not a real measurement | Flagged with a `visibility_was_zero` indicator, then replaced with the category's median non-zero visibility |

---

## Feature engineering

- **`item_type`** — 16 categories grouped into Food / Drinks / Non-Consumable
- **Store-level and product-level aggregates** (`store_avg_sales`, `store_avg_price`, `product_avg_sales`, `product_avg_price`) — computed **fold-safe** (only from the current fold's training rows, never leaking a row's own target into its own feature) for cross-validation, and from the full training set for the final model
- **Smoothed (regularized) target encoding** for these aggregates — `product_code` averages only ~4.4 rows each, so a plain mean is noisy for low-count products. Smoothing blends the group mean toward the global mean, weighted by how much data that group actually has:

  ```
  smoothed_mean = (count * group_mean + smoothing * global_mean) / (count + smoothing)
  ```

  Swept smoothing strength from 1 to 1000; performance plateaus around 150-500. Used **smoothing=150**.

All train/test overlap was checked before relying on these features: all 10 test stores appear in train (zero leakage risk for store aggregates); 1,078 of 1,082 test products appear in train (4 unseen products fall back to the global mean).

---

## Target transformation — tested, not assumed

Skewness of raw `total_sales` is +1.15 (right-skewed). The "standard" fix is a log-transform, so that was tried — but it **overcorrects** (skew flips to -0.90, same magnitude, wrong direction). A direct RMSE comparison with the actual model settled it:

| Transform | CV RMSE |
|---|---|
| Raw (no transform) | **1177.94** (best) |
| sqrt | 1194.73 |
| log1p | 1240.61 |

**Lesson:** skewness-reduction heuristics are for linear models (which assume roughly normal residuals). Gradient boosted trees don't make that assumption, and transforming the target for a tree model distorts the loss it's actually optimizing relative to the metric we're scored on. Always test this empirically per model family rather than applying a blanket rule.

---

## Models tried

| Model | CV RMSE | Notes |
|---|---|---|
| Naive baseline (predict the mean) | 1697.72 | Floor — every real model must beat this |
| Linear Regression | 1285.00 | One-hot encoded low-cardinality categoricals; `store_code`/`product_code` excluded (too high-cardinality for one-hot), signal carried through aggregate features instead |
| XGBoost (manual hyperparameters, sqrt target) | 1261.95 | First attempt overfit badly (train RMSE 670 vs val RMSE 1340) until constrained with early stopping + `min_child_samples` |
| LightGBM (tuned) | 1264.92 | |
| XGBoost (20-combination random search, sqrt target) | 1194.73 | Search consistently favored shallow trees + regularization over the manual guess |
| CatBoost (tuned, 12-combination search) | 1222.92 | Tested fairly against XGBoost's tuning effort; still lost |
| XGBoost (tuned, raw target) | 1177.94 | Raw target beat sqrt/log after direct comparison |
| **XGBoost (tuned, raw target, smoothed encoding)** | **1158.67** | **Current best** |
| Stacking (XGBoost + LightGBM + CatBoost) | No improvement | Optimal blend weight search converged to 100% XGBoost, 0% others |
| Neural network (embeddings for store/product + MLP) | 1210.00 | Underperformed — dataset is too small (~5,450 rows/fold) for embeddings to beat mean-based aggregates |
| Tweedie loss objective | 1161.26 (best variance power) | No improvement over squared error — likely because this data isn't zero-inflated (min sales = 32.70) |

### Hyperparameters (final model)
```
max_depth=3, min_child_weight=10, subsample=1.0, colsample_bytree=0.7,
reg_alpha=1.0, reg_lambda=5.0, learning_rate=0.05, n_estimators≈55-72 (via early stopping)
```

---

## Encoding

Explicit `OrdinalEncoder`-based label encoding was tested against pandas' native `category` dtype (which XGBoost/LightGBM read directly). Result: 1179.06 vs 1177.94 — no meaningful difference for this feature set's cardinality. Native categorical handling was kept for simplicity.

---

## What didn't work (and why that's still useful to know)

Feature/technique ideas that were tested and **rejected** after honest evaluation:

- **Relative pricing features** (`price / product's average price elsewhere`) — no improvement (1261.82 vs 1261.95). Diagnosis: product prices are nearly identical across all stores (std of the ratio = 0.019), so there's no real relative-pricing signal to extract.
- **Store assortment size** (number of distinct products carried) — not tested in the model directly after diagnosis showed it's almost perfectly collinear with existing store identity features.
- **Group standard deviation / variability features** — hurt performance (1169.08 vs 1158.67). `product_code` groups average only ~4.4 rows, so a standard deviation computed from so few points is itself too noisy to be useful.
- **Explicit interaction categorical** (`store_format × product_category`) — hurt performance (1189.19 vs 1158.67). Trees already discover interactions through sequential splits; forcing a pre-combined high-cardinality category just fragments the data.
- **Segmented/specialized model for the highest-variance store** — see below. Made things worse, not better.

### The `STORE-7WS` finding

Out-of-fold residual analysis showed `STORE-7WS` (11% of rows) accounts for **21% of total squared error**. This store is the **only** `Flagship Hypermarket` in the dataset — a 1-to-1 mapping between store identity and format, so the model has no second example of that format to generalize from.

A targeted fix — training a specialized model just for this store's 748 rows — was tested and made things **worse** (RMSE on that store alone: 1594 → 1785), because it lost the benefit of pooling information across all stores.

Further investigation checked whether this store's extreme top-sellers follow any pattern in price, category, or visibility. They don't: the amplification these products get at this store (median ~1.83-2.08x what the same product sells elsewhere) matches the store's overall sales ratio almost exactly — meaning the model already captures the systematic effect via `store_avg_sales`. The *remaining* variance (individual products ranging 0.15x to 11x that baseline) shows no correlation with any available feature (price: -0.03, visibility: 0.06). This strongly suggests the true driver is something outside this dataset entirely (a promotion, a local event, a stockout elsewhere) — not a modeling gap.

- **Seed averaging** (5 different random seeds, averaged) — no improvement (1169.65 vs single best seed's 1159.27).

---

## Reproducibility

The notebook auto-detects the data location (works in this sandbox, on Kaggle, or with a local `./data` folder) — no hardcoded absolute paths. See `notebooks/dsn_sales_prediction.ipynb`, Step 1.

### To run:
```bash
pip install -r requirements.txt
# place train.csv, test.csv, sample_submission.csv in ./data/
jupyter nbconvert --to notebook --execute --inplace notebooks/dsn_sales_prediction.ipynb
```

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── dsn_sales_prediction.ipynb    # full pipeline: EDA -> cleaning -> features -> modeling -> predictions
├── submissions/
│   └── submission_v4.csv             # best submission (CV RMSE ~1158.67)
└── data/                             # not included - place train.csv, test.csv, sample_submission.csv here
```

---

## Honest limitations and next steps

- Public leaderboard is only ~50% of test data; final ranking uses the other half. Some top public scores are anomalously far ahead of the rest of the field (13x better than the next tier) and are more likely explained by leaderboard-specific overfitting or a data characteristic not yet identified than by conventional modeling.
- A wider, Bayesian hyperparameter search (Optuna) beyond the random search used here has not been tried and could still yield a small improvement.
- The `STORE-7WS` error concentration suggests any further meaningful gain likely requires information not present in the given files, rather than better feature engineering on what we have.
