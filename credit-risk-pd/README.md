# Credit default model: calibrated, explained, monitored

A probability-of-default (PD) model for credit-card clients, built the way a bank model team
would structure it: strict **train / validation / test** separation with no data leakage,
**calibrated probabilities**, **SHAP** explanations, a **fairness audit**, and **PSI drift monitoring**.

The point of the project is not the AUC (this dataset caps out around 0.78) but the discipline
around it: every statistic is learned on the right split, every choice is documented, and the
limits are stated.

## Data

*Default of Credit Card Clients*, UCI Machine Learning Repository (Taiwan, 2005):
30,000 clients, 23 features, binary target `default payment next month` (22.1 % defaults).
Original study: I.-C. Yeh and C.-h. Lien, *Expert Systems with Applications* 36(2), 2009.
Download the `.xls` from the UCI page and place it in `data/raw/` (not committed; check the
dataset page for licence and citation terms).

### Cleaning (kept to the strict minimum, done before splitting)

* **35 exact duplicate rows removed** (all columns identical except `ID`): 30,000 -> 29,965 clients.
  They are removed *before* the split so that the same client cannot appear in two sets.
* Nothing else is done to the raw data. Everything that learns from data (scaling, encoding)
  is fitted on the training set only, inside a scikit-learn `Pipeline`.

### Split

60 % / 20 % / 20 %, stratified on the target, fixed seed (`42`): 17,979 / 5,993 / 5,993 clients,
22.1 % defaults in each. `data/splits/MANIFEST.json` stores sizes and file hashes.
`01_make_splits.py` refuses to overwrite existing splits without `--force`.

## How leakage is prevented

| Risk | Control |
|---|---|
| Duplicate clients on both sides of a split | Exact duplicates removed before the split; checked by `tests/test_leakage.py` |
| Scaler / encoder learning from validation or test data | Fitted on train only inside a `Pipeline`; test asserts that its statistics equal the train statistics |
| Row-wise derived features seeing other rows | Features depend on one client only; test asserts that results do not change with the surrounding rows |
| Tuning on the test set | `02`, `03`, `04` never read `test.csv` (enforced by a test); hyper-parameters, early stopping, calibrator and decision threshold are all chosen on **validation** |
| Test set reused after tweaks | `05_final_evaluation.py` runs once and then locks itself (`reports/test_evaluation.json`); `--smoke-test-on-val` lets you debug it on validation instead |
| Bin edges for PSI built from future data | Edges come from the training set (reference) only |
| Using SHAP findings to re-select features | Explanations are descriptive; they are not fed back into the model |

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_make_splits.py` | Deduplicate, stratified 60/20/20 split, manifest with hashes |
| 2 | `02_train.py` | Logistic regression baseline and LightGBM; selection on validation log-loss; calibration (none / sigmoid / isotonic, chosen by out-of-fold log-loss within the validation set); cost-based decision threshold |
| 3 | `03_explain_shap.py` | SHAP summary, importance and one borderline-client waterfall (validation sample) |
| 4 | `04_monitor_drift.py` | Monthly PSI on repayment status, bill amounts and payment amounts |
| 5 | `05_final_evaluation.py` | One-shot test evaluation: AUC/Gini/KS, Brier, log-loss, ECE, bootstrap CIs, reliability diagram, fairness audit |

Run:

```bash
pip install -r requirements.txt
python 01_make_splits.py
pytest -q                                   # leakage guards
python 02_train.py
python 03_explain_shap.py
python 04_monitor_drift.py
python 05_final_evaluation.py --smoke-test-on-val   # optional dry run, test untouched
python 05_final_evaluation.py                       # once, at the very end
```

## Design decisions (and why)

* **`SEX` is not a model feature**, but it is kept aside and **audited**. Dropping a column does not
  remove its proxies, so the fairness report compares groups on `SEX`, `AGE` bands, `MARRIAGE` and
  `EDUCATION`. Metrics per group: observed vs predicted default rate, refusal rate, TPR, FPR,
  approval-rate ratio against the largest group, with a warning for small groups (`n < 200` or
  fewer than 30 defaulters). The "4/5 rule" is shown as an indicative rule of thumb, not a legal test.
* **Rare codes are regrouped with a fixed rule** (`EDUCATION` 0/5/6 -> 4, `MARRIAGE` 0 -> 3), chosen
  without looking at the target.
* **Calibration is selected, not assumed.** The calibrator (none, Platt, isotonic) is picked by
  out-of-fold log-loss inside the validation set. On this data the boosted model was already
  well calibrated, so "none" can win; the reliability diagram on test shows the result.
* **Threshold from business costs, not from 0.5.** The cost ratio (`COST_FN = 5`, `COST_FP = 1`
  in `src/config.py`) is an **assumption**. Under it, the validation-optimal threshold flags a large
  share of clients, which is what a 5:1 cost ratio and a modest AUC imply. Change the costs and
  re-run `02_train.py`; do not read the flag rate as a production policy.
* **The final model is trained on train only** (not refit on train+validation), so that the
  calibrator and threshold stay valid for the model they were fitted on.
* **Validation is used three times** (model selection, calibrator, threshold), so validation
  metrics are slightly optimistic; the test set is the unbiased estimate.

## Known data issues

* **Coding discontinuity in `PAY_0`.** Repayment-status code `1` ("one month late") covers about
  12 % of clients in `PAY_0` (September) but is almost absent from `PAY_2`..`PAY_6`. The PSI of
  `PAY_0` against April is therefore very large (about 0.96) and must **not** be read as real
  drift: the columns are not coded comparably. `04_monitor_drift.py` flags this automatically.
* Undocumented codes: `EDUCATION` 0/5/6 (345 clients), `MARRIAGE` 0 (54 clients), repayment
  statuses -2 and 0. There is no `PAY_1` column (the first one is `PAY_0`).
* Negative bill amounts (about 600-690 per month) are legitimate credit balances and are kept.
* **There is no application date.** Drift is measured **month to month on the same clients**
  (reference: training set, April 2005; monitored: validation set, May-September 2005, with the
  same-month train-vs-validation PSI as a noise floor). This demonstrates the monitoring method;
  it is not a test on new cohorts of customers.

## Results

_Fill in after running `05_final_evaluation.py` once._

| | Test |
|---|---|
| AUC (95 % bootstrap CI) | |
| Gini / KS | |
| Brier / log-loss / ECE | |
| Recall and refusal rate at the validation threshold | |

Figures: `reports/figures/` (`test_performance.png`, `shap_beeswarm.png`, `psi_monthly.png`).
Tables: `reports/fairness_test.csv`, `reports/psi_monthly.csv`, `reports/shap_importance.csv`.

## Limitations

Single dataset from one market and one year, so no claim about other populations or time periods.
AUC is capped by the information in the data. Fairness results on small groups are unstable
(flagged `small_sample`). SHAP describes the model, not causal effects. This is a methodology
showcase, not a production credit policy.

## Tested with

Python 3.13, pandas 3.0, numpy 2.5, scikit-learn 1.9, LightGBM 4.7, SHAP 0.54.
