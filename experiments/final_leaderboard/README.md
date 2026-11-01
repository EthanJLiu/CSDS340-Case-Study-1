# Final leaderboard preparation

Decision: retain `classifySpam.py` byte-for-byte. The new evaluator already
concatenates `spamTrain1.csv` and `spamTrain2.csv` and passes all 3,000 training
rows to `predictTest`. The classifier must continue to learn only from those
arguments and return continuous scores. No CSV loading or cutoff is needed
inside the submission.

The supplied second training CSV and final evaluator were copied unchanged to
the repository root. The actual `spamTest.csv` is still unavailable and was not
created in the project. `baseline_classifier.py` preserves the submitted model.

## Reproduction and protocol

- Training on train1 and scoring train2 reproduced the teacher's numbers exactly:
  AUC **0.8931869397990477**; TPR at FPR=0.01 **0.43455497382198954**.
- Both releases contain 1,500 emails with 30 features. Missing values (`-1`)
  account for approximately 20% of measurements. No exact duplicate feature
  rows were found within or between the releases.
- Previous work had already examined all train1 rows. We reserved 500 stratified
  train2 rows (seed 20261031) before comparing new candidates. The remaining
  2,500 rows formed development data. The original aggregate train2 leaderboard
  result was already known, so this is not a completely blind external test.
- Development used two repetitions of five-fold stratified cross-validation,
  seeds 340 and 341. Each fit used 2,000 training rows and 500 validation rows.
  All transforms were fit on each fold's training rows only.
- Metrics below are means of fold metrics, not pooled out-of-fold scores.
  Splits reuse the same emails; folds are not independent replicates.
- Only one candidate was selected for confirmation. Its recipe and input hashes
  were saved in `frozen_candidate.json` before computing confirmation metrics.
  The predeclared gate retained the baseline if either metric declined.

## Development comparisons

Each entry gives mean AUC and TPR at FPR=0.01:

- Original 50/25/25 ensemble: **0.917844; 51.5572%**.
- Scale all C parameters by 1500 / training rows: **0.917818; 51.6572%**.
- Halve interaction logistic C: **0.917986; 51.8572%**.
- Double interaction logistic C: **0.917441; 51.5052%**.
- Average bin models with 2, 3, and 4 positive-frequency bins:
  **0.919026; 52.2062%**.
- Use the positive-frequency median as the bin boundary:
  **0.915450; 51.7550%**.
- Average five models fit on stratified 80% training subsamples:
  **0.918111; 51.4082%**.

Bin averaging was the strongest candidate. It improved AUC in 8 of 10 folds,
but TPR in only 4 of 10 (one tie). No additional blends or combined parameter
changes were tried after observing these results.

## One confirmation comparison

Both models were fit on the same 2,500 development emails and scored on the same
500 reserved emails: 309 ham and 191 spam.

- Original: **0.884664 AUC; 27.7487% TPR**.
- Bin averaging: **0.886765 AUC; 26.1780% TPR**.
- Candidate minus original: **+0.002101 AUC; -1.5707 percentage points TPR**.

The candidate failed the gate, so the original submission was retained. No
runner-up was then tested against these confirmation labels. The two-sided 95%
paired stratified bootstrap intervals for the differences were [-0.000864,
0.005337] AUC and [-3.6649, 5.2356] percentage points TPR. These intervals describe
resampling the fixed predictions; they do not account for selection, refitting,
or distribution shifts. At 1% FPR, only about three ham emails determine the
operating point in this partition, so TPR is particularly unstable.

These scores are internal checks on released data, not estimates of the final
leaderboard rank. The 500-row values are not directly comparable with the
teacher's 1,500-row result. The confirmation set was used for the retention
decision and should not be reused for more tuning.

## Reproduce

From the repository root:

```bash
python3 experiments/final_leaderboard/regularization_experiments.py
python3 experiments/final_leaderboard/stability_experiments.py
python3 experiments/final_leaderboard/confirm_candidate.py
```

The stability script resumes existing caches. Its caches are valid for this
frozen run only: do not reuse them after changing the data, split plan, or model
code. Confirmation checks the frozen input hashes. Its rerun reproduces the
same comparison; it does not supply fresh validation evidence. Exact numeric
reproduction was obtained with Python 3.12.10 and scikit-learn 1.9.0.

Per-fold predictions, metrics, the split plan, and the confirmation decision are
saved alongside the scripts. `verification_results.json` records the submission
interface checks. The final evaluator was also executed unchanged in a temporary
directory using the existing internal confirmation split; it reproduced the
baseline confirmation metrics exactly. This is an integration check, not a new
test-set result.

Once the instructor releases the actual final test CSV, place it beside the
training files and run:

```bash
python3 evaluateClassifierTest.py
```

That evaluator trains on both complete releases automatically. The 500-row
reservation belongs only to these experiments; it does not reduce final training.
