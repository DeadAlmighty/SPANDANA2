# SPANDANA label limitations (PTB-XL)

- LVH-positive = SCP statement `LVH` with likelihood >= 100. The SCP likelihood reflects annotator confidence in the ECG statement, not an echocardiographic diagnosis.
- PTB-XL labels are cardiologist ECG-statement annotations. Studies have shown only moderate agreement between ECG-derived and echo-derived LVH; any model trained here learns to reproduce ECG annotations, not true anatomy.
- 1047 low-certainty LVH statements (<100) are EXCLUDED from both classes rather than guessed.
- The control group (19667 recordings) is defined as *no LVH statement* — 96.3% of them still carry other cardiac statements. Control is NOT healthy.
- Single recordings per patient may share patients across... none: the official strat_fold is patient-wise, so no patient leaks across splits (verified by create_splits.py).
