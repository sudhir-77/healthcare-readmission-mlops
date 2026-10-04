# Medication change and 30-day readmission

This is an observational analysis. It is not evidence that changing a diabetes medication causes a change in readmission. The estimate is an association that would equal an average treatment effect only if the assumptions below hold.

## Estimand

The treatment is the dataset field `change` (Ch versus No): whether any diabetes medication dose changed during the encounter. The outcome is readmission within 30 days. Each encounter is one row. Patients in the test split are not used here, so the same held-out patients stay untouched by both the predictive model and this analysis.

## Estimates

- Naive risk difference (treated minus untreated): 0.010865179048724097
- IPW (Hajek) risk difference: 0.007092148228606043
- IPW 95% patient-cluster bootstrap interval: [0.002062662329659777, 0.011755002774594741]
- AIPW risk difference: 0.008060354639866983
- AIPW 95% patient-cluster bootstrap interval: [0.004128876976854863, 0.012314887843538114]
- Analysis encounters: 84271
- Treated encounters: 39081
- Propensity range before clipping: 0.08381460094774339 to 0.9952969376520868
- Share of propensities inside [0.1, 0.9]: 0.9998101363458366
- Propensity clipping used for weights: 0.05
- Logistic penalty C chosen for balance: 1.0
- Square terms added to the propensity model: False
- Max absolute SMD before weighting: 0.22599113999404483
- Max absolute SMD after weighting: 0.02857282477751592

Positive values mean a higher readmission rate in the medication-change group. Intervals are percentile intervals from a patient-cluster bootstrap of the fixed IPW and AIPW scores. They do not include uncertainty from choosing the propensity specification.

## Balance and overlap

Standardized mean differences use inverse-probability weights and the unweighted pooled standard deviation (Austin's definition). The success rule is that every pre-specified covariate has absolute weighted SMD below 0.1. The penalty and the propensity clip are chosen to reduce that maximum. The outcome is not used in that choice. The covariate table is `results/smd_table.csv`. Overlap is plotted in `results/propensity_overlap.png`.

## Assumptions

1. Conditional exchangeability: given the pre-treatment covariates, medication change is independent of the potential readmission outcomes. Unmeasured severity can violate this.
2. Positivity: both decisions have non-zero probability at every covariate pattern we keep. Extreme scores are clipped, which changes the estimand slightly toward the trimmed population.
3. Consistency: the recorded change flag matches the treatment contrast we describe.
4. No interference across encounters, and the encounter order in `encounter_id` is a usable time order.
5. Discharge disposition, length of stay, and in-hospital medication counts are excluded from the covariate set because the medication decision can affect them.

A1C result group is included as a severity marker. The lab can be drawn during the stay, so it is not a pure pre-treatment covariate. That is a limitation.

## What this does not say

Do not read the point estimate as proof that a medication-change program would reduce or increase readmissions. A randomized or stepped-wedge evaluation would be required for that claim. The predictive model is a separate tool and was not used to choose this specification.
