# Resume notes

## Draft bullets

- Built a 30-day readmission model on 99343 at-risk encounters from 101766 inpatient records, with SQL features in DuckDB and calibrated LightGBM, reaching 0.681 AUROC on held-out patients (0.005 ECE).
- Served the model through a FastAPI endpoint at 6.7 ms p95 latency over 500 local single-record requests, with the model version stored in an MLflow registry.
- Estimated the effect of medication changes on readmission with IPW (0.71 pts) and AIPW (0.81 pts, 95% CI 0.41 to 1.23), checking overlap and covariate balance (max SMD 0.029).
- Compared the boosted model with an L2 logistic regression baseline (0.662 test AUROC) and kept every test patient out of fitting, calibration, and model selection.

## Skills and tools

Python, SQL, DuckDB, pandas, scikit-learn, LightGBM, MLflow, FastAPI, Docker, SHAP, pytest, propensity scores, IPW, AIPW

## Project summary

This project trains a 30-day readmission classifier on the UCI Diabetes 130-US Hospitals encounters (101766 rows) and serves the calibrated scores from a local FastAPI app. It also estimates the observational association between an in-hospital diabetes medication change and readmission with propensity-score weighting.

## GitHub description

30-day hospital readmission scoring with DuckDB features, calibrated LightGBM, a FastAPI service, and a propensity-score effect estimate.
