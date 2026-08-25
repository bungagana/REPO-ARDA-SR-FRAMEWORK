# Real-World Deployment — BPJS/INA-CBG Claim Screening

A real-world deployment of ARDA-SR for automated BPJS Kesehatan / INA-CBG
inpatient-claim screening. This is the "run in production" instance of the ARDA-SR
framework described in the accompanying manuscript.

## Data

The de-identified evaluation dataset is provided as a single **password-protected
archive**:

```
data/deployment_data_public.zip
```

- The archive is **encrypted** (ZIP password). It can be downloaded directly from this
  repository, but **cannot be opened without the password**.
- The password is **not** stored in this repository. To obtain it and the associated
  usage terms, **email the corresponding author** (address in the manuscript).
- Access is granted on a per-request basis for research/reproducibility purposes.

No personal data is included: the released material contains no patient names, IDs,
national identity numbers (NIK), phone numbers, addresses, e-mail addresses, or
medical-record numbers; only coarse age and sex are retained. The institution and the
platform are anonymised (referred to as "Hospital A" / "the platform").
The dataset is subject to on-premise data-residency requirements under Indonesia's
Personal Data Protection Act (UU No. 27/2022).

## Figure

`figures/` contains the deployment figure referenced in the manuscript (aggregate
results only; no patient-level detail).

## Reproducibility

The aggregate metrics and figure are reproducible from the released tables; the
evaluation and plotting scripts live in the root `evaluation/` and `pipeline/` folders.
Data additionally available on request per the terms above.