---
title: Summary
updated: 2026-11-04
---

- **Released:** easy_sm 2.0.0 (the hf-style rewrite) and 2.0.1 (adds `easy_sm dockerfile`) are on `main`, PyPI and the Pages site.
- **Not yet checked by the user:** Claude ran every command against Docker and the personal AWS account; your own run of the README examples is still open.
- **Left in the personal AWS account by testing (2026-11-04):** ECR repo `churn` with test images, `s3://sagemaker-eu-west-1-<account>/churn/` (data, models, run output), churn models and endpoint configs, finished or stopped test jobs. All test endpoints and the temporary IAM role were deleted. Unrelated endpoints `mpg` and `mpgr` have been `InService` since 2024-12 and may be billing.
- **Credentials caveat:** the `personal` profile uses long-lived IAM keys, which `--local` containers receive (one set, read-only). SSO profiles would pass short-lived ones (*decisions.md, Security*).
