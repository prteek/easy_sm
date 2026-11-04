---
title: Summary
updated: 2026-11-04
---

Only what tools can't tell you. For version, open PRs and CI, run `git tag`, `gh pr list` and `gh run list`.

- **Work in progress:** easy_sm 2 is a complete rewrite on branch `v2` (steps 0–7b done). It is **local only, not pushed**, and `main` still holds 1.4.0. Remaining: step 8 (*v2 rewrite*).
- **Not yet checked by the user:** every v2 command was run against Docker and the personal AWS account by Claude; the user's own end-to-end run is still open (*v2 rewrite*, `@me`).
- **Left in the personal AWS account by testing (2026-11-04):** ECR repo `churn` with test images, `s3://sagemaker-eu-west-1-<account>/churn/` (data, models), churn models and endpoint configs, and finished or stopped test jobs. All test endpoints were deleted. Unrelated endpoints `mpg` and `mpgr` have been `InService` since 2024-12 and may be billing.
- **Credentials caveat:** the `personal` profile uses long-lived IAM keys, which `--local` containers now receive (one set, read-only). SSO profiles would pass short-lived ones (*decisions.md, Security*).
- **Stale published docs:** the 1.x GitHub Pages site still describes 1.x; unpublish it after `v2` is merged (*v2 rewrite*, `@me`).
