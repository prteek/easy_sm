---
title: Endpoint updates delete the config SageMaker rolls back to
updated: 2026-10-15
---

# Issue

`deploy` and `deploy-serverless` on an existing endpoint can leave it pointing at an endpoint config that no longer exists. The endpoint keeps serving, but every later deploy fails with:

```
ClientError: An error occurred (ValidationException) when calling the UpdateEndpoint operation:
Could not find endpoint configuration "<endpoint>-model-<hash>-config".
```

## Cause

`SageMakerClient._create_or_update_endpoint` (`easy_sm/sagemaker/sagemaker.py`) does this on an existing endpoint:

1. Reads the endpoint's current config name (`previous_config_name`).
2. Calls `update_endpoint` with the new config.
3. Immediately deletes `previous_config_name`, ignoring `ClientError`.

`update_endpoint` returns as soon as the update **starts**; the rollout runs for several minutes afterwards. The comment on the delete assumes SageMaker refuses to delete a config that is still in use. It doesn't: the delete succeeds, because a running endpoint keeps its own copy of the config.

If the new model then fails to start, SageMaker rolls the endpoint back to `previous_config_name`, which easy_sm has already deleted. The endpoint is `InService` on the old model, `describe-endpoint` still names the deleted config, and SageMaker refuses any further `UpdateEndpoint` until a config with that name exists again.

## What happened (2026-10-15)

A serverless endpoint was updated to a new image that crashed on start-up (`python: command not found`: the image's venv Python was under `/root`, which serverless inference can't read because it doesn't run the container as root).

| Time | Event |
|---|---|
| 16:57:44 | easy_sm creates the new config and calls `update_endpoint` |
| 16:57:45 | easy_sm deletes the previous config (CloudTrail `DeleteEndpointConfig`) |
| ~17:04 | New containers crash; SageMaker rolls back to the deleted config. Endpoint `InService` on the old model, `FailureReason` set |
| Later | Next `deploy-serverless` fails with the `ValidationException` above |

CloudTrail also shows four configs for the same endpoint deleted within 15 minutes during redeploys in August; whether any of those were rollbacks wasn't checked.

Two things made it worse:

- **The command reported success.** easy_sm returned the endpoint name, so CI went on to the smoke test, which ran against the old model still serving during the rollout and passed.
- **The recovery path was gone.** Rolling back by pointing the endpoint at the previous config was impossible, because that config had been deleted.

**Workaround used:** CloudTrail's `CreateEndpointConfig` event had the original settings, and the model it referenced still existed, so the config was recreated under the same name with `aws sagemaker create-endpoint-config`. `UpdateEndpoint` then worked again.

# Potential fix

Wait for the update to finish, confirm it took, and only then clean up:

```python
if self._check_endpoint_exists(endpoint_name):
    previous_config_name = self.sagemaker_client.describe_endpoint(
        EndpointName=endpoint_name
    )["EndpointConfigName"]
    self.sagemaker_client.update_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    self.sagemaker_client.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    current = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    if current["EndpointConfigName"] != endpoint_config_name:
        raise RuntimeError(
            f"Update of '{endpoint_name}' to '{endpoint_config_name}' failed and rolled back to "
            f"'{current['EndpointConfigName']}': {current.get('FailureReason', 'no reason given')}"
        )
    if previous_config_name != endpoint_config_name:
        self.sagemaker_client.delete_endpoint_config(EndpointConfigName=previous_config_name)
```

- **Wait, then check the config name, not just the status.** A failed update also ends in `InService`, after the rollback. Whether the endpoint's current config is the new one is what says whether the update worked.
- **Fail loudly.** A non-zero exit on rollback stops CI before the smoke test, and `FailureReason` points at the CloudWatch logs.
- **Delete only after success.** On failure the previous config is kept, so the endpoint stays updatable. Optionally keep the previous config even on success, as a one-step rollback target.
- **Clear error for endpoints already in this state.** Before `update_endpoint`, if `describe_endpoint_config(previous_config_name)` raises, say so and point at the fix (recreate the config, or delete and recreate the endpoint) instead of surfacing the raw `ValidationException`.
- **Same change for `deploy`.** Provisioned endpoints use the same `_create_or_update_endpoint`.
- **`--no-wait` to opt out.** This matches the existing decision for `train`, `process` and `run` (wait by default, `0eba322`). With `--no-wait`, skip the cleanup and leave old configs to `delete-endpoint --delete-config`.

The waiter's defaults (30 s delay, 120 attempts) cover up to an hour, which is enough for serverless and most provisioned updates.

### Related: model names aren't stable

`_create_model` names models `model-{hash(s3_model_location)}`. Python randomises `hash()` for strings per process, so the same `model.tar.gz` gets a different, sometimes negative, name every run. That is what keeps redeploys from colliding (`62c900e`), but a config name says nothing about which model it holds, which makes rollback targets hard to identify. A name derived from the job or a deterministic digest (e.g. `hashlib.sha1(s3_model_location.encode()).hexdigest()[:12]`, plus a timestamp to stay unique) would be readable and still collision-free.

# Recommendation: should updating be a separate command?

**No. Keep `deploy` / `deploy-serverless` as create-or-update, and make the update path safe as above.**

- **Create-or-update is what callers want.** CI and scripts run one command whether or not the endpoint exists. A separate `update-endpoint` would make every caller check existence and branch, which goes against the composable, minimal-flags design.
- **Splitting doesn't fix the bug.** The bug is deleting before the rollout finishes. A separate update command would need exactly the same wait, verify and cleanup logic.
- **Both paths do the same thing.** Each creates a model and a config, then points the endpoint at it. Only the last call differs (`create_endpoint` vs `update_endpoint`).

**What does justify a separate command is pointing an endpoint at an existing config**, i.e. rollback, e.g.:

```bash
easy_sm update-endpoint -n my-endpoint --config my-endpoint-model-abc123-config
```

This is a different operation: no new model, no new config, no image. It is only useful once old configs are kept (see the fix above). Consider adding it after the fix, together with listing an endpoint's previous configs so the rollback target can be found.

# Suggested order

1. Fix `_create_or_update_endpoint`: wait, verify, delete only on success, `--no-wait` opt-out (tests: success, rollback raises and keeps the old config, `--no-wait` skips cleanup).
2. Clear error when the endpoint's current config is missing.
3. Deterministic model names.
4. Optional: `update-endpoint` for rollback, keeping the last N configs.
