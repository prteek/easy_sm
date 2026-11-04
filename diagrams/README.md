# easy_sm v2 diagrams

Open the `.excalidraw` files at [excalidraw.com](https://excalidraw.com) (File → Open) or with the Excalidraw VS Code extension. They describe the code on the `v2` branch: steps 0–7b and AWS credentials for local containers.

Colours: green = function, amber diamond = check, red = stops with an error, light green = success, purple = inside the container, blue = AWS, dashed white = your code.

| File | Shows |
|---|---|
| `01-overview` | Where the code runs: your machine, the image, AWS |
| `02-workflow` | One project from `init` to a cloud job, in order |
| `03-calls-setup` | Call map for `init`, `env`, `build` |
| `04-calls-local` | Call map for `train/run --local`, `endpoints serve/invoke --local` |
| `05-calls-cloud` | Call map for `push`, cloud `train` and `run`, `cloud.py` grouped by concern |
| `06-calls-runtime` | Call map inside the image (`python -m runtime ...`) |
| `07-settings` | Loading `easy_sm.toml` and resolving a setting |
| `08-env` | What `easy_sm env` checks and prints |
| `09-init` | Creating a project |
| `10-build` | Building the image and computing its tag |
| `11-what-rebuilds` | Which changes give a new tag and which layers rebuild |
| `12-train-local` | `train --local`, including the failure file |
| `13-run` | `run`, local and cloud, down to `execvp` |
| `14-serve-invoke` | A local endpoint and every response code |
| `15-push` | Pushing the image to ECR |
| `16-cloud-jobs` | Cloud `train` and `run`, from S3 check to exit code |
| `17-job-states` | SageMaker job states as easy_sm reports them |
| `18-endpoints-deploy` | `endpoints deploy` from source to `InService` on the new config |
| `19-calls-endpoints-jobs` | Call map for `endpoints`, `jobs`, `data upload`, `skill` |
| `20-jobs-data` | The checks in `jobs inspect/logs/cancel/artifacts` and `data upload` |
| `21-dry-run` | What `--dry-run` reads and what it never does |
| `22-credentials` | How AWS credentials reach a local container (names in argv, values in the environment) |

## Which command uses which function

| Function | init | env | build | push | train --local | run --local | serve | invoke | train | run |
|---|---|---|---|---|---|---|---|---|---|---|
| `config.load`, `Config.resolve` | | ● | ● | ● | ● | ● | ● | | ● | ● |
| `init.create_project` | ● | | | | | | | | | |
| `image.image_tag` | | ● | | | | | | | | |
| `image.build` | | | ● | ● | ● | ● | ● | | ● | ● |
| `local.local_dirs` | | | | | ● | ● | ● | | | |
| `local.check_script`, `parse_env` | | | | | | ● | | | | ● |
| `local.*_command`, `local.docker` | | | | | ● | ● | ● | | | |
| `local.request_body`, `local.invoke` | | | | | | | | ● | | |
| `cloud.session`, `account_id` | | ● | | ● | | | | | ● | ● |
| `cloud.push` | | | | ● | | | | | ● | ● |
| `cloud.default_bucket`, `ensure_bucket` | | | | | | | | | if input/output unset | |
| `cloud.check_s3_has_data` | | | | | | | | | ● | |
| `cloud.parse_timeout`, `job_name`, `start`, `follow` | | | | | | | | | ● | ● |
| `cloud.training_job_request` | | | | | | | | | ● | |
| `cloud.processing_job_request` | | | | | | | | | | ● |
| runtime `run_training` | | | | | ● | | | | ● | |
| runtime `run_command` | | | | | | ● | | | | ● |
| runtime `create`, `make_app` | | | | | | | ● | answers it | | |
| `cloud.container_credentials` | | | | | ● | ● | ● | | | |

### Endpoints, jobs and data

| Function | deploy | invoke | ls | describe | delete | jobs ls | inspect | logs | cancel | artifacts | data upload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `latest_job`, `pick_latest` | `--latest` | | | | | | | | | | |
| `is_own` | `--latest` | | | | | ● | | | | | |
| `job_source` | `--latest`, `--job` | | | | | | | | | | |
| `resource_name`, `model_request`, `endpoint_config_request` | ● | | | | | | | | | | |
| `deploy`, `wait_for_endpoint`, `deploy_outcome` | ● | | | | | | | | | | |
| `find_endpoint` | ● | | | | | | | | | | |
| `invoke_endpoint` | | cloud | | | | | | | | | |
| `list_endpoints` | | | ● | | | | | | | | |
| `endpoint_details`, `describe_lines` | | | | ● | | | | | | | |
| `delete_endpoint` | | | | | ● | | | | | | |
| `list_jobs`, `job_rows`, `fmt_duration` | | | | | | ● | | | | | |
| `find_job` | | | | | | | ● | ● | ● | ● | |
| `inspect_lines` | | | | | | | ● | | | | |
| `new_log_lines`, `follow` | | | | | | | | ● | | | |
| `stop_job` | | | | | | | | | ● | | |
| `upload_plan`, `upload` | | | | | | | | | | | ● |

## Where files are, in each place

| What | On your machine (`--local`) | In the container | In the cloud |
|---|---|---|---|
| Training data | `data/` (read-only mount) | `/opt/ml/input/data/training` | S3 `train.input`, copied in as channel `training` |
| Model | `.easy_sm/model/` | `/opt/ml/model` | `train.output/<job>/output/model.tar.gz` |
| Training failure | `.easy_sm/output/failure` | `/opt/ml/output/failure` | the job's `FailureReason` |
| `run` input | `data/` (read-only mount) | `/opt/ml/processing/input` | S3 `run.input`, if set |
| `run` output | `.easy_sm/output/` | `/opt/ml/processing/output` | S3 `run.output`, if set (uploaded at the end) |
| Your code | `src/` (baked into the image) | `/opt/app/src` | same image, pulled from ECR |
| easy_sm runtime | package `easy_sm/runtime/` (baked in) | `/opt/app/runtime` | same image |

## Pure functions and functions with side effects

Pure functions take inputs and return a value; they are what `tests/` checks, without mocks. Functions with side effects are kept thin and were checked by running them for real.

| File | Pure (tested) | Side effects (thin) |
|---|---|---|
| `config.py` | `_flatten`, `check_app_name`, `Config.resolve`, `Config.get` | `load` (reads the file) |
| `init.py` | `app_name_from_folder` | `scaffold_files`, `create_project` (read and write files) |
| `image.py` | `dockerfile`, `image_hash`, `build_command` | `python_version`, `image_files`, `_folder`, `write_context` (files); `build`, `exists_locally`, `_run`, `_succeeds` (uv, docker) |
| `local.py` | `docker_run`, `train_command`, `run_command`, `serve_command`, `parse_env`, `local_paths` | `local_dirs`, `empty`, `check_script`, `request_body` (files); `docker`, `invoke` (docker, HTTP) |
| `cloud.py` | `session_kwargs`, `default_bucket`, `ecr_uri`, `job_name`, `parse_timeout`, `split_s3`, `training_job_request`, `processing_job_request`, `host_of`, `check_endpoint_name`, `is_own`, `pick_latest`, `resource_name`, `model_request`, `endpoint_config_request`, `deploy_outcome`, `image_hash_of`, `describe_lines`, `fmt_duration`, `job_rows`, `inspect_lines`, `upload_plan`, `credential_env` | `session`, `account_id`, `push`, `in_ecr`, `_docker`, `check_s3_has_data`, `ensure_bucket`, `describe`, `new_log_lines`, `follow`, `start`, `latest_job`, `job_source`, `find_endpoint`, `deploy`, `wait_for_endpoint`, `invoke_endpoint`, `list_endpoints`, `endpoint_details`, `delete_endpoint`, `list_jobs`, `find_job`, `stop_job`, `upload`, `container_credentials` (AWS, docker) |
| `runtime/` | `build_command`, `make_app`, `decode`, `respond`, `gunicorn_command` | `load_from`, `run_training`, `create`, `serve`, `run_command`, `main` (imports, files, exec) |

## Exit codes

| Code | Means | Example |
|---|---|---|
| 0 | Done | train finished; build reused an image; serve stopped with Ctrl-C |
| 1 | A check or the job failed | unknown setting, missing script, no data in S3, job Failed |
| 2 | Wrong way to call the command | cloud-only option with `--local`; serve without `--local` |
| 130 | Ctrl-C while following a cloud job | the job keeps running on SageMaker |
| N | The command's own exit code, passed through | `run --local clean.py` ending in `sys.exit(3)` exits 3 |

stdout carries only the result (a tag, a path, a job name, a response); progress, logs and errors go to stderr.

## Regenerating

The diagrams are generated from `make_diagrams.py` (needs Graphviz: `brew install graphviz`). After changing the code, update the diagram definitions in that script and run:

```bash
python3 diagrams/make_diagrams.py diagrams
```
