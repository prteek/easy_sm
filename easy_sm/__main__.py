"""The easy_sm command line."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from botocore.exceptions import BotoCoreError, ClientError

from easy_sm import cloud, config, image, local
from easy_sm.cloud import CloudError
from easy_sm.config import CONFIG_FILE, SETTINGS, Config, ConfigError
from easy_sm.image import ImageError
from easy_sm.init import create_project

app = typer.Typer(
    help="Prototype SageMaker training, jobs and endpoints locally, then run the same commands in the cloud.",
    no_args_is_help=True,
)
endpoints = typer.Typer(help="Serve, call and manage endpoints.", no_args_is_help=True)
app.add_typer(endpoints, name="endpoints")
jobs = typer.Typer(help="List, inspect, follow and cancel training and processing jobs.", no_args_is_help=True)
app.add_typer(jobs, name="jobs")
data = typer.Typer(help="Move data to S3.", no_args_is_help=True)
app.add_typer(data, name="data")

LocalFlag = Annotated[bool, typer.Option("--local", help="Run in local Docker instead of SageMaker")]
PortOption = Annotated[int, typer.Option("--port", help="Local port of the endpoint")]
InstanceOption = Annotated[str | None, typer.Option("--instance", help="SageMaker instance type, e.g. ml.m5.large")]
TimeoutOption = Annotated[str | None, typer.Option("--timeout", help="Stop the job after this long, e.g. 90m, 2h")]
NoWaitFlag = Annotated[bool, typer.Option("--no-wait", help="Print the job name and return without waiting")]
QuietFlag = Annotated[bool, typer.Option("--quiet", "-q", help="Wait without printing logs")]

# Errors from AWS or our own checks that should end the command with a message, not a traceback.
AWS_ERRORS = (CloudError, ValueError, ClientError, BotoCoreError)


@app.callback()
def main(
    ctx: typer.Context,
    config_path: Annotated[
        Path, typer.Option("--config", help="Config file to use instead of ./easy_sm.toml")
    ] = Path(CONFIG_FILE),
) -> None:
    ctx.obj = config_path


def fail(message: str, code: int = 1) -> NoReturn:
    """Print the message on stderr and exit."""
    typer.echo(message, err=True)
    raise typer.Exit(code)


def load_config(ctx: typer.Context) -> Config:
    """Load the config file chosen by --config, or exit with the error."""
    try:
        return config.load(ctx.obj)
    except ConfigError as e:
        fail(str(e))


def build_image(cfg: Config) -> str:
    """Build the image if the code changed; return its tag."""
    try:
        return image.build(cfg.path.parent, cfg.get("app_name"))
    except ImageError as e:
        fail(str(e))


def docker_or_exit(command: list[str]) -> None:
    """Run a container; exit with its exit code if that isn't 0."""
    try:
        code = local.docker(command)
    except ImageError as e:
        fail(str(e))
    if code != 0:
        raise typer.Exit(code)


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Folder to create the project in")] = Path("."),
    name: Annotated[
        str | None, typer.Option("--name", help="App name (default: from the folder name)")
    ] = None,
) -> None:
    """Create a new project: easy_sm.toml, pyproject.toml, src/train.py, src/predict.py and sample data."""
    try:
        created = create_project(path, name)
    except ConfigError as e:
        fail(str(e))
    for file in created:
        typer.echo(file)


@app.command()
def env(ctx: typer.Context) -> None:
    """Show every setting, its value and where the value came from."""
    cfg = load_config(ctx)
    try:
        tag, source = image.image_tag(cfg.path.parent, cfg.get("app_name")), "computed"
    except ImageError:
        tag, source = "-", "run 'easy_sm build'"
    typer.echo(f"{'image':<22} {tag:<45} {source}")
    try:
        s = cloud.session(cfg)
        method = s.get_credentials().method
        via = f"{method}, profile {s.profile_name}" if method == "shared-credentials-file" else method
        account, source = cloud.account_id(s), f"{s.region_name}, credentials: {via}"
    except AWS_ERRORS as e:
        account, source = "-", f"not available ({type(e).__name__})"
    typer.echo(f"{'aws.account':<22} {account:<45} {source}")
    for key in SETTINGS:
        value, source = cfg.resolve(key)
        shown = "-" if value is None else str(value)
        typer.echo(f"{key:<22} {shown:<45} {source}")


@app.command()
def build(ctx: typer.Context) -> None:
    """Build the project's Docker image (skipped if this exact image already exists) and print its tag."""
    typer.echo(build_image(load_config(ctx)))


@app.command()
def push(ctx: typer.Context) -> None:
    """Build the image if needed, push it to ECR (skipped if already there) and print its URI."""
    cfg = load_config(ctx)
    tag = build_image(cfg)
    try:
        s = cloud.session(cfg)
        typer.echo(cloud.push(s, cloud.account_id(s), tag))
    except AWS_ERRORS as e:
        fail(str(e))


def reject_cloud_options(**options: object) -> None:
    """--local runs in Docker on this machine, so SageMaker-only options make no sense."""
    given = [name for name, value in options.items() if value not in (None, False)]
    if given:
        fail(f"{', '.join(given)} only apply to SageMaker jobs, not --local", 2)


def run_cloud_job(
    cfg: Config, kind: str, make_request: Callable[[str, str, str], dict[str, Any]], no_wait: bool, quiet: bool
) -> None:
    """Build and push the image, start the job, print its name, then follow it to the end."""
    tag = build_image(cfg)
    try:
        s = cloud.session(cfg)
        role = cfg.get("aws.role")
        if not role:
            fail("No IAM role: set [aws] role in easy_sm.toml or SAGEMAKER_ROLE")
        uri = cloud.push(s, cloud.account_id(s), tag)
        name = cloud.job_name(tag, cloud.now())
        cloud.start(s, kind, make_request(name, uri, role))
    except AWS_ERRORS as e:
        fail(str(e))
    typer.echo(name)  # stdout: just the job name, so $(easy_sm train) works
    if no_wait:
        return
    try:
        job = cloud.follow(s, kind, name, show_logs=not quiet)
    except KeyboardInterrupt:
        fail(
            f"Stopped following {name}; the job is still running.\n"
            f"Follow it again: easy_sm jobs logs {name} -f\nStop it: easy_sm jobs cancel {name}",
            130,
        )
    if job["status"] != "Completed":
        fail(f"{name} {job['status']}: {job['reason']}")


def s3_default(cfg: Config, folder: str) -> str:
    """s3://sagemaker-<region>-<account>/<app>/<folder>, creating the bucket if needed."""
    s = cloud.session(cfg)
    bucket = cloud.default_bucket(cloud.account_id(s), s.region_name)
    cloud.ensure_bucket(s, bucket)
    return f"s3://{bucket}/{cfg.get('app_name')}/{folder}"


@app.command()
def train(
    ctx: typer.Context,
    is_local: LocalFlag = False,
    input_path: Annotated[
        str | None,
        typer.Option("--input", help="Training data: S3 folder, or a local folder with --local (default data/)"),
    ] = None,
    output: Annotated[str | None, typer.Option("--output", help="S3 folder for the model")] = None,
    instance: InstanceOption = None,
    count: Annotated[int | None, typer.Option("--count", help="Number of instances")] = None,
    timeout: TimeoutOption = None,
    no_wait: NoWaitFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Train a model: calls train() in src/train.py.

    On SageMaker, prints the job name and streams the logs. With --local, prints the model folder.
    """
    cfg = load_config(ctx)
    project = cfg.path.parent
    if is_local:
        reject_cloud_options(
            output=output, instance=instance, count=count, timeout=timeout, no_wait=no_wait, quiet=quiet
        )
        data_dir = Path(input_path or "data")
        data_dir = data_dir if data_dir.is_absolute() else project / data_dir
        if not data_dir.is_dir():
            fail(f"{data_dir} not found")
        tag = build_image(cfg)
        model_dir, output_dir = local.local_dirs(project)
        local.empty(model_dir)
        docker_or_exit(local.train_command(tag, data_dir, model_dir, output_dir))
        typer.echo(model_dir)
        return

    try:
        input_s3 = cfg.get("train.input", input_path) or s3_default(cfg, "data")
        output_s3 = cfg.get("train.output", output) or s3_default(cfg, "models")
        cloud.check_s3_has_data(cloud.session(cfg), input_s3)
        seconds = cloud.parse_timeout(cfg.get("train.timeout", timeout))
    except AWS_ERRORS as e:
        fail(str(e))

    def request(name: str, uri: str, role: str) -> dict[str, Any]:
        return cloud.training_job_request(
            name, uri, role, cfg.get("train.instance", instance), cfg.get("train.count", count),
            input_s3, output_s3, seconds,
        )  # fmt: skip

    run_cloud_job(cfg, "training", request, no_wait, quiet)


@app.command(context_settings={"allow_interspersed_args": False})
def run(
    ctx: typer.Context,
    command: Annotated[
        list[str], typer.Argument(help="Command to run from src/, e.g. clean.py --date 2026-11-01")
    ],
    is_local: LocalFlag = False,
    env_pairs: Annotated[
        list[str] | None, typer.Option("--env", help="Environment variable KEY=VALUE (repeatable)")
    ] = None,
    instance: InstanceOption = None,
    timeout: TimeoutOption = None,
    no_wait: NoWaitFlag = False,
    quiet: QuietFlag = False,
) -> None:
    """Run any command from src/ in the project's image. A .py file runs with python.

    easy_sm options go before the command; everything after it is passed to the command.
    """
    cfg = load_config(ctx)
    project = cfg.path.parent
    try:
        local.check_script(project / "src", command)
        env_vars = {**(cfg.get("run.env") or {}), **local.parse_env(env_pairs or [])}
    except ValueError as e:
        fail(str(e))

    if is_local:
        reject_cloud_options(instance=instance, timeout=timeout, no_wait=no_wait, quiet=quiet)
        tag = build_image(cfg)
        _, output_dir = local.local_dirs(project)
        docker_or_exit(local.run_command(tag, project / "data", output_dir, env_vars, command))
        return

    try:
        seconds = cloud.parse_timeout(cfg.get("run.timeout", timeout))
    except ValueError as e:
        fail(str(e))

    def request(name: str, uri: str, role: str) -> dict[str, Any]:
        return cloud.processing_job_request(
            name, uri, role, cfg.get("run.instance", instance), command, env_vars,
            cfg.get("run.input"), cfg.get("run.output"), seconds,
        )  # fmt: skip

    run_cloud_job(cfg, "processing", request, no_wait, quiet)


@endpoints.command()
def serve(ctx: typer.Context, is_local: LocalFlag = False, port: PortOption = 8080) -> None:
    """Serve the model trained with 'train --local' on 127.0.0.1. Ctrl-C stops it."""
    if not is_local:
        fail("serve only runs locally: use 'easy_sm endpoints serve --local'", 2)
    cfg = load_config(ctx)
    model_dir, _ = local.local_dirs(cfg.path.parent)
    if not any(model_dir.iterdir()):
        fail(f"{model_dir} is empty. Run 'easy_sm train --local' first.")
    tag = build_image(cfg)
    typer.echo(f"Serving {tag} on http://127.0.0.1:{port} (Ctrl-C to stop)", err=True)
    try:
        docker_or_exit(local.serve_command(tag, model_dir, port))
    except KeyboardInterrupt:
        pass  # Ctrl-C also reaches docker, which stops the container


def endpoint_name(cfg: Config, name: str | None) -> str:
    """NAME argument → [endpoint] name → app_name."""
    chosen = name or cfg.get("endpoint.name") or cfg.get("app_name")
    try:
        cloud.check_endpoint_name(chosen)
    except CloudError as e:
        fail(str(e))
    return chosen


@endpoints.command()
def deploy(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Endpoint name (default: [endpoint] name, then app_name)")] = None,
    latest: Annotated[bool, typer.Option("--latest", help="Deploy the newest completed training job")] = False,
    job: Annotated[str | None, typer.Option("--job", help="Deploy this training job's model and image")] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Deploy this model.tar.gz with the CURRENT code (builds and pushes)")
    ] = None,
    instance: InstanceOption = None,
) -> None:
    """Deploy a model to a SageMaker endpoint, or update the endpoint if it exists.

    Prints the endpoint name once it is in service on the new model; exits 1 if it fails or rolls back.
    """
    if sum([latest, job is not None, model is not None]) != 1:
        fail("Choose exactly one of --latest, --job JOB or --model s3://.../model.tar.gz", 2)
    cfg = load_config(ctx)
    endpoint = endpoint_name(cfg, name)
    role = cfg.get("aws.role")
    if not role:
        fail("No IAM role: set [aws] role in easy_sm.toml or SAGEMAKER_ROLE")
    try:
        s = cloud.session(cfg)
        if model:
            cloud.check_s3_has_data(s, model)
            image_uri = cloud.push(s, cloud.account_id(s), build_image(cfg))
            model_s3 = model
        else:
            source_job = job or cloud.latest_job(s, cfg.get("app_name"))
            model_s3, image_uri = cloud.job_source(s, source_job)
            typer.echo(f"Deploying {source_job}", err=True)
        resource = cloud.resource_name(endpoint, cloud.image_hash_of(image_uri), cloud.now())
        config = cloud.endpoint_config_request(
            resource, resource, cfg.get("endpoint.instance", instance), cfg.get("endpoint.serverless"),
            cfg.get("endpoint.memory"), cfg.get("endpoint.concurrency"),
        )  # fmt: skip
        cloud.deploy(s, endpoint, cloud.model_request(resource, image_uri, model_s3, role), config)
        desc = cloud.wait_for_endpoint(s, endpoint)
    except KeyboardInterrupt:
        fail(f"Stopped waiting; {endpoint} keeps deploying. Check it with: easy_sm endpoints describe {endpoint}", 130)
    except AWS_ERRORS as e:
        fail(str(e))
    problem = cloud.deploy_outcome(desc, resource)
    if problem:
        fail(f"{endpoint} {problem}")
    typer.echo(endpoint)


@endpoints.command()
def invoke(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Endpoint name (default: [endpoint] name, then app_name)")] = None,
    is_local: LocalFlag = False,
    data: Annotated[
        str, typer.Option("--data", "-d", help="Request body, or @file to send a file")
    ] = "",
    content_type: Annotated[
        str, typer.Option("--content-type", help="application/json or text/csv")
    ] = "application/json",
    port: PortOption = 8080,
) -> None:
    """Send one prediction request and print the response."""
    try:
        body = local.request_body(data)
    except OSError as e:
        fail(str(e))
    if is_local:
        try:
            status, response = local.invoke(port, body, content_type)
        except ImageError as e:
            fail(str(e))
        if status != 200:
            fail(f"HTTP {status}: {response.decode()}")
        typer.echo(response.decode())
        return
    cfg = load_config(ctx)
    try:
        typer.echo(cloud.invoke_endpoint(cloud.session(cfg), endpoint_name(cfg, name), body, content_type).decode())
    except AWS_ERRORS as e:
        fail(str(e))


@endpoints.command("ls")
def list_endpoints(ctx: typer.Context) -> None:
    """List endpoints in this account and region, newest first: name, status, created."""
    cfg = load_config(ctx)
    try:
        rows = cloud.list_endpoints(cloud.session(cfg))
    except AWS_ERRORS as e:
        fail(str(e))
    for row in rows:
        typer.echo(f"{row['EndpointName']:<40} {row['EndpointStatus']:<14} {row['CreationTime']:%Y-%m-%d %H:%M}")


@endpoints.command()
def describe(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Endpoint name (default: [endpoint] name, then app_name)")] = None,
) -> None:
    """Show an endpoint's status, size, model data and image."""
    cfg = load_config(ctx)
    try:
        details = cloud.endpoint_details(cloud.session(cfg), endpoint_name(cfg, name))
    except AWS_ERRORS as e:
        fail(str(e))
    for label, value in cloud.describe_lines(details):
        typer.echo(f"{label:<12} {value}")


@endpoints.command()
def delete(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Endpoint to delete (always named explicitly)")],
    delete_config: Annotated[
        bool, typer.Option("--delete-config", help="Also delete the endpoint's current config")
    ] = False,
) -> None:
    """Delete an endpoint. Models and older configs are kept; nothing else is removed."""
    cfg = load_config(ctx)
    try:
        cloud.check_endpoint_name(name)
        deleted = cloud.delete_endpoint(cloud.session(cfg), name, delete_config)
    except AWS_ERRORS as e:
        fail(str(e))
    for item in deleted:
        typer.echo(f"deleted {item}")


JobArgument = Annotated[str, typer.Argument(help="Job name, e.g. from 'easy_sm jobs ls -q'")]


def session_or_exit(cfg: Config) -> Any:
    try:
        return cloud.session(cfg)
    except AWS_ERRORS as e:
        fail(str(e))


@jobs.command("ls")
def list_jobs(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("-n", help="How many jobs to show")] = 10,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print job names only")] = False,
) -> None:
    """This app's training and processing jobs, newest first: name, kind, status, created, duration."""
    cfg = load_config(ctx)
    try:
        rows = cloud.list_jobs(session_or_exit(cfg), cfg.get("app_name"), limit)
    except AWS_ERRORS as e:
        fail(str(e))
    for row in rows:
        if quiet:
            typer.echo(row["name"])
        else:
            typer.echo(f"{row['name']:<42} {row['kind']:<6} {row['status']:<11} "
                       f"{row['created']:%Y-%m-%d %H:%M}  {row['duration']}")  # fmt: skip


@jobs.command()
def inspect(ctx: typer.Context, name: JobArgument) -> None:
    """Show a job's status, instance, image, data locations, duration and failure reason."""
    try:
        kind, job = cloud.find_job(session_or_exit(load_config(ctx)), name)
    except AWS_ERRORS as e:
        fail(str(e))
    for label, value in cloud.inspect_lines(kind, job, cloud.now()):
        typer.echo(f"{label:<10} {value}")


@jobs.command()
def logs(
    ctx: typer.Context,
    name: JobArgument,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Keep printing new lines until the job ends")] = False,
) -> None:
    """Print a job's log lines (stdout), as [host] message."""
    s = session_or_exit(load_config(ctx))
    try:
        kind, _ = cloud.find_job(s, name)
        if follow:
            cloud.follow(s, kind, name, show_logs=True, log_out=sys.stdout)
        else:
            for line in cloud.new_log_lines(s, kind, name, {}):
                typer.echo(line)
    except KeyboardInterrupt:
        raise typer.Exit(130)
    except AWS_ERRORS as e:
        fail(str(e))


@jobs.command()
def cancel(ctx: typer.Context, name: JobArgument) -> None:
    """Stop a running job. Prints the job name."""
    s = session_or_exit(load_config(ctx))
    try:
        kind, job = cloud.find_job(s, name)
        status = job["TrainingJobStatus" if kind == "training" else "ProcessingJobStatus"]
        if status != "InProgress":
            fail(f"{name} is {status}; only InProgress jobs can be stopped")
        cloud.stop_job(s, kind, name)
    except AWS_ERRORS as e:
        fail(str(e))
    typer.echo(name)


@jobs.command()
def artifacts(ctx: typer.Context, name: JobArgument) -> None:
    """Print the S3 path of a training job's model.tar.gz."""
    try:
        kind, job = cloud.find_job(session_or_exit(load_config(ctx)), name)
    except AWS_ERRORS as e:
        fail(str(e))
    # SageMaker records a model path even for failed jobs, so the status decides, as for deploy.
    if kind != "training" or job["TrainingJobStatus"] != "Completed":
        fail(f"{name} has no usable model: only completed training jobs do")
    typer.echo(job["ModelArtifacts"]["S3ModelArtifacts"])


@data.command("upload")
def upload_data(
    ctx: typer.Context,
    local_dir: Annotated[Path, typer.Argument(help="Folder to upload")] = Path("data"),
    s3_path: Annotated[
        str | None, typer.Argument(help="S3 folder (default: train.input, then the default bucket)")
    ] = None,
) -> None:
    """Upload a folder to S3 (hidden files skipped). Prints the S3 folder."""
    cfg = load_config(ctx)
    folder = local_dir if local_dir.is_absolute() else cfg.path.parent / local_dir
    if not folder.is_dir():
        fail(f"{folder} not found")
    try:
        target = s3_path or cfg.get("train.input") or s3_default(cfg, "data")
        plan = cloud.upload_plan(folder, target)
        if not plan:
            fail(f"{folder} has no files to upload")
        cloud.upload(session_or_exit(cfg), plan)
    except AWS_ERRORS as e:
        fail(str(e))
    typer.echo(target)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
