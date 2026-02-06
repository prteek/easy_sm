# init

Initialize a new easy_sm project with SageMaker template.

## Synopsis

```bash
easy_sm init
```

## Description

The `init` command sets up a new easy_sm project by creating the necessary directory structure, configuration file, and template files. It interactively prompts you for project configuration including app name, AWS credentials, Python version, and dependencies.

This command creates:

- A JSON configuration file (`{app_name}.json`)
- Project directory structure with Docker templates
- Entry points for training, serving, and processing
- Local test directories and scripts

## Options

This command has no options. It runs interactively and prompts for all required information.

## Interactive Prompts

The command will ask you for the following information:

1. **App name**: Alphanumeric characters and hyphens only
2. **New project**: Whether you're starting a new project or integrating with existing code
3. **Root directory**: (Only for existing projects) Where your code lives
4. **Python version**: Choose from 3.10, 3.11, 3.12, or 3.13
5. **AWS profile**: Select from available AWS profiles in `~/.aws/credentials`
6. **AWS region**: Your preferred AWS region (e.g., `us-east-1`, `eu-west-1`)
7. **Requirements file**: Path to your `requirements.txt` file

## Examples

### Initialize a new project

```bash
easy_sm init
```

**Interactive session:**

```
Type in a name for your SageMaker app (Only alphanumeric characters and - are allowed): my-ml-app
Are you starting a new project? [y/n]: y
Select Python interpreter:
1 - Python310
2 - Python311
3 - Python312
4 - Python313
Choose from 1, 2, 3, 4 [4]: 4
Select AWS profile:
1 - default
2 - dev
3 - prod
Choose from 1, 2, 3 [1]: 2
Type in your preferred AWS region name [eu-west-1]: us-east-1
Type in the path to requirements.txt. Example: requirements.txt: requirements.txt

easy_sm module is created! ヽ(´▽`)/
```

### Initialize with existing code

If you have existing code you want to integrate with easy_sm:

```bash
easy_sm init
```

When prompted "Are you starting a new project?", answer `n`, then provide the directory where your code lives:

```
Are you starting a new project? [y/n]: n
Type in the directory where your code lives. Example: src: src
```

## Output

After successful initialization, you'll have:

```
my-project/
├── my-ml-app.json                    # Configuration file
├── requirements.txt                   # Your dependencies
└── my-ml-app/                        # (or your specified directory)
    ├── __init__.py
    └── easy_sm_base/
        ├── Dockerfile                # Docker configuration
        ├── build.sh                  # Build script
        ├── push.sh                   # ECR push script
        ├── executor.sh               # Container executor
        ├── training/
        │   ├── train                 # Training entry point
        │   └── training.py           # Your training code goes here
        ├── prediction/
        │   └── serve                 # Serving entry point
        ├── processing/               # Processing scripts directory
        └── local_test/
            ├── train_local.sh        # Local training script
            ├── deploy_local.sh       # Local deployment script
            ├── process_local.sh      # Local processing script
            ├── stop_local.sh         # Stop local deployment
            └── test_dir/             # Test data directory
```

## Configuration File

The generated configuration file (`{app_name}.json`) contains:

```json
{
    "image_name": "my-ml-app",
    "aws_profile": "dev",
    "aws_region": "us-east-1",
    "python_version": "3.13",
    "easy_sm_module_dir": "my-ml-app",
    "requirements_dir": "requirements.txt"
}
```

## Prerequisites

- Python >=3.13 installed
- AWS CLI configured with at least one profile in `~/.aws/credentials` (optional but recommended)
- A `requirements.txt` file for your Python dependencies

!!! tip "AWS Profile Configuration"
    If no AWS profiles are found, the command will allow you to proceed with an empty profile. In this case, easy_sm will use AWS environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) or the default credential chain.

## Next Steps

After initialization:

1. **Add your training code** to `{app_name}/easy_sm_base/training/training.py`
2. **Add your serving code** to `{app_name}/easy_sm_base/prediction/serve`
3. **Add test data** to `{app_name}/easy_sm_base/local_test/test_dir/input/data/training/`
4. **Build the Docker image**: `easy_sm build`
5. **Test locally**: `easy_sm local train`

## Troubleshooting

### Invalid app name error

**Problem**: `invalid app name: my_app_123. App name must only have alphanumeric characters or dashes '-'`

**Solution**: Use only alphanumeric characters and hyphens. Underscores are not allowed.

```bash
# Bad
my_ml_app

# Good
my-ml-app
```

### Directory already exists

**Problem**: `There is a easy_sm directory/module already. Please, rename it in order to use easy_sm.`

**Solution**: The `easy_sm_base` directory already exists in your project. Either:

- Delete or rename the existing directory
- Use a different app name
- Initialize in a different directory

### No AWS profiles found

**Problem**: `No AWS profiles found in ~/.aws/credentials`

**Solution**: Either:

- Configure AWS CLI: `aws configure --profile dev`
- Set environment variables before running commands:
  ```bash
  export AWS_ACCESS_KEY_ID=your_key
  export AWS_SECRET_ACCESS_KEY=your_secret
  ```

## Related Commands

- [`build`](build.md) - Build the Docker image
- [`local train`](local.md#train) - Test training locally
- [`update-scripts`](update-scripts.md) - Update shell scripts in existing projects

## See Also

- [Quick Start Guide](../getting-started/quick-start.md)
- [Project Structure](../getting-started/configuration.md)
- [Configuration Reference](../getting-started/configuration.md)
