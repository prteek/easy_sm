# update-scripts

Update shell scripts in easy_sm projects with the latest secure versions.

## Synopsis

```bash
easy_sm update-scripts [--app-name APP_NAME]
```

## Description

The `update-scripts` command copies the latest shell scripts from the easy_sm package template to your project's `easy_sm_base` directory. This is useful for:

- Applying security fixes (proper variable quoting, path handling)
- Getting new features added to shell scripts
- Fixing bugs in build, push, or local test scripts
- Updating existing projects after easy_sm package updates

The command updates 7 shell scripts:
- `build.sh` - Docker image build script
- `push.sh` - ECR push script
- `executor.sh` - Container executor
- `local_test/train_local.sh` - Local training script
- `local_test/process_local.sh` - Local processing script
- `local_test/deploy_local.sh` - Local deployment script
- `local_test/stop_local.sh` - Stop local deployment script

## Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |

## Examples

### Update with auto-detected app

```bash
easy_sm update-scripts
```

Output:
```
Updating shell scripts in /path/to/my-app/easy_sm_base...

  Updated: build.sh
  Updated: push.sh
  Updated: executor.sh
  Updated: local_test/train_local.sh
  Updated: local_test/process_local.sh
  Updated: local_test/deploy_local.sh
  Updated: local_test/stop_local.sh

Successfully updated 7 shell script(s).
```

### Update specific app

```bash
easy_sm update-scripts -a my-ml-app
```

## What Gets Updated

### build.sh
Handles Docker image building with:
- Proper variable quoting to prevent injection
- Safe path handling
- Error handling improvements

### push.sh
Handles ECR authentication and image push with:
- Secure credential handling
- Proper error propagation
- Support for IAM role and profile authentication

### executor.sh
Container entry point executor with:
- Safe command execution
- Environment variable handling
- Proper signal handling

### Local Test Scripts

**train_local.sh:**
- Local training container setup
- Volume mounting
- Path handling improvements

**process_local.sh:**
- Local processing job execution
- AWS credential forwarding
- Proper cleanup

**deploy_local.sh:**
- Local endpoint deployment
- Port binding
- Container lifecycle management

**stop_local.sh:**
- Safe container termination
- Port cleanup

## Security Fixes

The update-scripts command primarily addresses security issues found in older versions:

### 1. Shell Variable Quoting

**Old (vulnerable):**
```bash
docker build -t $IMAGE_NAME:$TAG $BUILD_CONTEXT
```

**New (secure):**
```bash
docker build -t "$IMAGE_NAME:$TAG" "$BUILD_CONTEXT"
```

Without quotes, special characters in variables can cause:
- Command injection
- Path traversal
- Unexpected behavior

### 2. Path Sanitization

**Old:**
```bash
cd $PROJECT_DIR
```

**New:**
```bash
cd "${PROJECT_DIR}" || exit 1
```

Ensures paths exist and exits cleanly on errors.

### 3. Error Handling

**Old:**
```bash
docker push $IMAGE
```

**New:**
```bash
if ! docker push "$IMAGE"; then
  echo "Error: Failed to push image" >&2
  exit 1
fi
```

Proper error detection and user feedback.

## When to Update

Update scripts when:

1. **After upgrading easy_sm package:**
   ```bash
   pip install --upgrade easy-sm
   easy_sm update-scripts
   ```

2. **Security advisories**: Check release notes for security fixes

3. **Bug fixes**: If you encounter issues with shell scripts

4. **New features**: To access new functionality in scripts

5. **Project migration**: When moving projects between environments

## Prerequisites

- Valid easy_sm project (initialized with `easy_sm init`)
- Configuration file (`{app_name}.json`) in current directory
- `easy_sm_base/` directory exists in project

## Backup Considerations

The command overwrites existing scripts. If you've customized scripts, back them up first:

```bash
# Backup custom scripts
cp my-app/easy_sm_base/build.sh my-app/easy_sm_base/build.sh.custom
cp my-app/easy_sm_base/push.sh my-app/easy_sm_base/push.sh.custom

# Update scripts
easy_sm update-scripts

# Compare and merge customizations
diff my-app/easy_sm_base/build.sh my-app/easy_sm_base/build.sh.custom
```

## Customized Scripts

If you've customized shell scripts, the update will overwrite your changes. Options:

### Option 1: Version Control

Use git to track changes and merge updates:

```bash
# Commit current state
git add my-app/easy_sm_base/*.sh
git commit -m "Current shell scripts"

# Update scripts
easy_sm update-scripts

# Review changes
git diff

# Merge or revert as needed
git checkout -- build.sh  # Revert specific file
```

### Option 2: Manual Merge

```bash
# Backup
cp build.sh build.sh.custom

# Update
easy_sm update-scripts

# Compare
diff build.sh build.sh.custom

# Manually merge customizations into new build.sh
```

### Option 3: Override Pattern

Keep customizations in separate files:

```bash
# build.sh (updated by easy_sm)
# build_custom.sh (your customizations)

# Wrapper script
#!/bin/bash
# build_wrapper.sh
source ./build.sh
source ./build_custom.sh
```

## File Permissions

The update command sets proper file permissions:
- All scripts: `755` (rwxr-xr-x)
- Executable by owner, readable by all
- Not world-writable (security)

## Verification

After updating, verify scripts:

```bash
# Check files updated
ls -la my-app/easy_sm_base/*.sh
ls -la my-app/easy_sm_base/local_test/*.sh

# Test build
easy_sm build

# Test local training
easy_sm local train
```

## Version Compatibility

Scripts are version-locked to the easy_sm package version. After updating:

```bash
# Check package version
pip show easy-sm | grep Version

# Update scripts to match
easy_sm update-scripts
```

Always update scripts after upgrading the package.

## Troubleshooting

### "easy_sm_base directory not found"

**Problem**: Project structure invalid.

**Solution**: Ensure you're in the correct directory with valid project:

```bash
ls my-app/easy_sm_base/
# Should show: Dockerfile, build.sh, etc.
```

If missing, re-initialize:
```bash
easy_sm init
```

### "Warning: Source script not found"

**Problem**: easy_sm package installation incomplete.

**Solution**: Reinstall package:

```bash
pip install --force-reinstall easy-sm
```

### Scripts don't work after update

**Problem**: Incompatibility with customizations.

**Solution**: Restore backups and manually merge:

```bash
# Restore backup
git checkout -- my-app/easy_sm_base/

# Or from manual backup
cp build.sh.backup build.sh

# Review release notes for breaking changes
```

### Permission denied errors

**Problem**: Scripts not executable after update.

**Solution**: The command should set permissions automatically, but you can manually fix:

```bash
chmod +x my-app/easy_sm_base/*.sh
chmod +x my-app/easy_sm_base/local_test/*.sh
```

## Update Frequency

**Recommended schedule:**
- **Security updates**: Immediately when announced
- **Package upgrades**: Every time you upgrade easy_sm
- **Bug fixes**: As needed when encountering issues
- **Regular maintenance**: Monthly or quarterly

## Breaking Changes

Major version updates may have breaking changes. Check release notes:

```bash
# Before updating
pip show easy-sm

# Read release notes
# https://github.com/prteek/easy_sm/releases

# Update package
pip install --upgrade easy-sm

# Update scripts
easy_sm update-scripts

# Test thoroughly
easy_sm build
easy_sm local train
```

## Complete Update Workflow

```bash
# 1. Check current version
pip show easy-sm

# 2. Backup current scripts (if customized)
mkdir -p backups/$(date +%Y%m%d)
cp -r my-app/easy_sm_base/*.sh backups/$(date +%Y%m%d)/

# 3. Commit to git (if using version control)
git add my-app/easy_sm_base/
git commit -m "Backup shell scripts before update"

# 4. Upgrade package
pip install --upgrade easy-sm

# 5. Update scripts
easy_sm update-scripts

# 6. Review changes
git diff my-app/easy_sm_base/

# 7. Test build and local operations
easy_sm build
easy_sm local train

# 8. If everything works, commit
git add my-app/easy_sm_base/
git commit -m "Update shell scripts to latest version"

# 9. If issues, restore backup
# git checkout -- my-app/easy_sm_base/
# or
# cp -r backups/$(date +%Y%m%d)/* my-app/easy_sm_base/
```

## Script Descriptions

### build.sh
Builds Docker image with your code, dependencies, and SageMaker entry points.

**What it does:**
- Validates Dockerfile exists
- Copies source code into build context
- Runs `docker build` with proper tags
- Handles build errors

**Parameters:**
- Source directory
- Target directory
- Dockerfile path
- Requirements file
- Docker tag
- Image name
- Python version

### push.sh
Pushes Docker image to AWS ECR.

**What it does:**
- Authenticates with ECR
- Creates repository if needed
- Tags image with ECR URI
- Pushes image layers
- Handles authentication via profile or IAM role

**Parameters:**
- Docker tag
- AWS region
- IAM role ARN (optional)
- AWS profile (optional)
- External ID (optional)
- Image name

### executor.sh
Entry point executor for training/serving containers.

**What it does:**
- Validates environment
- Executes training or serving code
- Handles signals (SIGTERM, SIGINT)
- Manages cleanup

### Local Test Scripts

**train_local.sh:**
- Mounts local test data
- Runs training container
- Maps volume paths
- Captures output

**process_local.sh:**
- Sets up processing environment
- Forwards AWS credentials
- Runs processing script
- Cleans up resources

**deploy_local.sh:**
- Starts serving container
- Binds to specified port
- Mounts model directory
- Runs in background

**stop_local.sh:**
- Finds running container
- Stops container gracefully
- Cleans up resources
- Frees port

## Related Commands

- [`init`](init.md) - Initialize new projects (includes latest scripts)
- [`build`](build.md) - Uses build.sh
- [`push`](push.md) - Uses push.sh
- [`local train`](local.md#train) - Uses train_local.sh
- [`local deploy`](local.md#deploy) - Uses deploy_local.sh

## See Also

- [Security Best Practices](../user-guide/security.md)
- [Project Structure](../user-guide/project-structure.md)
- [Release Notes](https://github.com/prteek/easy_sm/releases)
- [Contributing Guide](../developer-guide/contributing.md)
