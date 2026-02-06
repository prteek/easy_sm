# Documentation Workflow

Guide for working with and maintaining documentation in easy_sm.

## Overview

Documentation must always match the actual code implementation. To ensure this, we use automated validation that:

1. Extracts parameters from Python command files
2. Compares against documented parameters
3. Reports any discrepancies
4. Blocks commits if validation fails

## Workflow

### Before Editing Documentation

1. Understand what needs to change
2. Why the change is needed
3. Which files need to be updated
4. How it impacts other docs

### While Editing Documentation

1. Update the main documentation file
2. Update related files (examples, guides, etc.)
3. Keep terminology consistent
4. Ensure examples are accurate

### After Editing Documentation (MANDATORY)

**You MUST run validation before committing:**

```bash
python3 scripts/validate_docs.py
```

Expected output if valid:
```
🔍 Validating documentation against code implementation...

✅ All documentation is valid!
```

If issues found:
```
🔍 Validating documentation against code implementation...

❌ Documentation issues found:

  ❌ deploy.md: Flag '--tags' documented but not in implementation
  ⚠️  cloud-deployment.md: May contain unsupported feature: 'spot.*instance'

❌ Total: 2 issues
```

Fix the issues and run validation again until it passes.

### Automatic Validation

When you try to commit documentation changes, a pre-commit hook automatically runs validation:

```bash
git add docs/commands/train.md
git commit -m "docs: update train command"

# Output:
# 🔍 Validating documentation...
# ✅ All documentation is valid!
```

If validation fails:
```bash
❌ Documentation validation failed. Fix issues and try again.
   Run: python3 scripts/validate_docs.py
```

## Common Scenarios

### Adding a New Parameter

1. **Add to command code** (`easy_sm/commands/cloud.py`):
```python
@cloud_app.command(name="train")
def train(
    ...
    new_param: Annotated[str, typer.Option("--new-param", "-np", help="Description")] = "default",
):
```

2. **Document in `docs/commands/train.md`**:
```markdown
| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--new-param` | `-np` | string | No | `default` | Description |
```

3. **Run validation**:
```bash
python3 scripts/validate_docs.py
```

4. **Commit when valid**:
```bash
git add easy_sm/commands/cloud.py docs/commands/train.md
git commit -m "feat: add new-param to train command"
```

### Removing a Parameter

1. **Remove from command code**
2. **Remove from all documentation** (docs/commands, docs/user-guide, docs/examples)
3. **Run validation** to ensure removal is complete
4. **Commit** when valid

### Updating Parameter Documentation

1. **Update `docs/commands/*.md`** parameter table
2. **Update `docs/user-guide/*.md`** if applicable
3. **Update examples** if they use the parameter
4. **Run validation** to check for consistency
5. **Commit** when all docs match code

## Validation Rules

The validation script checks for:

### ✅ Valid Documentation

- All documented flags exist in code
- Parameter names match exactly
- Defaults match code defaults
- No unsupported features documented

### ❌ Invalid Documentation

**Extra flags documented:**
```
Flag '--tags' documented but not in implementation
```
→ Remove from documentation

**Wrong flag name:**
```
Flag '--num-instances' documented but actual flag is '-c, --instance-count'
```
→ Update documentation to match code

**Unsupported features:**
```
May contain unsupported feature: 'spot.*instance'
```
→ Remove or update to supported alternative

**Deprecated references:**
```
May contain unsupported feature: 'hyperparameter'
```
→ Remove hyperparameter references

## Supported Features (as of latest validation)

### Cloud Commands

**train**:
- ✅ `--base-job-name`, `-n`
- ✅ `--ec2-type`, `-e`
- ✅ `--input-s3-dir`, `-i`
- ✅ `--output-s3-dir`, `-o`
- ✅ `--instance-count`, `-c`
- ✅ `--iam-role-arn`, `-r`
- ✅ `--app-name`, `-a`
- ❌ `--tags` (not implemented)
- ❌ `--hyperparameters` (not implemented)
- ❌ `--spot-instances` (not implemented)

**deploy**:
- ✅ `--endpoint-name`, `-n`
- ✅ `--instance-type`, `-e`
- ✅ `--s3-model-location`, `-m`
- ✅ `--instance-count`, `-c`
- ✅ `--iam-role-arn`, `-r`
- ✅ `--app-name`, `-a`
- ❌ `--tags` (not implemented)
- ❌ `--vpc-id`, `--subnet-ids`, `--security-group-ids` (not implemented)

**deploy-serverless**:
- ✅ `--endpoint-name`, `-n`
- ✅ `--memory-size-in-mb`, `-s`
- ✅ `--s3-model-location`, `-m`
- ✅ `--max-concurrency`, `-mc`
- ✅ `--iam-role-arn`, `-r`
- ✅ `--app-name`, `-a`
- ❌ `--tags` (not implemented)

**batch-transform**:
- ✅ `--s3-model-location`, `-m`
- ✅ `--s3-input-location`, `-i`
- ✅ `--s3-output-location`, `-o`
- ✅ `--num-instances`
- ✅ `--ec2-type`, `-e`
- ✅ `--wait`, `-w`
- ✅ `--job-name`, `-n`
- ✅ `--iam-role-arn`, `-r`
- ✅ `--app-name`, `-a`
- ❌ `--content-type` (not implemented)
- ❌ `--tags` (not implemented)

### Local Commands

**local train, deploy, process, stop**:
- ✅ `--app-name`, `-a`
- ✅ `--port`, `-p` (deploy/stop)
- ✅ `--file`, `-f` (process)
- ❌ Custom data paths (not supported via flags)
- ❌ Hyperparameter passing (not supported)

## Files Involved

### Validation
- `scripts/validate_docs.py` - Main validation script
- `.githooks/pre-commit` - Git hook that runs before commit
- `.claude/skills/validate-docs/SKILL.md` - Skill definition

### Documentation
- `docs/commands/*.md` - Command reference
- `docs/user-guide/*.md` - User guides
- `docs/examples/*.md` - Examples and workflows
- `README.md` - Project overview

### Implementation
- `easy_sm/commands/cloud.py` - Cloud commands
- `easy_sm/commands/local.py` - Local commands
- `easy_sm/commands/build.py` - Build command
- `easy_sm/commands/push.py` - Push command

## Troubleshooting

### Validation passes locally but fails in CI

- Ensure Python 3.13+ is used
- Check that all doc files are in correct directory
- Verify no uncommitted changes to implementation files

### Pre-commit hook not running

- Check hook is executable: `ls -la .githooks/pre-commit`
- Verify git config: `git config core.hooksPath`
- Reinstall hook: `chmod +x .githooks/pre-commit`

### False positives in validation

- Some patterns (like unsupported features) check for keywords
- May need to adjust regex patterns in `scripts/validate_docs.py`
- Report if patterns are too broad/narrow

## Adding New Commands

When adding a new command:

1. Implement in `easy_sm/commands/*.py`
2. Create documentation in `docs/commands/new-command.md`
3. Add entry to validation in `scripts/validate_docs.py`
4. Add to user guide if applicable
5. Run full validation: `python3 scripts/validate_docs.py`
6. Commit when validation passes

## Questions?

For issues with documentation validation:
- Check `scripts/validate_docs.py` for implementation
- Review `.claude/skills/validate-docs/SKILL.md` for skill details
- See CLAUDE.md for agent workflow guidelines
