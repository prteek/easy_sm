# validate-docs

Validate all documentation against actual command implementations.

## Description

This skill runs the documentation validation check to ensure that:
- All documented parameters actually exist in the code
- No unsupported features are mentioned (hyperparameters, spot instances, tags, etc.)
- Flag names match between documentation and implementation
- No deprecated features are documented

## Usage

```bash
/validate-docs
```

## What It Does

1. **Parameter Validation**: Extracts parameter definitions from command implementation files and compares against documented parameters in markdown files
2. **Discrepancy Detection**: Reports any parameter mismatches:
   - Extra flags documented but not implemented
   - Unsupported features mentioned
   - Incorrect parameter names
   - Missing options
3. **Example Validation**: Analyzes command examples in documentation for common usage mistakes:
   - Using `-n` flag with date filtering (won't work - names-only output lacks timestamps)
   - Incorrect `awk` field extraction for job names (should use `$1`, not `$2`)
   - Incomplete variable assignments missing field extraction

## Return Codes

- `0` - All documentation is valid
- `1` - Issues found in documentation

## When to Use

- After editing any documentation files
- Before committing documentation changes
- As part of CI/CD validation
- When adding new command parameters
- When updating command examples or workflows
- To catch common bash scripting mistakes in examples

## Example Output

```
✅ All documentation is valid!
```

Or if parameter issues found:

```
❌ Documentation issues found:

  ❌ deploy.md: Flag '--tags' documented but not in implementation
  ⚠️  cloud-deployment.md: May contain unsupported feature: 'spot.*instance'

❌ Total: 2 issues
```

Example validation issues:

```
❌ Documentation issues found:

  ⚠️  piped-workflows.md: Using -n flag with list-training-jobs then grepping for date - dates are not in output with -n
  ⚠️  train.md: Using awk '$2' to extract from list-training-jobs - should use '$1' for job name
  ⚠️  advanced-workflows.md: Assigning full list-training-jobs output to JOB variable without awk extraction

❌ Total: 3 issues
```

## Files Validated

- `docs/commands/*.md`
- `docs/user-guide/*.md`
- `docs/examples/*.md`
- Against: `easy_sm/commands/*.py`

## Common Example Mistakes Detected

### Date Filtering with `-n` Flag
```bash
# ❌ WRONG: -n outputs only names, no timestamps
easy_sm list-training-jobs -n -m 100 | grep $DATE

# ✅ CORRECT: Without -n to include timestamps
easy_sm list-training-jobs -m 100 | grep $DATE
```

### Incorrect `awk` Field Extraction
```bash
# ❌ WRONG: $2 is status, not job name
JOB=$(easy_sm list-training-jobs | grep Completed | head -1 | awk '{print $2}')

# ✅ CORRECT: $1 is job name
JOB=$(easy_sm list-training-jobs | grep Completed | head -1 | awk '{print $1}')
```

### Incomplete Variable Assignment
```bash
# ❌ WRONG: Assigns full output line, not just name
JOB=$(easy_sm list-training-jobs -m 20 | grep Completed | head -1)

# ✅ CORRECT: Extracts just the job name
JOB=$(easy_sm list-training-jobs -m 20 | grep Completed | head -1 | awk '{print $1}')
```

## Implementation

Location: `scripts/validate_docs.py`

The script uses AST parsing to extract actual parameters from Python command files and regex to extract documented parameters from markdown files.
