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

1. Extracts parameter definitions from command implementation files
2. Compares against documented parameters in markdown files
3. Reports any discrepancies:
   - Extra flags documented but not implemented
   - Unsupported features mentioned
   - Incorrect parameter names
   - Missing options

## Return Codes

- `0` - All documentation is valid
- `1` - Issues found in documentation

## When to Use

- After editing any documentation files
- Before committing documentation changes
- As part of CI/CD validation
- When adding new command parameters

## Example Output

```
✅ All documentation is valid!
```

Or if issues found:

```
❌ Documentation issues found:

  ❌ deploy.md: Flag '--tags' documented but not in implementation
  ⚠️  cloud-deployment.md: May contain unsupported feature: 'spot.*instance'

❌ Total: 2 issues
```

## Files Validated

- `docs/commands/*.md`
- `docs/user-guide/*.md`
- Against: `easy_sm/commands/*.py`

## Implementation

Location: `scripts/validate_docs.py`

The script uses AST parsing to extract actual parameters from Python command files and regex to extract documented parameters from markdown files.
