#!/usr/bin/env python3
"""
Validate documentation against actual command implementations.

This script extracts parameters from command files and compares them
against documented parameters to find discrepancies.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


class CommandValidator:
    """Validates command documentation against implementation."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.commands_dir = repo_root / "easy_sm" / "commands"
        self.docs_dir = repo_root / "docs"
        self.issues: List[str] = []

    def extract_command_params(self, file_path: Path) -> Dict[str, Dict[str, Set[str]]]:
        """Extract parameter definitions from a command file."""
        with open(file_path) as f:
            tree = ast.parse(f.read())

        commands = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's decorated with @app.command() or similar
                is_command = any(
                    isinstance(dec, ast.Call) and
                    getattr(getattr(dec.func, 'attr', None), 'name', None) in ['command', 'Command']
                    or getattr(dec.func, 'id', None) == 'command'
                    for dec in node.decorator_list
                )

                if is_command or node.name in ['train', 'deploy', 'deploy_serverless', 'batch_transform', 'process', 'upload_data', 'delete_endpoint', 'list_endpoints', 'list_training_jobs', 'get_model_artifacts']:
                    params = self._extract_function_params(node)
                    command_name = node.name.replace('_', '-')
                    commands[command_name] = params

        return commands

    def _extract_function_params(self, func_node: ast.FunctionDef) -> Dict[str, Set[str]]:
        """Extract parameter names and their short/long flags from function signature."""
        params = {'names': set(), 'flags': set(), 'short_flags': set()}

        for arg in func_node.args.args:
            param_name = arg.arg
            if param_name in ['self', 'cls']:
                continue

            params['names'].add(param_name)

            # Try to extract typer.Option flags from type annotations
            if arg.annotation and isinstance(arg.annotation, ast.Subscript):
                # Look for typer.Option(...) in Annotated[type, typer.Option(...)]
                if hasattr(arg.annotation, 'slice'):
                    slice_node = arg.annotation.slice
                    if isinstance(slice_node, ast.Tuple):
                        for elt in slice_node.elts:
                            if isinstance(elt, ast.Call):
                                self._extract_flags_from_call(elt, params)

        return params

    def _extract_flags_from_call(self, call_node: ast.Call, params: Dict[str, Set[str]]):
        """Extract flag names from typer.Option() call."""
        for arg in call_node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flag = arg.value
                if flag.startswith('--'):
                    params['flags'].add(flag)
                elif flag.startswith('-') and len(flag) == 2:
                    params['short_flags'].add(flag)

    def extract_doc_params(self, doc_file: Path) -> Dict[str, Set[str]]:
        """Extract documented parameters from a markdown file."""
        with open(doc_file) as f:
            content = f.read()

        params = {'flags': set(), 'short_flags': set(), 'deprecated': set()}

        # Find parameter tables (| Parameter | Flag | ...)
        table_pattern = r'\|[^\n]*Flag[^\n]*\|[^\n]*\n\|[-:| ]+\|[^\n]*\n((?:\|[^\n]+\n)+)'

        for match in re.finditer(table_pattern, content):
            table_content = match.group(1)
            # Extract flags from table rows
            for row in table_content.split('\n'):
                if '|' in row:
                    cells = [cell.strip() for cell in row.split('|')]
                    if len(cells) >= 3:
                        flag_cell = cells[2] if len(cells) > 2 else ''
                        # Extract all flags (both long and short)
                        flags = re.findall(r'(-{1,2}[a-zA-Z0-9-]+)', flag_cell)
                        for flag in flags:
                            if flag.startswith('--'):
                                params['flags'].add(flag)
                            elif flag.startswith('-'):
                                params['short_flags'].add(flag)

        # Check for deprecated/unsupported features only in parameter documentation sections
        # (not in code examples or AWS CLI commands)
        deprecated_patterns = [
            # Spot instances - look for "spot" in parameter descriptions or examples specific to easy_sm
            (r'easy_sm.*--spot-instances', 'spot instances'),
            (r'enable spot training', 'spot training'),
            # Hyperparameters - only if documented as easy_sm parameter
            (r'^\s*\|\s*[^|]*hyperparameter[^|]*\|.*\|\s*`--', 'hyperparameters'),
            # Tags parameter for easy_sm commands
            (r'^\s*\|\s*Tags\s*\|\s*`?--tags', 'tags parameter'),
            # VPC parameters
            (r'^\s*\|\s*VPC', 'vpc parameters'),
        ]

        for pattern, name in deprecated_patterns:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                params['deprecated'].add(f"{name} ({pattern})")

        return params

    def validate_command_docs(self, command_name: str, impl_file: Path, doc_file: Path):
        """Validate a single command's documentation."""
        if not impl_file.exists():
            self.issues.append(f"❌ Implementation file not found: {impl_file}")
            return

        if not doc_file.exists():
            self.issues.append(f"❌ Documentation file not found: {doc_file}")
            return

        # Extract parameters
        impl_commands = self.extract_command_params(impl_file)
        doc_params = self.extract_doc_params(doc_file)

        # Find the matching command in implementation
        impl_params = None
        for cmd_name, params in impl_commands.items():
            if cmd_name == command_name or cmd_name.replace('-', '_') == command_name.replace('-', '_'):
                impl_params = params
                break

        if not impl_params:
            self.issues.append(f"⚠️  Command '{command_name}' not found in {impl_file.name}")
            return

        # Compare flags
        impl_flags = impl_params['flags']
        impl_short = impl_params['short_flags']
        doc_flags = doc_params['flags']
        doc_short = doc_params['short_flags']

        # Find flags in docs but not in implementation
        extra_flags = doc_flags - impl_flags
        extra_short = doc_short - impl_short

        if extra_flags:
            for flag in extra_flags:
                self.issues.append(f"❌ {doc_file.name}: Flag '{flag}' documented but not in implementation")

        if extra_short:
            for flag in extra_short:
                self.issues.append(f"❌ {doc_file.name}: Short flag '{flag}' documented but not in implementation")

        # Check for deprecated features
        if doc_params['deprecated']:
            for pattern in doc_params['deprecated']:
                self.issues.append(f"⚠️  {doc_file.name}: May contain unsupported feature: '{pattern}'")

    def validate_all(self) -> bool:
        """Validate all command documentation."""
        print("🔍 Validating documentation against code implementation...\n")

        # Map of commands to validate
        validations = [
            ('train', self.commands_dir / 'cloud.py', self.docs_dir / 'commands' / 'train.md'),
            ('deploy', self.commands_dir / 'cloud.py', self.docs_dir / 'commands' / 'deploy.md'),
            ('deploy_serverless', self.commands_dir / 'cloud.py', self.docs_dir / 'commands' / 'deploy.md'),
            ('batch_transform', self.commands_dir / 'cloud.py', self.docs_dir / 'commands' / 'batch-transform.md'),
            ('process', self.commands_dir / 'cloud.py', self.docs_dir / 'commands' / 'process.md'),
            ('upload_data', self.commands_dir / 'cloud.py', self.docs_dir / 'commands' / 'train.md'),
            ('train', self.commands_dir / 'local.py', self.docs_dir / 'commands' / 'local.md'),
            ('deploy', self.commands_dir / 'local.py', self.docs_dir / 'commands' / 'local.md'),
        ]

        for command_name, impl_file, doc_file in validations:
            self.validate_command_docs(command_name, impl_file, doc_file)

        # Also check user guide
        user_guide = self.docs_dir / 'user-guide' / 'cloud-deployment.md'
        if user_guide.exists():
            doc_params = self.extract_doc_params(user_guide)
            if doc_params['deprecated']:
                for pattern in doc_params['deprecated']:
                    self.issues.append(f"⚠️  cloud-deployment.md: May contain unsupported feature: '{pattern}'")

        # Report results
        if self.issues:
            print("❌ Documentation issues found:\n")
            for issue in self.issues:
                print(f"  {issue}")
            print(f"\n❌ Total: {len(self.issues)} issues")
            return False
        else:
            print("✅ All documentation is valid!")
            return True


def main():
    """Run documentation validation."""
    import sys

    repo_root = Path(__file__).parent.parent
    validator = CommandValidator(repo_root)

    success = validator.validate_all()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
