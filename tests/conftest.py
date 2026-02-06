"""
Shared pytest configuration and fixtures.
"""

import os
import pytest


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip tests that require app/ directory in CI.

    The app/ directory contains sample app code but is gitignored, so it won't
    be available in CI. These tests are integration tests that require the full
    app structure to be present.
    """
    app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
    app_exists = os.path.exists(app_dir) and os.path.isdir(app_dir)

    skip_local = pytest.mark.skip(reason="Requires app/ directory (not available in CI)")

    for item in items:
        # Skip tests in test_local_commands.py if app/ doesn't exist
        if "test_local_commands" in item.nodeid and not app_exists:
            item.add_marker(skip_local)
