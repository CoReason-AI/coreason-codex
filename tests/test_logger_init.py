# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import importlib
import shutil
import sys
from pathlib import Path

from loguru import logger


def test_logger_creates_directory() -> None:
    """Test that the logger module creates the logs directory if it doesn't exist."""
    log_path = Path("logs")

    # 0. Ensure logger releases file handles (critical for Windows)
    logger.remove()

    # 1. Cleanup before test
    if log_path.exists():
        shutil.rmtree(log_path)

    # 2. Unload the module if it's already loaded
    if "coreason_codex.utils.logger" in sys.modules:
        del sys.modules["coreason_codex.utils.logger"]

    # 3. Import the module (should trigger execution of the top-level code)
    import coreason_codex.utils.logger

    importlib.reload(coreason_codex.utils.logger)

    # 4. Assert directory exists
    assert log_path.exists()
    assert log_path.is_dir()

    # Cleanup after (optional but good practice)
    # logger.remove()
    # We don't remove here because subsequent tests might rely on the logger being active.
