# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from coreason_codex.utils.logger import logger


def test_logger_interface() -> None:
    # Verify logger is accessible
    logger.info("Test log")
    assert True


def test_logger_file_creation(tmp_path) -> None:  # type: ignore
    # Note: We can't easily change the logger path at runtime without re-configuring loguru.
    # The default logger writes to logs/app.log relative to CWD.
    # We just check if that path exists.
    import pathlib

    log_file = pathlib.Path("logs/app.log")

    # We can write something
    logger.info("Verification")

    # Force flush/close isn't straightforward with global logger, but file should exist
    assert log_file.exists()
