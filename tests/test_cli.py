"""
Unit and Integration Tests for the CLI module (main.py).
"""

import os
import sys
import pytest
from unittest.mock import patch

from main import main


def test_cli_integration(tmp_path):
    """
    Integration test for main.py CLI using the sample exports folder.
    Verifies that running main.py generates PDF, Excel, CSV reports and chart PNGs.
    """
    # Define paths
    sample_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "tradingview-xlsx-export-samples"
        )
    )
    output_dir = os.path.join(tmp_path, "output_reports")

    # Construct CLI arguments
    test_args = [
        "main.py",
        "--capital", "50000.0",
        "--leverage", "5.0",
        "--folder", sample_dir,
        "--output", output_dir,
    ]

    # Patch sys.argv and call main
    with patch.object(sys, "argv", test_args):
        # We also patch sys.exit to verify it completes without exiting with non-zero
        with patch.object(sys, "exit") as mock_exit:
            main()
            # If sys.exit was called, assert it wasn't called with a non-zero exit code
            if mock_exit.called:
                args, _ = mock_exit.call_args
                assert args[0] == 0 or args[0] is None

    # Verify that output directory was created
    assert os.path.exists(output_dir)

    # Verify that report files were created
    expected_files = [
        "portfolio_report.pdf",
        "portfolio_report.xlsx",
        "portfolio_report_summary.csv",
        "portfolio_report_trades.csv",
        "portfolio_report_conflicts.csv",
        "equity_drawdown.png",
        "monthly_heatmap.png",
        "yearly_returns.png",
        "capital_usage.png",
    ]

    for fname in expected_files:
        filepath = os.path.join(output_dir, fname)
        assert os.path.exists(filepath), f"File not found: {filepath}"
        assert os.path.getsize(filepath) > 0, f"File is empty: {filepath}"


def test_cli_invalid_folder(tmp_path):
    """
    Verifies that main.py CLI exits with an error code when provided a non-existent folder.
    """
    output_dir = os.path.join(tmp_path, "output_reports")
    test_args = [
        "main.py",
        "--folder", "/non/existent/path/to/folder",
        "--output", output_dir,
    ]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1


def test_cli_date_filters(tmp_path):
    """
    Verifies that main.py CLI successfully filters trades using --start-date and --end-date.
    """
    sample_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "tradingview-xlsx-export-samples"
        )
    )
    output_dir = os.path.join(tmp_path, "output_reports_filtered")

    # Let's filter to keep trades after 2020-01-01 and before 2026-12-31
    test_args = [
        "main.py",
        "--folder", sample_dir,
        "--output", output_dir,
        "--start-date", "2020-01-01",
        "--end-date", "2026-12-31",
    ]

    with patch.object(sys, "argv", test_args):
        with patch.object(sys, "exit") as mock_exit:
            main()
            if mock_exit.called:
                args, _ = mock_exit.call_args
                assert args[0] == 0 or args[0] is None

    assert os.path.exists(output_dir)


def test_cli_invalid_date_formats(tmp_path):
    """
    Verifies that main.py CLI exits with an error code (1) when provided invalid date formats.
    """
    sample_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "tradingview-xlsx-export-samples"
        )
    )
    output_dir = os.path.join(tmp_path, "output_reports_invalid_date")

    # Invalid start date
    test_args = [
        "main.py",
        "--folder", sample_dir,
        "--output", output_dir,
        "--start-date", "invalid-date-format",
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    # Invalid end date
    test_args = [
        "main.py",
        "--folder", sample_dir,
        "--output", output_dir,
        "--end-date", "2025/12/31",  # incorrect format, requires YYYY-MM-DD
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

