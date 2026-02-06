"""
Unit tests for the helpers module.

Tests the safe_run_subprocess function for subprocess execution,
error handling, and output handling.
"""

import subprocess
from unittest.mock import MagicMock, patch, call
from io import StringIO

import pytest

from easy_sm.commands.helpers import safe_run_subprocess


class TestSafeRunSubprocess:
    """Tests for the safe_run_subprocess function."""

    @patch("subprocess.Popen")
    def test_successful_subprocess_execution(self, mock_popen: MagicMock) -> None:
        """Test successful subprocess execution."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["Output line 1\n", "Output line 2\n"])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        return_code = safe_run_subprocess(["echo", "test"])

        assert return_code == 0
        mock_popen.assert_called_once_with(
            ["echo", "test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    @patch("subprocess.Popen")
    def test_subprocess_with_success_message(self, mock_popen: MagicMock) -> None:
        """Test subprocess execution with success message."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with patch("builtins.print") as mock_print:
            return_code = safe_run_subprocess(
                ["echo", "test"], success_message="Success!"
            )

            assert return_code == 0
            # Check that success message was printed
            mock_print.assert_called()
            print_calls = [str(call_arg) for call_arg in mock_print.call_args_list]
            assert any("Success!" in str(call_arg) for call_arg in print_calls)

    @patch("subprocess.Popen")
    def test_subprocess_failure(self, mock_popen: MagicMock) -> None:
        """Test subprocess execution failure."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["Error occurred\n"])
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process

        return_code = safe_run_subprocess(["false"])

        assert return_code == 1

    @patch("subprocess.Popen")
    def test_subprocess_output_printed(self, mock_popen: MagicMock) -> None:
        """Test that subprocess output is printed."""
        output_lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        mock_process = MagicMock()
        mock_process.stdout = iter(output_lines)
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with patch("builtins.print") as mock_print:
            return_code = safe_run_subprocess(["echo", "test"])

            assert return_code == 0
            # Verify that each line was printed
            for line in output_lines:
                mock_print.assert_any_call(line, end="")

    @patch("subprocess.Popen")
    def test_subprocess_no_stdout(self, mock_popen: MagicMock) -> None:
        """Test subprocess with None stdout."""
        mock_process = MagicMock()
        mock_process.stdout = None
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        return_code = safe_run_subprocess(["test"])

        assert return_code == 0

    @patch("subprocess.Popen")
    def test_subprocess_empty_stdout(self, mock_popen: MagicMock) -> None:
        """Test subprocess with empty stdout."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        return_code = safe_run_subprocess(["test"])

        assert return_code == 0

    @patch("subprocess.Popen")
    def test_subprocess_with_complex_command(self, mock_popen: MagicMock) -> None:
        """Test subprocess with complex command arguments."""
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            "/path/to/data:/data",
            "image:tag",
            "python",
            "script.py",
        ]

        mock_process = MagicMock()
        mock_process.stdout = iter(["Container output\n"])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        return_code = safe_run_subprocess(command)

        assert return_code == 0
        mock_popen.assert_called_once_with(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    @patch("subprocess.Popen")
    def test_subprocess_called_process_error(self, mock_popen: MagicMock) -> None:
        """Test subprocess that raises CalledProcessError."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["Some output\n"])
        mock_process.wait.return_value = 127  # Command not found

        mock_popen.return_value = mock_process

        with patch("builtins.print") as mock_print:
            return_code = safe_run_subprocess(["nonexistent_command"])

            assert return_code == 127
            # Verify error message was printed
            print_calls = [str(call_arg) for call_arg in mock_print.call_args_list]
            assert any(
                "Command failed" in str(call_arg)
                for call_arg in print_calls
            )

    @patch("subprocess.Popen")
    def test_subprocess_non_zero_exit_code(self, mock_popen: MagicMock) -> None:
        """Test subprocess with non-zero exit code."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["Exit code 2\n"])
        mock_process.wait.return_value = 2
        mock_popen.return_value = mock_process

        return_code = safe_run_subprocess(["exit", "2"])

        assert return_code == 2

    @patch("subprocess.Popen")
    def test_subprocess_long_output(self, mock_popen: MagicMock) -> None:
        """Test subprocess with long output."""
        long_output = [f"Line {i}\n" for i in range(100)]
        mock_process = MagicMock()
        mock_process.stdout = iter(long_output)
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with patch("builtins.print") as mock_print:
            return_code = safe_run_subprocess(["echo", "test"])

            assert return_code == 0
            # Verify all lines were printed
            assert mock_print.call_count == 100

    @patch("subprocess.Popen")
    def test_subprocess_with_special_characters(self, mock_popen: MagicMock) -> None:
        """Test subprocess output with special characters."""
        special_output = [
            "Unicode: ñ, é, ü\n",
            "Symbols: @#$%^&*()\n",
            "Quotes: \"single\" and 'double'\n",
        ]

        mock_process = MagicMock()
        mock_process.stdout = iter(special_output)
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with patch("builtins.print") as mock_print:
            return_code = safe_run_subprocess(["echo", "test"])

            assert return_code == 0
            # Verify special characters were handled
            for output_line in special_output:
                mock_print.assert_any_call(output_line, end="")

    @patch("subprocess.Popen")
    def test_subprocess_error_output_format(self, mock_popen: MagicMock) -> None:
        """Test error output format when subprocess fails."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["Error: something went wrong\n"])
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process

        with patch("builtins.print") as mock_print:
            return_code = safe_run_subprocess(["failing_command", "arg1", "arg2"])

            assert return_code == 1
            # Verify error message format
            print_calls = [str(call_arg) for call_arg in mock_print.call_args_list]
            combined_output = " ".join(print_calls)

            assert "Command failed" in combined_output
            assert "return code" in combined_output.lower()

    @patch("subprocess.Popen")
    def test_subprocess_return_code_propagated(
        self, mock_popen: MagicMock
    ) -> None:
        """Test that subprocess return code is properly propagated."""
        for exit_code in [0, 1, 2, 127, 255]:
            mock_process = MagicMock()
            mock_process.stdout = iter([])
            mock_process.wait.return_value = exit_code
            mock_popen.return_value = mock_process

            return_code = safe_run_subprocess(["test"])

            assert return_code == exit_code

    @patch("subprocess.Popen")
    def test_subprocess_text_mode_enabled(self, mock_popen: MagicMock) -> None:
        """Test that subprocess is called with text mode enabled."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        safe_run_subprocess(["test"])

        # Verify text=True is set
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["text"] is True

    @patch("subprocess.Popen")
    def test_subprocess_stderr_redirected_to_stdout(
        self, mock_popen: MagicMock
    ) -> None:
        """Test that stderr is redirected to stdout."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        safe_run_subprocess(["test"])

        # Verify stderr=subprocess.STDOUT
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["stderr"] == subprocess.STDOUT

    @patch("subprocess.Popen")
    def test_subprocess_stdout_piped(self, mock_popen: MagicMock) -> None:
        """Test that stdout is piped."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        safe_run_subprocess(["test"])

        # Verify stdout=subprocess.PIPE
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["stdout"] == subprocess.PIPE

    @patch("subprocess.Popen")
    def test_subprocess_success_without_message(self, mock_popen: MagicMock) -> None:
        """Test successful subprocess without success message."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["Output\n"])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with patch("builtins.print") as mock_print:
            return_code = safe_run_subprocess(["echo", "test"])

            assert return_code == 0
            # Should only print output, not success message
            calls_with_success = [
                c
                for c in mock_print.call_args_list
                if "Success" in str(c) or "success" in str(c)
            ]
            assert len(calls_with_success) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
