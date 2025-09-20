"""
Tests for tensorQTL command-line interface.

This module tests the CLI functionality and argument parsing.
"""

import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run([
            sys.executable, '-m', 'tensorqtl', '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        assert result.returncode == 0
        assert 'tensorQTL' in result.stdout
        assert '--mode' in result.stdout

    def test_cli_version_info(self):
        """Test that CLI can be imported without errors."""
        result = subprocess.run([
            sys.executable, '-c', 'import tensorqtl; print("Import successful")'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        assert result.returncode == 0
        assert 'Import successful' in result.stdout

class TestCLIArgumentValidation:
    """Test CLI argument validation."""

    def test_missing_required_args(self):
        """Test CLI behavior with missing required arguments."""
        result = subprocess.run([
            sys.executable, '-m', 'tensorqtl'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        # Should fail due to missing required arguments
        assert result.returncode != 0
        assert 'required' in result.stderr or 'error' in result.stderr

    def test_invalid_mode(self):
        """Test CLI behavior with invalid mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy files
            dummy_genotype = os.path.join(temp_dir, "test")
            dummy_phenotype = os.path.join(temp_dir, "test.bed")

            # Create minimal files
            with open(dummy_phenotype, 'w') as f:
                f.write("#chr\tstart\tend\tgene_id\tS001\nchr1\t1000\t2000\tGENE1\t1.0\n")

            result = subprocess.run([
                sys.executable, '-m', 'tensorqtl',
                dummy_genotype, dummy_phenotype, 'output',
                '--mode', 'invalid_mode'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

            # Should fail due to invalid mode
            assert result.returncode != 0

class TestCLIModeSelection:
    """Test different CLI modes."""

    def test_mode_options(self):
        """Test that different modes are recognized."""
        valid_modes = ['cis', 'cis_nominal', 'cis_independent', 'trans']

        for mode in valid_modes:
            result = subprocess.run([
                sys.executable, '-m', 'tensorqtl', '--help'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

            assert mode in result.stdout

@pytest.mark.slow
class TestCLIWithTestData:
    """Test CLI with actual test data (if available)."""

    def test_cli_dry_run_simulation(self, test_data_dir):
        """Test CLI argument parsing with test data paths."""
        # This doesn't actually run analysis, just tests argument parsing
        expr_file = test_data_dir / "test_expression.bed.gz"
        cov_file = test_data_dir / "test_covariates.txt"

        if expr_file.exists() and cov_file.exists():
            # Test that CLI would accept these arguments
            # We'll simulate by checking help with these modes
            result = subprocess.run([
                sys.executable, '-m', 'tensorqtl', '--help'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

            # Should list the expected arguments
            assert '--covariates' in result.stdout
            assert '--mode' in result.stdout
            assert '--window' in result.stdout

class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_nonexistent_files(self):
        """Test CLI behavior with nonexistent input files."""
        result = subprocess.run([
            sys.executable, '-m', 'tensorqtl',
            'nonexistent_genotype_file',
            'nonexistent_phenotype_file',
            'output_prefix',
            '--mode', 'cis'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        # Should fail gracefully
        assert result.returncode != 0
        # Error message should be informative
        assert len(result.stderr) > 0 or len(result.stdout) > 0

    def test_invalid_file_format(self):
        """Test CLI behavior with invalid file formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with wrong format
            invalid_bed = os.path.join(temp_dir, "invalid.bed")
            with open(invalid_bed, 'w') as f:
                f.write("This is not a valid BED file\n")

            result = subprocess.run([
                sys.executable, '-m', 'tensorqtl',
                'dummy_genotype',
                invalid_bed,
                'output_prefix',
                '--mode', 'cis'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

            # Should fail due to file format issues
            assert result.returncode != 0