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

class TestHapmixQTLDefaults:
    """The hapmixQTL CLI defaults are the validated ones."""

    def test_tau_mode_defaults_to_estimate(self):
        """tau_mode='zero' was the CLI default long after the library had
        moved to 'estimate' (docs/ase_validation.md: 'zero' is anticonservative
        by up to 107x at alpha = 1e-3). The CLI must not silently reproduce
        the defect."""
        from tensorqtl.tensorqtl import build_parser
        args = build_parser().parse_args(['geno', 'pheno.bed', 'out'])
        assert args.tau_mode == 'estimate'
        assert args.se_mode == 'model'
        # the allelic channel is fitted with an intercept only by default: the
        # 17-covariate set explains 24% of its whitened residual variance
        # against 22% expected by chance, while each column costs a sample
        assert args.ase_covariates == 'none'
        assert args.tau_refit is False
        assert build_parser().parse_args(['g', 'p', 'o', '--tau_refit']).tau_refit is True
        assert build_parser().parse_args(['g', 'p', 'o', '--tau_mode', 'zero']).tau_mode == 'zero'
        assert build_parser().parse_args(['g', 'p', 'o', '--ase_covariates', 'shared']).ase_covariates == 'shared'

    def test_help_documents_the_hapmixqtl_options(self):
        result = subprocess.run([
            sys.executable, '-m', 'tensorqtl', '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        assert result.returncode == 0
        assert '--tau_mode' in result.stdout and '--ase_covariates' in result.stdout


class TestRasqualRowParsing:
    """A malformed RASQUAL output line must not kill a multi-hour run."""

    def _row(self, pos='1000', chi2='12.5', pi='0.6', conv='0'):
        f = ['GENE', 'rs1', 'chr1', pos, 'A', 'G'] + ['0.5'] * 4
        f += [chi2, pi, '0.01', '0.5', '1.0']          # 10..14
        f += ['0'] * 7                                  # 15..21
        f += [conv] + ['0'] * 2                         # 22..24
        return '\t'.join(f)

    def test_interleaved_line_is_skipped_and_counted(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from compare_pipelines import best_rasqual_row
        good = self._row()
        # what actually killed a null round: a thread-interleaved line whose
        # position field holds an allele with a byte of binary on the end
        bad = self._row(pos='CCCGGCTGCCGCGTCTGGGAGGTGAGCGCC\udcc0')
        best, n = best_rasqual_row('\n'.join([bad, good, bad]), 'GENE')
        assert n == 2
        assert best is not None and best['stat'] == 12.5
        assert best['lead'] == 'chr1_1000_A_G'

    def test_short_and_skipped_lines_are_not_counted_as_malformed(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from compare_pipelines import best_rasqual_row
        lines = [self._row(), 'GENE\tSKIPPED', 'too\tshort']
        best, n = best_rasqual_row('\n'.join(lines), 'GENE')
        assert n == 0 and best['stat'] == 12.5

    def test_unconverged_rows_are_excluded_and_tested_set_respected(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from compare_pipelines import best_rasqual_row
        big_unconverged = self._row(pos='2000', chi2='99.0', conv='1')
        ok = self._row(pos='1000', chi2='12.5')
        best, n = best_rasqual_row('\n'.join([big_unconverged, ok]), 'GENE')
        assert n == 0 and best['stat'] == 12.5
        best2, _ = best_rasqual_row(ok, 'GENE', tested_pos={('chr1', 9999)})
        assert best2 is None
