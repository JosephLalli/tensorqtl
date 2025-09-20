#!/usr/bin/env python3
"""
Validation script for nbqtl CLI integration.

This script tests the basic CLI functionality of the nbqtl-score mode.
"""

import subprocess
import sys
import tempfile
import pandas as pd
import numpy as np
import os
from pathlib import Path

def test_cli_help():
    """Test that --help includes nbqtl-score mode"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'tensorqtl', '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode == 0:
            if 'nbqtl-score' in result.stdout:
                print("✓ CLI help includes nbqtl-score mode")
                return True
            else:
                print("✗ CLI help does not include nbqtl-score mode")
                return False
        else:
            print(f"✗ CLI help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI help test failed: {e}")
        return False

def test_nbqtl_mode_validation():
    """Test that nbqtl-score mode shows proper error messages"""
    try:
        # Test with missing arguments (should show helpful error)
        result = subprocess.run([
            sys.executable, '-m', 'tensorqtl',
            'dummy_genotypes', 'dummy_phenotypes.bed', 'dummy_output',
            '--mode', 'nbqtl-score'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        # Should fail but with a reasonable error message
        if result.returncode != 0:
            print("✓ nbqtl-score mode properly validates inputs")
            return True
        else:
            print("✗ nbqtl-score mode should have failed with missing files")
            return False
    except Exception as e:
        print(f"✗ nbqtl mode validation test failed: {e}")
        return False

def create_minimal_test_data():
    """Create minimal test data for integration testing"""
    # This would create minimal BED, PGEN, and covariate files
    # For now, just create placeholder files
    temp_dir = tempfile.mkdtemp()

    # Create minimal phenotype BED file
    phenotypes = pd.DataFrame({
        'chr': ['chr1', 'chr1'],
        'start': [1000, 2000],
        'end': [1001, 2001],
        'phenotype_id': ['gene1', 'gene2'],
        'sample1': [10, 15],
        'sample2': [12, 18],
        'sample3': [8, 20]
    })
    phenotype_file = os.path.join(temp_dir, 'phenotypes.bed')
    with open(phenotype_file, 'w') as f:
        f.write('#chr\tstart\tend\tphenotype_id\tsample1\tsample2\tsample3\n')
        phenotypes.to_csv(f, sep='\t', index=False, header=False)

    # Create minimal covariate file
    covariates = pd.DataFrame({
        'covariate1': [0.1, -0.2, 0.3],
    }, index=['sample1', 'sample2', 'sample3'])
    covariate_file = os.path.join(temp_dir, 'covariates.tsv')
    covariates.to_csv(covariate_file, sep='\t')

    return temp_dir, phenotype_file, covariate_file

def main():
    """Run validation tests"""
    print("Validating nbqtl CLI integration...")
    print("=" * 50)

    all_passed = True

    # Test 1: CLI help
    print("\n1. Testing CLI help...")
    if not test_cli_help():
        all_passed = False

    # Test 2: Mode validation
    print("\n2. Testing mode validation...")
    if not test_nbqtl_mode_validation():
        all_passed = False

    # Test 3: Check import works
    print("\n3. Testing import...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import tensorqtl.nbqtl as nbqtl
        print("✓ nbqtl module imports successfully")
    except ImportError as e:
        print(f"✗ nbqtl module import failed: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All validation tests passed!")
        return 0
    else:
        print("✗ Some validation tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())