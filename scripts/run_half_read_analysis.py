"""Regenerate half-read tables and figures from the recorded benchmark inputs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from half_read_io import REPO, atomic_path, digest

TRIAL = 'half_read_trial_20260929'
SE = 'half_read_se_comparison_20260929'
PVALUES = 'half_read_pvalue_comparison_20260929'
POWER = 'half_read_unit_power_pr_20260929'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-root', required=True, type=Path,
                        help='read-only root of the recorded datasets and benchmark results')
    parser.add_argument('--output', required=True, type=Path,
                        help='new directory for regenerated tables, figures, and provenance')
    args = parser.parse_args()
    source, output = args.deploy_root.resolve(strict=True), args.output.resolve()
    if output.exists():
        parser.error(f'output already exists: {output}; choose a new directory')
    requirements = REPO / 'benchmark/plasmode/requirements.txt'
    pins = dict(line.split('==') for line in requirements.read_text().splitlines()
                if line and not line.startswith('#'))
    versions = {name: importlib.metadata.version(name) for name in pins}
    if versions != pins or platform.python_version() != '3.11.14':
        parser.error(f'use Python 3.11.14 and {requirements}; found {platform.python_version()}, {versions}')

    copies = []
    for stratum in ('deep', 'low'):
        copies.extend(Path(TRIAL) / stratum / name for name in (
            'per_gene_null.parquet', 'null_monte_carlo.tsv', 'independent_nb.parquet',
            'mapper_timing.tsv', 'manifest.json'))
        copies.extend(Path(SE) / f'half_read_extra_{stratum}.{extension}' for extension in ('parquet', 'json'))
        copies.extend(Path(PVALUES) / name for name in (
            f'half_read_pvalues_{stratum}.parquet', f'manifest_{stratum}.json'))
        copies.extend(Path(POWER) / name for name in (
            f'half_read_fixed_{stratum}.parquet', f'half_read_leads_{stratum}.parquet',
            f'manifest_{stratum}.json'))
    copies.append(Path('half_read_default_adoption_20260929/verification.json'))
    missing = [str(source / name) for name in copies if not (source / name).is_file()]
    if missing:
        parser.error('missing recorded inputs: ' + ', '.join(missing))
    output.mkdir(parents=True)
    sources = sorted((REPO / 'scripts').glob('half_read*.py')) + [
        Path(__file__).resolve(), REPO / 'scripts/unit_power_inputs.py', requirements]
    source_hashes = {str(p.relative_to(REPO)): digest(p) for p in sources}
    for path in sources:
        with atomic_path(output / 'source' / path.relative_to(REPO)) as temporary:
            shutil.copyfile(path, temporary)
    inputs = {}
    for name in copies:
        inputs[str(source / name)] = digest(source / name)
        with atomic_path(output / name) as temporary:
            shutil.copyfile(source / name, temporary)
    print(f'Copied {len(copies)} recorded input artifacts; original inputs remain unchanged', flush=True)

    env = os.environ | {'HALF_READ_DEPLOY_ROOT': str(source), 'HALF_READ_OUTPUT_ROOT': str(output)}
    stages = [
        ('half_read_score.py', '--output', output / TRIAL),
        ('half_read_report.py', '--root', output / TRIAL),
        ('half_read_se_mse_audit.py', '--output', output / 'half_read_mse_audit'),
        ('half_read_se_plot.py', '--output', output / SE),
        ('half_read_pvalue_plot.py', '--output', output / PVALUES),
        ('half_read_unit_power_pr.py', '--output', output / POWER),
    ]
    runtime = {'python': sys.version, 'executable': sys.executable,
               'platform': platform.platform(), 'packages': versions,
               'requirements_sha256': digest(requirements)}
    with atomic_path(output / 'runtime.json') as temporary:
        temporary.write_text(json.dumps(runtime, indent=2) + '\n')
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                            capture_output=True, text=True, check=True)
    with atomic_path(output / 'environment.txt') as temporary:
        temporary.write_text(result.stdout)
    commands = []
    for index, (script, flag, destination) in enumerate(stages, 1):
        command = [sys.executable, str(REPO / 'scripts' / script), flag, str(destination)]
        log = output / 'logs' / f'{index:02d}_{Path(script).stem}.log'
        print(f'{index}/{len(stages)} {script} -> {log}', flush=True)
        with atomic_path(log) as temporary, temporary.open('w') as handle:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{script} failed with exit {result.returncode}; see {log}; run is incomplete')
        commands.append(command)

    # This comparison is a regression check against saved results, not a calibration test.
    import pandas as pd
    compared = []
    for directory, names in (
        (TRIAL, ('causal_comparison.parquet',)),
        (SE, ('comparison_units.parquet', 'summary.tsv')),
        (PVALUES, ('pvalue_units.parquet', 'summary.tsv', 'paired_half_minus_split.tsv')),
        (POWER, ('baseline_fixed.parquet', 'baseline_leads.parquet', 'gene_design_units.parquet',
                 'comparison_units.parquet', 'comparison_summary.tsv', 'gene_discovery_units.parquet',
                 'power_summary.tsv', 'precision_recall_points.tsv', 'average_precision.tsv', 'null_gene_calls.tsv')),
    ):
        for name in names:
            relative = Path(directory) / name
            if relative.suffix == '.parquet':
                actual, expected = pd.read_parquet(output / relative), pd.read_parquet(source / relative)
            else:
                actual = pd.read_csv(output / relative, sep='\t')
                expected = pd.read_csv(source / relative, sep='\t')
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            compared.append({'file': str(relative), 'rows': len(actual),
                             'reference_sha256': digest(source / relative)})
    with atomic_path(output / POWER / 'baseline_acceptance.json') as temporary:
        temporary.write_text(json.dumps({'check': 'exact saved-table regression parity',
                                         'tables': compared}, indent=2) + '\n')
    if source_hashes != {str(p.relative_to(REPO)): digest(p) for p in sources}:
        raise RuntimeError('analysis sources changed during execution; use a fresh output directory')
    for name, command in (('git_state.txt', ['git', 'status', '--short', '--branch']),
                          ('source.patch', ['git', 'diff', 'HEAD'])):
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True)
        with atomic_path(output / name) as temporary:
            temporary.write_text(result.stdout)
    receipt = {'complete': True, 'scope': 'saved-input analysis; no regression or simulation scans',
               'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                          capture_output=True, text=True, check=True).stdout.strip(),
               'input_sha256': inputs, 'source_sha256': source_hashes,
               'commands': commands, 'known_answer_check': 'half_read_unit_power_pr.check_ranking passed',
               'exact_table_comparisons': compared,
               'output_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    with atomic_path(output / 'run_manifest.json') as temporary:
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Complete: {len(compared)} tables match the saved results exactly; {output / "run_manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
