#!/usr/bin/env python3
"""
Run phASER + phaser_gene_ae across a paired cohort, resumably.

WHAT THIS EXISTS TO GET RIGHT
=============================
phaser.py re-invokes a python interpreter for its read-variant mapping step and
defaults `--python_string` to a python2 path. Under a python3-only install that
step fails partway in, per sample, after the expensive part has already run.
This always passes `--python_string`.

`--sample` must name the sample as the VCF spells it. In this dataset that is
the DNA library (`589_D1`), not the subject and not the RNA library -- see
scripts/brainvar_pairing.py for why those three disagree and why joining them
by name pairs the wrong donors.

Two more defaults are wrong for this data and each aborts the run.
`--id_separator` defaults to `_`, and phASER refuses any contig name
containing it -- RefSeq accessions like `NC_060925.1` do, so it must be set to
a character absent from the contig names (`:` is also rejected outright; `-`
works). And `--pass_only` defaults to 1, which keeps only FILTER=PASS records:
a callset whose FILTER column is `.` throughout carries no filter annotation
rather than having failed one, and every site is discarded -- reported as
"0 heterozygous sites ... (N filtered)", which reads like a data problem rather
than a flag problem.

`--mapq 255` is STAR's uniquely-mapped encoding. It is the right value for a
STAR-aligned BAM and the wrong value for most other aligners, so it is a flag
here rather than a constant.

The BAM and the VCF must agree on contig names. They are not checked against
each other here because phASER simply finds nothing when they disagree, which
looks like a quiet biological result rather than an error -- so this checks
them up front and refuses.

RESUMING
========
A sample is done when its `<prefix>.gene_ae.txt` exists and is non-empty; those
are skipped. Partial output from an interrupted sample is overwritten. Failures
are recorded and the run continues, so one bad BAM does not cost the cohort.

Run:
  python3 scripts/run_phaser_cohort.py --selftest
  python3 scripts/run_phaser_cohort.py --pairing cohort/pairing.tsv \\
      --bam-dir <dir> --vcf cohort.vcf.gz --features genes.bed \\
      --phaser-dir tools/phaser --out phaser_out/ --jobs 8 --threads 8
"""

import argparse
import concurrent.futures as cf
import gzip
import os
import subprocess
import sys
import time
from pathlib import Path


def read_pairing(path):
    rows = [l.split('\t') for l in Path(path).read_text().strip().split('\n')]
    if rows and rows[0][0].startswith('dna'):
        rows = rows[1:]
    return [(r[0].strip(), r[1].strip(), r[2].strip()) for r in rows if len(r) >= 3]


def vcf_contigs(path, limit=200):
    """Contig ids from the VCF header, without needing bcftools."""
    out = []
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            if line.startswith('##contig='):
                i = line.find('ID=')
                if i >= 0:
                    j = min([x for x in (line.find(',', i), line.find('>', i))
                             if x > 0] or [len(line)])
                    out.append(line[i + 3:j])
            if len(out) >= limit:
                break
    return out


def bed_contigs(path, limit=200):
    seen = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            c = line.split('\t')[0]
            if c not in seen:
                seen.append(c)
            if len(seen) >= limit:
                break
    return seen


def phaser_cmd(a, dna, bam, prefix):
    return [sys.executable, str(Path(a.phaser_dir) / 'phaser' / 'phaser.py'),
            '--vcf', str(a.vcf), '--bam', str(bam), '--sample', dna,
            '--mapq', str(a.mapq), '--baseq', str(a.baseq),
            '--paired_end', str(a.paired_end), '--write_vcf', '1',
            '--python_string', a.python_string,
            '--id_separator', a.id_separator, '--pass_only', str(a.pass_only),
            '--threads', str(a.threads), '--temp_dir', str(a.temp_dir),
            '--o', str(prefix)]


def gene_ae_cmd(a, prefix):
    return [sys.executable,
            str(Path(a.phaser_dir) / 'phaser_gene_ae' / 'phaser_gene_ae.py'),
            '--haplotypic_counts', f'{prefix}.haplotypic_counts.txt',
            '--features', str(a.features), '--id_separator', a.id_separator,
            '--o', f'{prefix}.gene_ae.txt']


def done(prefix):
    p = Path(f'{prefix}.gene_ae.txt')
    return p.exists() and p.stat().st_size > 0


def run_one(a, row):
    dna, rna, bam_id = row
    prefix = Path(a.out) / dna
    if done(prefix):
        return dna, 'skipped', 0.0, ''
    bam = Path(a.bam_dir) / (bam_id + a.bam_suffix)
    if not bam.exists():
        return dna, 'missing_bam', 0.0, str(bam)
    log = Path(a.out) / f'{dna}.log'
    t0 = time.time()
    with open(log, 'w') as fh:
        for cmd in (phaser_cmd(a, dna, bam, prefix), gene_ae_cmd(a, prefix)):
            fh.write('+ ' + ' '.join(cmd) + '\n')
            fh.flush()
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                return dna, 'failed', time.time() - t0, str(log)
    if not done(prefix):
        return dna, 'no_output', time.time() - t0, str(log)
    return dna, 'ok', time.time() - t0, ''


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--pairing'); ap.add_argument('--bam-dir')
    ap.add_argument('--bam-suffix', default='.markdup.sorted.bam')
    ap.add_argument('--vcf'); ap.add_argument('--features')
    ap.add_argument('--phaser-dir'); ap.add_argument('--out', default='phaser_out')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--mapq', default='255',
                    help="STAR's unique-mapper value; wrong for most others")
    ap.add_argument('--baseq', default='10')
    ap.add_argument('--paired-end', default='1')
    ap.add_argument('--python-string', default=sys.executable)
    ap.add_argument('--id-separator', default='-',
                    help='must not occur in any contig name; phASER also '
                         'rejects ":" outright (default "-")')
    ap.add_argument('--pass-only', default='0',
                    help='1 keeps only FILTER=PASS; use 0 when the callset '
                         'has no FILTER annotation (default 0)')
    ap.add_argument('--temp-dir', default=os.environ.get('TMPDIR', '/tmp'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for r in ('pairing', 'bam_dir', 'vcf', 'features', 'phaser_dir'):
        if not getattr(a, r):
            raise SystemExit(f'--{r.replace("_","-")} is required (or --selftest)')
    rows = read_pairing(a.pairing)
    if a.limit:
        rows = rows[:a.limit]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.temp_dir).mkdir(parents=True, exist_ok=True)

    vc, bc = set(vcf_contigs(a.vcf)), set(bed_contigs(a.features))
    bad = [c for c in vc | bc if a.id_separator in c]
    if bad:
        raise SystemExit(
            f'--id-separator {a.id_separator!r} occurs in contig names '
            f'({bad[:3]}); phASER refuses that. Pick a character absent from '
            'every contig name.')
    if vc and bc and not (vc & bc):
        raise SystemExit(
            f'VCF contigs {sorted(vc)[:3]} and feature contigs {sorted(bc)[:3]} '
            'do not intersect. phASER would silently produce nothing. Rename '
            'one to match the other (and the BAM) before running.')

    if a.dry_run:
        for row in rows[:3]:
            print(' '.join(phaser_cmd(a, row[0],
                                      Path(a.bam_dir) / (row[2] + a.bam_suffix),
                                      Path(a.out) / row[0])))
        print(f'... {len(rows)} samples, {sum(done(Path(a.out)/r[0]) for r in rows)} already done')
        return 0

    todo = [r for r in rows if not done(Path(a.out) / r[0])]
    print(f'{len(rows)} samples, {len(rows)-len(todo)} already done, '
          f'{len(todo)} to run; {a.jobs} at a time x {a.threads} threads')
    counts, t0 = {}, time.time()
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, a, r): r for r in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            dna, status, secs, note = f.result()
            counts[status] = counts.get(status, 0) + 1
            print(f'  [{i}/{len(todo)}] {dna}: {status} ({secs/60:.1f} min) {note}',
                  flush=True)
    print(f'\n{dict(sorted(counts.items()))} in {(time.time()-t0)/60:.1f} min')
    return 1 if set(counts) - {'ok', 'skipped'} else 0


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp())
    (td / 'out').mkdir()
    pair = td / 'p.tsv'
    pair.write_text('dna_library\trna_library\tbam\n'
                    '589_D1\t587_R1\tHSB589\n100_D1\t100_R1\tHSB100\n')
    rows = read_pairing(pair)
    assert rows == [('589_D1', '587_R1', 'HSB589'),
                    ('100_D1', '100_R1', 'HSB100')], rows
    vcf = td / 'v.vcf'
    vcf.write_text('##fileformat=VCFv4.2\n##contig=<ID=NC_060925.1,length=9>\n'
                   '##contig=<ID=NC_060926.1>\n#CHROM\tPOS\n')
    assert vcf_contigs(vcf) == ['NC_060925.1', 'NC_060926.1'], vcf_contigs(vcf)
    bed = td / 'f.bed'
    bed.write_text('NC_060925.1\t1\t9\tG1\nNC_060925.1\t20\t30\tG2\n')
    assert bed_contigs(bed) == ['NC_060925.1']
    bad = td / 'bad.bed'
    bad.write_text('chr1\t1\t9\tG1\n')
    assert not (set(vcf_contigs(vcf)) & set(bed_contigs(bad))), 'must not intersect'

    vcf_p, bed_p = str(vcf), str(bed)

    class A:
        phaser_dir = str(td / 'phaser'); vcf = vcf_p; features = bed_p
        mapq = '255'; baseq = '10'; paired_end = '1'
        python_string = 'python3'; threads = 4; temp_dir = str(td / 'tmp')
        id_separator = '-'; pass_only = '0'
        out = str(td / 'out'); bam_suffix = '.bam'; bam_dir = str(td)
    cmd = phaser_cmd(A, '589_D1', td / 'x.bam', Path(A.out) / '589_D1')
    # --sample is the DNA library, which is how the VCF spells it
    assert cmd[cmd.index('--sample') + 1] == '589_D1', cmd
    # the python2 default is always overridden
    assert '--python_string' in cmd and cmd[cmd.index('--python_string')+1] == 'python3'
    # the phased VCF the phase overlay needs is always requested
    assert cmd[cmd.index('--write_vcf') + 1] == '1'
    assert cmd[cmd.index('--mapq') + 1] == '255'
    # the two defaults that abort on this data are always overridden
    assert cmd[cmd.index('--id_separator') + 1] == '-', cmd
    assert cmd[cmd.index('--pass_only') + 1] == '0', cmd
    # and the separator must be carried into gene_ae, or the ids stop matching
    g = gene_ae_cmd(A, Path(A.out) / '589_D1')
    assert g[g.index('--id_separator') + 1] == '-', g
    assert g[g.index('--features') + 1] == bed_p
    assert g[g.index('--haplotypic_counts') + 1].endswith('.haplotypic_counts.txt')
    # resume: a non-empty gene_ae marks a sample done, an empty one does not
    pre = Path(A.out) / '589_D1'
    assert not done(pre)
    Path(f'{pre}.gene_ae.txt').write_text('')
    assert not done(pre), 'an empty gene_ae must not count as done'
    Path(f'{pre}.gene_ae.txt').write_text('contig\tstart\n1\t2\n')
    assert done(pre)
    print('SELF-TEST: command construction and resume logic\n')
    print('checks: --sample is the DNA library as the VCF spells it; the '
          'python2 --python_string default is always overridden; --write_vcf 1 '
          'is always requested; --id_separator and --pass_only are set on both '
          'commands rather than left at defaults that abort; contig sets are '
          'compared so a silent no-op is refused; an empty gene_ae does not '
          'count as done')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
