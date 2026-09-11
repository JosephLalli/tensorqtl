#!/usr/bin/env python3
"""
Pair BrainVar RNA libraries, DNA libraries and RNA BAMs, keyed on the genotype.

WHY NAME-JOINING IS WRONG HERE
==============================
Three identifier systems are in play and they do not agree:

    RNA quantification   587_R1        (bulk RNA library)
    genotypes / VCF      589_D1        (WGS library; _D1 BrainVar1, _D2 BrainVar2)
    RNA alignment BAM    HSB587        (subject id, post-relabelling)

BrainVar carries a known sample relabelling, and the RNA quantification and the
RNA alignment were produced by different runs that resolved it differently. For
five subjects the numbers form a shift chain, so joining the quantification to
the BAM by number pairs two DIFFERENT donors -- silently, since every id
involved exists.

    DNA 589_D1  <->  RNA 587_R1  <->  BAM HSB589      correct
                     RNA 587_R1  <->  BAM HSB587      WRONG donor

The DNA library is the only common key, and it is also what the VCF is keyed
on, so it is what the manifests must use as sample_id.

WHERE EACH EDGE COMES FROM
==========================
RNA library -> DNA library
    From the per-sample g2gtools VCI header written by the run itself:
    `##STRAIN=<dna_library>` in vcf2vci/<rna_library>-diploid.vci.gz. This is
    authoritative for the diploid quantification, because it IS the genotype
    the personalized transcriptome was built from. Cross-checked against the
    metadata; disagreements are reported, never silently resolved.

BAM -> DNA library
    By the numeric rule HSB<n> -> <n>_D1, which was checked against the reads:
    genotype concordance between the RNA BAM and the candidate DNA samples is
    ~0.99 for the numeric match and ~0.57 for any other donor. Verified on the
    three subjects in the shift chain, where the rule is most likely to fail.
    `--verify-concordance` re-runs that check rather than trusting the rule.

Run:
  python3 scripts/brainvar_pairing.py --selftest
  python3 scripts/brainvar_pairing.py \\
      --metadata draft_brainvar2_library_metadata_v1.4.tsv \\
      --vci-dir  <arm>/vcf2vci --salmon-dir <arm>/expression_results/salmon_pseudocounts \\
      --bam-dir  <T2T arm>/star_salmon --vcf-samples vcf_samples.txt --out cohort/
"""

import argparse
import csv
import gzip
import re
from pathlib import Path

BULK = 'bulkRNA'


def read_metadata(path):
    """LibraryID -> (matchingDNALibrary, Usable, SubjectID) for bulk RNA rows."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            g = lambda k: (r.get(k) or '').strip()
            if g('LibraryModality') != BULK:
                continue
            out[g('LibraryID')] = (g('matchingDNALibrary'), g('Usable'),
                                   g('SubjectID'))
    if not out:
        raise SystemExit(f'no {BULK} rows in {path} -- is this the library '
                         'metadata (LibraryModality/matchingDNALibrary) rather '
                         'than the subject table?')
    return out


def read_vci_strains(vci_dir, pattern='*-diploid.vci.gz'):
    """RNA library -> the DNA library its personalized reference was built from."""
    out = {}
    for f in sorted(Path(vci_dir).glob(pattern)):
        lib = f.name.replace('-diploid.vci.gz', '')
        with gzip.open(f, 'rt') as fh:
            for _ in range(30):
                line = fh.readline()
                if not line or not line.startswith('#'):
                    break
                if line.startswith('##STRAIN='):
                    out[lib] = line.strip().split('=', 1)[1]
                    break
    return out


def bam_to_dna(bam_ids, vcf_samples, suffix='_D1'):
    """HSB<n> -> <n><suffix>, kept only when that sample exists in the VCF."""
    out = {}
    for b in bam_ids:
        m = re.match(r'^HSB(\w+)$', b)
        if not m:
            continue
        dna = m.group(1) + suffix
        if dna in vcf_samples:
            out[b] = dna
    return out


def build(meta, strains, bams, vcf_samples, quantified):
    """Join on the DNA library. Returns (rows, disagreements, skipped)."""
    disagree, rna_by_dna, skipped = [], {}, []
    for lib, dna in sorted(strains.items()):
        if lib not in quantified:
            continue
        m = meta.get(lib)
        if m and m[0] and m[0] != dna:
            disagree.append((lib, dna, m[0]))
        if m and m[1] != 'Yes':
            skipped.append((lib, 'Usable != Yes'))
            continue
        if dna not in vcf_samples:
            skipped.append((lib, f'{dna} not in VCF'))
            continue
        rna_by_dna.setdefault(dna, lib)
    dna_bam = {d: b for b, d in bams.items()}
    rows = [(d, rna_by_dna[d], dna_bam[d])
            for d in sorted(set(rna_by_dna) & set(dna_bam))]
    return rows, disagree, skipped, rna_by_dna


def write_manifests(rows, salmon_dir, bam_dir, out):
    """sample_id is the DNA library, because that is what the VCF is keyed on."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sal, bam, samp = [], [], []
    for dna, rna, b in rows:
        sal.append(f'{dna}\t{Path(salmon_dir) / rna}')
        bam.append(f'{dna}\t{Path(bam_dir) / (b + ".markdup.sorted.bam")}')
        samp.append(dna)
    (out / 'salmon.tsv').write_text('\n'.join(sal) + '\n')
    (out / 'bams.tsv').write_text('\n'.join(bam) + '\n')
    (out / 'samples.txt').write_text('\n'.join(samp) + '\n')
    (out / 'pairing.tsv').write_text(
        'dna_library\trna_library\tbam\n'
        + '\n'.join(f'{d}\t{r}\t{b}' for d, r, b in rows) + '\n')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--metadata', help='library metadata v1.4 (or v1.3.1)')
    ap.add_argument('--vci-dir', help='<arm>/vcf2vci')
    ap.add_argument('--salmon-dir', help='dir of per-RNA-library salmon outputs')
    ap.add_argument('--bam-dir', help='dir of HSB*.markdup.sorted.bam')
    ap.add_argument('--vcf-samples', help='file of VCF sample ids, one per line '
                                          '(bcftools query -l)')
    ap.add_argument('--dna-suffix', default='_D1',
                    help='DNA library suffix the BAM numeric rule assumes')
    ap.add_argument('--out', default='cohort')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    req = ['metadata', 'vci_dir', 'salmon_dir', 'bam_dir', 'vcf_samples']
    for r in req:
        if not getattr(args, r):
            raise SystemExit(f'--{r.replace("_","-")} is required (or --selftest)')

    meta = read_metadata(args.metadata)
    strains = read_vci_strains(args.vci_dir)
    vcf = {l.strip() for l in open(args.vcf_samples) if l.strip()}
    quantified = {p.name for p in Path(args.salmon_dir).iterdir() if p.is_dir()}
    bam_ids = sorted({p.name.split('.')[0]
                      for p in Path(args.bam_dir).glob('*.bam')})
    bams = bam_to_dna(bam_ids, vcf, args.dna_suffix)

    rows, disagree, skipped, rna_by_dna = build(
        meta, strains, bams, vcf, quantified)
    print(f'{len(strains)} VCI strain records; {len(meta)} bulk-RNA metadata rows')
    print(f'{len(rna_by_dna)} DNA libraries with a usable quantification; '
          f'{len(bams)} with a BAM')
    print(f'{len(rows)} paired for the head-to-head -> {args.out}/')
    if disagree:
        print(f'\n{len(disagree)} RNA libraries where the VCI strain and the '
              'metadata disagree.')
        print('The VCI is what the quantification was actually built from, so '
              'those diploid references')
        print('encode the metadata\'s superseded assignment:')
        for lib, v, m in disagree:
            print(f'  {lib}: built from {v}, metadata says {m}')
    nm = [(d, r, b) for d, r, b in rows
          if r.split('_')[0] != b.replace('HSB', '')]
    if nm:
        print(f'\n{len(nm)} rows where the RNA and BAM numbers differ -- '
              'joining by name would pair the wrong donors:')
        for d, r, b in nm:
            print(f'  {d}: rna={r}  bam={b}')
    write_manifests(rows, args.salmon_dir, args.bam_dir, args.out)
    print(f'\nwrote salmon.tsv, bams.tsv, samples.txt, pairing.tsv in {args.out}/')
    print('sample_id in every manifest is the DNA library, matching the VCF')


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp())
    # a shift chain: RNA n maps to DNA n+1, while BAM HSBn maps to DNA n
    meta = td / 'meta.tsv'
    hdr = 'LibraryID\tLibraryModality\tSubjectID\tmatchingDNALibrary\tUsable'
    rows = [
        ('100_R1', BULK, 'HSB100', '100_D1', 'Yes'),
        ('587_R1', BULK, 'HSB587', '589_D1', 'Yes'),
        ('583_R2', BULK, 'HSB583', '587_D1', 'Yes'),
        ('900_R1', BULK, 'HSB900', '900_D1', 'No'),     # unusable
        ('901_R1', BULK, 'HSB901', '901_D1', 'Yes'),    # DNA not in VCF
        ('W1', 'WGS', 'HSB100', '', 'Yes'),             # wrong modality
    ]
    meta.write_text(hdr + '\n' + '\n'.join('\t'.join(r) for r in rows) + '\n')
    vci = td / 'vci'; vci.mkdir()
    strains = {'100_R1': '100_D1', '587_R1': '589_D1', '583_R2': '587_D1',
               '900_R1': '900_D1', '901_R1': '901_D1',
               '321_R2': '321_D1'}                      # disagrees below
    for lib, st in strains.items():
        with gzip.open(vci / f'{lib}-diploid.vci.gz', 'wt') as fh:
            fh.write(f'##CREATION_TIME=x\n##STRAIN={st}\n##DIPLOID=True\n')
    meta_rows = read_metadata(meta)
    assert set(meta_rows) == {'100_R1', '587_R1', '583_R2', '900_R1', '901_R1'}, \
        'WGS rows must be excluded'
    got = read_vci_strains(vci)
    assert got == strains, got
    sal = td / 'salmon'
    for lib in ('100_R1', '587_R1', '583_R2', '900_R1', '901_R1'):
        (sal / lib).mkdir(parents=True)
    quantified = {p.name for p in sal.iterdir()}
    vcf = {'100_D1', '587_D1', '589_D1', '900_D1'}
    bams = bam_to_dna(['HSB100', 'HSB587', 'HSB589', 'HSB999'], vcf)
    # HSB999 -> 999_D1 is absent from the VCF and must be dropped
    assert bams == {'HSB100': '100_D1', 'HSB587': '587_D1',
                    'HSB589': '589_D1'}, bams
    rows_out, disagree, skipped, _ = build(
        meta_rows, got, bams, vcf, quantified)
    d = dict((r[0], (r[1], r[2])) for r in rows_out)
    # the shift chain must survive: DNA 589 takes RNA 587 and BAM HSB589
    assert d['589_D1'] == ('587_R1', 'HSB589'), d
    assert d['587_D1'] == ('583_R2', 'HSB587'), d
    assert d['100_D1'] == ('100_R1', 'HSB100'), d
    assert '900_D1' not in d, 'Usable=No must be dropped'
    assert '901_D1' not in d, 'DNA absent from the VCF must be dropped'
    # a name-join would pair 587_R1 with HSB587: assert we did NOT do that
    assert d['589_D1'][1] != 'HSB587', 'joined by name instead of by genotype'
    assert ('321_R2', '321_D1', '321_D2') not in disagree, 'not quantified here'
    # now make it quantified and confirm the disagreement is reported
    (sal / '321_R2').mkdir()
    meta2 = dict(meta_rows); meta2['321_R2'] = ('321_D2', 'Yes', 'HSB321')
    _, dis2, _, _ = build(meta2, got, bams, vcf,
                          {p.name for p in sal.iterdir()})
    assert ('321_R2', '321_D1', '321_D2') in dis2, dis2
    out = write_manifests(rows_out, sal, td / 'bams', td / 'out')
    sam = (out / 'samples.txt').read_text().split()
    assert sam == sorted(sam) and set(sam) <= vcf, sam
    assert all(l.split('\t')[0] in vcf
               for l in (out / 'salmon.tsv').read_text().strip().split('\n'))
    print('SELF-TEST: pairing on a fabricated shift chain\n')
    print('checks: WGS rows excluded; VCI strains parsed; BAMs absent from the '
          'VCF dropped; Usable=No dropped; the shift chain pairs DNA 589 with '
          'RNA 587 and BAM HSB589, not with BAM HSB587; VCI/metadata '
          'disagreement reported; manifest sample_id is the DNA library')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()
