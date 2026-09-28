#!/usr/bin/env python3
"""Build the VCF phASER reads from the cohort VCF, so that it meets phASER's assumptions.

phASER (phaser.py, generate_mapping_table) calls a record an SNV when REF and every ALT are one base long.
On records merged with bcftools norm -m +any this admits '*' spanning-deletion alleles, which no read base
can match, and drops true SNVs that share a record with a longer ALT. XY donors also carry heterozygous
calls on hemizygous chrX and chrY. Steps, in order:
  1. a GT whose two alleles are different non-reference alleles (1|2, 1|*) is set to missing, because
     splitting would turn it into a REF/ALT heterozygote on each record;
  2. records are split to biallelic and normalized against the reference, ALT='*' records are dropped and
     only single-base REF and ALT records are kept;
  3. for XY donors, heterozygous GT is set to missing on chrX outside the PAR used by the RNA pipeline;
     for every donor, heterozygous GT is set to missing on chrY (haploid in XY, absent in XX).
Per-donor counts are printed and written to the facts JSON.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

VCF_DIR = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/vcf')
IN_VCF = VCF_DIR / 'cohort92.NC.vcf.gz'                  # SHAPEIT5-phased, 92 donors, merged with norm -m +any
OUT_VCF = VCF_DIR / 'cohort92.phaser_input.NC.vcf.gz'
FACTS = VCF_DIR / 'cohort92.phaser_input.facts.json'
CHR2NC = VCF_DIR / 'chr2nc.tsv'                          # UCSC contig name -> NC_ accession, 25 contigs
FASTA = '/mnt/ssd/lalli/phasing_T2T/chm13v2.0_maskedY_rCRS.fasta'  # the --fasta of IN_VCF's ##bcftools_normCommand
MANIFEST = ('/mnt/ssd/lalli/nf_stage/brainvar_eqtl_pc_tuning_20260722T173803/shared_prepared/prepared/common/'
            'sample_manifest.tsv')                       # SexForAnalysis by matchingDNALibrary, 225 donors
METADATA = '/mnt/ssd/lalli/nf_stage/draft_brainvar2_library_metadata_v1.4.tsv'  # metadata v1.4, WGS rows by LibraryID
RNA_SAMPLESHEET = ('/mnt/ssd/lalli/nf_stage/personalized_rnaseq/JLL_specific_files/test_data/'
                   'brainvar1+2.trimmed.samplesheet.with_metadata.csv')  # sex the RNA run built transcriptomes with
RNA_PAR_BED = '/mnt/ssd/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/T2T.rnaseqPAR.bed'  # par_bed of every
#   params_*.json of the BrainVar personalized_rnaseq run (personalized_T2T_NCBI110_pseudoalignment)
CHM13_PAR_BED = '/mnt/ssd/lalli/nf_stage/genome_refs/bedfiles/CHM13v2.0/XY/CHM13v2.0_PAR.bed'  # T2T-CHM13v2.0 PARs
CHRX, CHRY = 'NC_060947.1', 'NC_060948.1'
THREADS = 24
WORK = Path(os.environ['TMPDIR']) / 'phaser_input_vcf'   # TMPDIR must be on the SSD
PHASER_NON_SNV = 'strlen(REF)>1 || strlen(ALT)>1'        # true when any allele is longer than one base


def run(*cmd):
    """Run a command, echo its stderr, stop on failure; return the stderr."""
    cmd = [str(c) for c in cmd]
    print('+', ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(p.stderr)
    if p.returncode:
        sys.exit(f'failed with exit code {p.returncode}: {" ".join(cmd[:2])}')
    return p.stderr


def smpl_stats(vcf, *opts):
    """Per-donor heterozygous and missing genotype counts, and the number of records, from bcftools +smpl-stats."""
    out = subprocess.run(['bcftools', '+smpl-stats', *opts, str(vcf)], check=True, stdout=subprocess.PIPE,
                         text=True).stdout
    rows = [line.split('\t') for line in out.splitlines()]
    flt = {r[1]: (int(r[6]), int(r[11])) for r in rows if r[0] == 'FLT0'}   # column 7 het, 12 missing
    het = pd.Series({d: v[0] for d, v in flt.items()})
    missing = pd.Series({d: v[1] for d, v in flt.items()})
    (sites,) = [int(r[1]) for r in rows if r[0] == 'SITE0']
    return het.reindex(DONORS), missing.reindex(DONORS), sites


def header_contigs(vcf):
    return [line for line in subprocess.run(['bcftools', 'view', '-h', str(vcf)], check=True, stdout=subprocess.PIPE,
                                            text=True).stdout.splitlines() if line.startswith('##contig')]


DONORS = subprocess.run(['bcftools', 'query', '-l', str(IN_VCF)], check=True, stdout=subprocess.PIPE,
                        text=True).stdout.split()
print(f'input: {IN_VCF} ({len(DONORS)} donors)')
WORK.mkdir(exist_ok=True)

# Sex: manifest, then metadata v1.4 for donors the manifest lacks; both checked against v1.4 and the RNA run.
manifest = pd.read_csv(MANIFEST, sep='\t').set_index('matchingDNALibrary').SexForAnalysis
meta = pd.read_csv(METADATA, sep='\t').query('LibraryModality == "WGS"').set_index('LibraryID').SexForAnalysis
rna = pd.read_csv(RNA_SAMPLESHEET).drop_duplicates(['dna_id', 'sex']).set_index('dna_id').sex
assert manifest.index.is_unique and meta.index.is_unique and rna.index.is_unique
sex = manifest.reindex(DONORS)
from_meta = sex.index[sex.isna()].tolist()
sex = sex.fillna(meta.reindex(DONORS))
print(f'sex: {len(DONORS) - len(from_meta)} donors from the manifest, {from_meta} from metadata v1.4 '
      f'({", ".join(f"{d} {sex[d]}" for d in from_meta)})')
assert sex.isin(['XX', 'XY']).all(), sex[~sex.isin(['XX', 'XY'])]
disagree = sex.index[(sex != meta.reindex(DONORS)) | (sex != rna.reindex(DONORS))].tolist()
if disagree:
    sys.exit(f'sex disagrees between the manifest, metadata v1.4 and the RNA samplesheet for {disagree}')
XY = sex.index[sex == 'XY'].tolist()
print(f'sex: {len(XY)} XY, {len(DONORS) - len(XY)} XX; manifest, metadata v1.4 and RNA samplesheet agree')

# PAR: the RNA pipeline's BED, compared with the T2T-CHM13v2.0 PARs. BED is 0-based half-open.
rna_par = pd.read_csv(RNA_PAR_BED, sep='\t', header=None, names=['chrom', 'start', 'end'])
chm13_par = pd.read_csv(CHM13_PAR_BED, sep='\t', header=None, names=['chrom', 'start', 'end'])
print(f'PAR, RNA pipeline ({RNA_PAR_BED}):\n{rna_par.to_string(index=False)}')
print(f'PAR, T2T-CHM13v2.0 ({CHM13_PAR_BED}):\n{chm13_par.to_string(index=False)}')
assert rna_par.chrom.tolist() == ['chrX', 'chrX'] and rna_par.start[0] == 0
x_chm13 = chm13_par[chm13_par.chrom == 'chrX'].reset_index(drop=True)
y_chm13 = chm13_par[chm13_par.chrom == 'chrY'].reset_index(drop=True)
assert rna_par.end[1] == x_chm13.end[1], 'PAR2 ends differ, so chrX lengths differ'
PAR1_END, PAR2_START, X_END = int(rna_par.end[0]), int(rna_par.start[1]), int(rna_par.end[1])
REG = {  # 1-based closed regions for bcftools -r
    'chrX_par': f'{CHRX}:1-{PAR1_END},{CHRX}:{PAR2_START + 1}-{X_END}',
    'chrX_nonpar': f'{CHRX}:{PAR1_END + 1}-{PAR2_START}',
    'chrX_par_outside_chm13_par': f'{CHRX}:{x_chm13.end[0] + 1}-{PAR1_END},{CHRX}:{PAR2_START + 1}-{x_chm13.start[1]}',
    'chrY': CHRY,
}
print(f'PAR applied (1-based): {REG["chrX_par"]}; masked for XY: {REG["chrX_nonpar"]} and all of {CHRY}. '
      f'The RNA rule is wider than the T2T PAR by {PAR1_END - x_chm13.end[0]:,} bp at PAR1 and '
      f'{x_chm13.start[1] - PAR2_START:,} bp at PAR2 ({REG["chrX_par_outside_chm13_par"]}); it has no chrY rows '
      f'because the pipeline treats all of chrY as haploid in XY donors.')

# Input, as phASER sees it: every record, and records phASER calls SNVs (every allele one base, '*' included).
in_het, in_mis, n_in = smpl_stats(IN_VCF)
in_snv = WORK / 'input.phaser_snv.bcf'   # +smpl-stats -e segfaults in bcftools 1.22, so the subset is written first
run('bcftools', 'view', '--threads', THREADS, '-e', PHASER_NON_SNV, IN_VCF, '-Ob', '-o', in_snv)
ph_het, _, n_in_ph = smpl_stats(in_snv)
in_snv.unlink()
print(f'input: {n_in:,} records, {n_in_ph:,} of them SNVs by phASER\'s rule; {in_het.sum():,} het genotypes, '
      f'{ph_het.sum():,} at phASER-rule SNVs')

# Step 1: GT="Aa" is a heterozygote of two different ALT alleles.
A = WORK / 'step1.bcf'
err = run('bcftools', '+setGT', '--threads', THREADS, IN_VCF, '-Ob', '-o', A, '--', '-t', 'q', '-n', '.', '-i', 'GT="Aa"')
filled1 = int(re.search(r'Filled (\d+) alleles', err).group(1))
a_het, a_mis, n_a = smpl_stats(A)
alt_alt = in_het - a_het
assert n_a == n_in and (a_mis - in_mis).equals(alt_alt) and filled1 == 2 * alt_alt.sum()
print(f'step 1: {alt_alt.sum():,} ALT/ALT\' het genotypes set to missing ({filled1:,} alleles); records {n_a:,}')

# Step 2: norm -f is what trims a split SNV such as AGACCCT>TGACCCT to A>T; without it phASER still drops it.
# The FASTA uses UCSC names, so contigs are renamed to UCSC and back; the provenance header line is added then.
nc2chr = WORK / 'nc2chr.tsv'
pd.read_csv(CHR2NC, sep='\t', header=None)[[1, 0]].to_csv(nc2chr, sep='\t', header=False, index=False)
hdr = WORK / 'header.txt'
hdr.write_text(f'##phaserInput="scripts/phaser_input_vcf.py from {IN_VCF.name}: (1) GT with two different non-reference '
               f'alleles set to missing; (2) bcftools norm -m -any -f {Path(FASTA).name}, ALT=* records dropped, single-base '
               f'REF and ALT kept; (3) XY donors (SexForAnalysis): heterozygous GT set to missing on {REG["chrX_nonpar"]} '
               f'(outside the PAR of {Path(RNA_PAR_BED).name}); all donors: heterozygous GT set to missing on {CHRY}"\n')
A_chr, B_chr, B = WORK / 'step1.chr.bcf', WORK / 'step2.chr.bcf', WORK / 'step2.bcf'
run('bcftools', 'annotate', '--threads', THREADS, '--rename-chrs', nc2chr, A, '-Ob', '-o', A_chr)
err = run('bcftools', 'norm', '--threads', THREADS, '-f', FASTA, '-m', '-any', A_chr, '-Ob', '-o', B_chr)
norm_line = dict(zip(['total', 'split', 'joined', 'realigned', 'mismatch_removed', 'dup_removed', 'skipped'],
                     map(int, re.search(r'skipped:\s+(\S+)', err).group(1).split('/'))))
run('bcftools', 'annotate', '--threads', THREADS, '--rename-chrs', CHR2NC, '-h', hdr, B_chr, '-Ob', '-o', B)
A_chr.unlink()
B_chr.unlink()
b_het, b_mis, n_b = smpl_stats(B)
assert b_het.equals(a_het), 'split changed a donor\'s het count'
star_het, _, n_star = smpl_stats(B, '-i', 'ALT="*"')
C = WORK / 'step2.snv.vcf.gz'
run('bcftools', 'view', '--threads', THREADS, '-i', 'strlen(REF)==1 && strlen(ALT)==1 && ALT!="*"', B, '-Oz', '-o', C)
run('tabix', '-p', 'vcf', C)
c_het, c_mis, n_c = smpl_stats(C)
non_snv_het = a_het - star_het - c_het
n_non_snv = n_b - n_star - n_c
err = run('bcftools', 'norm', '-d', 'exact', '-m', '+snps', C, '-Ou', '-o', os.devnull)
shared = dict(zip(['total', 'split', 'joined', 'realigned', 'mismatch_removed', 'dup_removed', 'skipped'],
                  map(int, re.search(r'skipped:\s+(\S+)', err).group(1).split('/'))))
assert shared['dup_removed'] == 0, 'exact duplicate SNV records after the split'
print(f'step 2: split {norm_line["split"]:,} multiallelic records, realigned {norm_line["realigned"]:,}; records '
      f'{n_a:,} -> {n_b:,} after split -> {n_b - n_star:,} after dropping {n_star:,} ALT=* records -> {n_c:,} after '
      f'dropping {n_non_snv:,} non-SNV records. Het genotypes {a_het.sum():,} -> {a_het.sum() - star_het.sum():,} '
      f'-> {c_het.sum():,}. {shared["joined"]:,} SNV records share a position with another SNV record; 0 exact '
      f'duplicates.')

# Step 3. Sex is SexForAnalysis as given (user decision 2026-09-28): the chrX het ranges below overlap between
# the labels and are reported, not used to relabel donors.
before = {k: smpl_stats(C, '-r', r)[0] for k, r in REG.items()}
n_y_chm13_par = smpl_stats(C, '-r', ','.join(f'{CHRY}:{s + 1}-{e}' for s, e in zip(y_chm13.start, y_chm13.end)))[2]
n_boundary = smpl_stats(C, '-r', f'{CHRX}:{PAR1_END}-{PAR1_END},{CHRX}:{PAR2_START}-{PAR2_START}')[2]
assert n_y_chm13_par == 0, 'records inside the T2T chrY PAR; masking all of chrY would remove PAR calls'
xnon = before['chrX_nonpar']
xx_min, xy_max = xnon[sex == 'XX'].min(), xnon[sex == 'XY'].max()
print(f'step 3: chrX non-PAR het SNVs per donor before masking: XX {xx_min:,}-{xnon[sex == "XX"].max():,}, '
      f'XY {xnon[sex == "XY"].min():,}-{xy_max:,}; {CHRY} records inside the T2T chrY PAR: {n_y_chm13_par}; '
      f'records at the two positions the RNA pipeline puts in both its haploid and diploid regions: {n_boundary}')
xy_file = WORK / 'xy_donors.txt'
xy_file.write_text('\n'.join(XY) + '\n')
out_tmp = OUT_VCF.with_name(OUT_VCF.name.replace('.vcf.gz', '.tmp.vcf.gz'))
D = WORK / 'step3.x.bcf'
expr_x = f'GT[@{xy_file}]="het" && CHROM=="{CHRX}" && POS>{PAR1_END} && POS<={PAR2_START}'
err = run('bcftools', '+setGT', '--threads', THREADS, C, '-Ob', '-o', D, '--', '-t', 'q', '-n', '.', '-i', expr_x)
filled_x = int(re.search(r'Filled (\d+) alleles', err).group(1))
err = run('bcftools', '+setGT', '--threads', THREADS, D, '-Oz', '-o', out_tmp, '--', '-t', 'q', '-n', '.', '-i',
          f'GT="het" && CHROM=="{CHRY}"')
filled_y = int(re.search(r'Filled (\d+) alleles', err).group(1))
run('tabix', '-p', 'vcf', out_tmp)
os.replace(f'{out_tmp}.tbi', f'{OUT_VCF}.tbi')
os.replace(out_tmp, OUT_VCF)
o_het, o_mis, n_o = smpl_stats(OUT_VCF)
after = {k: smpl_stats(OUT_VCF, '-r', r)[0] for k, r in REG.items()}
masked = c_het - o_het
x_masked = before['chrX_nonpar'].where(sex == 'XY', 0)
assert n_o == n_c and masked.equals(x_masked + before['chrY'])
assert filled_x == 2 * x_masked.sum() and filled_y == 2 * before['chrY'].sum()
assert (after['chrY'] == 0).all() and after['chrX_nonpar'].where(sex == 'XY', 0).eq(0).all()
assert after['chrX_par'].equals(before['chrX_par'])
assert header_contigs(OUT_VCF) == header_contigs(IN_VCF), 'contig header lines changed'
assert subprocess.run(['bcftools', 'query', '-l', str(OUT_VCF)], check=True, stdout=subprocess.PIPE,
                      text=True).stdout.split() == DONORS, 'sample list changed'
print(f'step 3: set to missing {x_masked.sum():,} XY non-PAR chrX het genotypes and {before["chrY"].sum():,} chrY '
      f'het genotypes ({before["chrY"][sex == "XX"].sum():,} of them in XX donors); output {n_o:,} records, '
      f'{o_het.sum():,} het genotypes; contig lines and samples unchanged')

table = pd.DataFrame({
    'sex': sex,
    'het_input': in_het,
    'het_input_phaser_snv_rule': ph_het,
    'set_missing_alt_alt': alt_alt,
    'het_on_star_records': star_het,
    'het_on_non_snv_records': non_snv_het,
    'het_snv_before_sex_step': c_het,
    'chrX_par_het_before': before['chrX_par'],
    'chrX_par_outside_chm13_par_het_before': before['chrX_par_outside_chm13_par'],
    'chrX_nonpar_het_before': before['chrX_nonpar'],
    'chrY_het_before': before['chrY'],
    'set_missing_sex_step': masked,
    'chrX_par_het_after': after['chrX_par'],
    'chrX_nonpar_het_after': after['chrX_nonpar'],
    'chrY_het_after': after['chrY'],
    'het_snv_output': o_het,
})
table.index.name = 'donor'
with pd.option_context('display.width', 400, 'display.max_columns', None, 'display.max_rows', None):
    print(table.to_string())
    print(table.drop(columns='sex').groupby(table.sex).sum().to_string())

facts = {
    'script': 'scripts/phaser_input_vcf.py',
    'input_vcf': str(IN_VCF), 'output_vcf': str(OUT_VCF), 'fasta': FASTA,
    'sex_sources': {'manifest': MANIFEST, 'metadata_v1.4': METADATA, 'rna_samplesheet': RNA_SAMPLESHEET,
                    'from_metadata_v1.4': {d: sex[d] for d in from_meta}},
    'n_xy': len(XY), 'n_xx': len(DONORS) - len(XY),
    'par': {'rna_pipeline_bed': RNA_PAR_BED, 'rna_pipeline_bed_rows': rna_par.values.tolist(),
            'chm13v2_bed': CHM13_PAR_BED, 'chm13v2_bed_rows': chm13_par.values.tolist(), 'regions_1based': REG,
            'chrY_records_in_chm13_chrY_par': n_y_chm13_par, 'records_at_rna_pipeline_boundary_positions': n_boundary},
    'records': {'input': n_in, 'input_phaser_snv_rule': n_in_ph, 'after_split': n_b, 'star_dropped': n_star,
                'after_star_drop': n_b - n_star, 'non_snv_dropped': n_non_snv, 'output': n_o,
                'snv_records_sharing_a_position': shared['joined'], 'norm_split_line': norm_line},
    'genotypes': {'alt_alt_set_missing': int(alt_alt.sum()), 'alt_alt_alleles_filled': filled1,
                  'xy_chrX_nonpar_het_set_missing': int(x_masked.sum()), 'chrY_het_set_missing': int(before['chrY'].sum()),
                  'chrY_het_set_missing_in_xx': int(before['chrY'][sex == 'XX'].sum()),
                  'alleles_filled_chrX': filled_x, 'alleles_filled_chrY': filled_y},
    'totals_by_sex': json.loads(table.drop(columns='sex').groupby(table.sex).sum().to_json(orient='index')),
    'per_donor': json.loads(table.to_json(orient='index')),
}
facts_tmp = FACTS.with_suffix('.tmp')
facts_tmp.write_text(json.dumps(facts, indent=1) + '\n')
os.replace(facts_tmp, FACTS)
shutil.rmtree(WORK)
print(f'wrote {OUT_VCF} (+ .tbi) and {FACTS}')
