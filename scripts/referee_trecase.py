"""Real-data referee, TReCASE discovery: asSeq's trecase on the observed 92-donor data for the referee genes
(referee_replication.py wrote the gene order, the tested variants and the other arms), twice.

INPUTS. (salmon) 05_run_trecase.py's construction on the unpermuted, unthinned records: Y = the Salmon point-estimate
totals pT as they are (fractional), Y1, Y2 = rint(pL), rint(pR) on the records common.allelic_kept admits (Va from
summaries_from_point_estimates on the Gibbs draws, as referee_replication.py computed it) and 0 elsewhere, offset =
log(eff_lib), the edgeR effective library size of the Salmon totals. (native) the alignment-based integer counts of
native_counts.py: Y = featureCounts fragments, Y1, Y2 = phASER's haplotype counts a, b as written (a on the analysis
VCF's first GT allele = xL = asSeq's haplotype 1; phASER counts only heterozygous sites, so a record with one
haplotype at 0 is an observation here and allelic_kept's zero-haplotype rule, a Salmon rule, is not applied), offset =
log of the edgeR effective library size of the native counts, built as the Salmon one was
(run_hapmixqtl_from_salmon.edger_normalize: every gene of the featureCounts output, filterByExpr, the cache's
restrict_calibration.txt, TMM). Both: X = the RNA-tied covariates and the genotype PCs of every other arm (the
expression PCs among them come from Salmon log2 CPM), Z = 3 xL + xR, asSeq's defaults except p.cut (run_trecase.R,
unchanged). Donor columns are joined on the DNA library id.

GENOTYPES. One Z table per gene, holding its tested variants (referee_replication.py's rule: phased biallelic SNPs of
the analysis VCF within CM.WIN of the TSS at MAF >= CM.MAF over the 92 donors, present in the 225-donor store, i.e.
variant_map.tsv.gz rows with a store_id), rebuilt here and checked against referee_order.tsv's n_tested and against
the gibbs arm's nominal rows for every gene run; asSeq's window (|ePos - mPos| <= 1e6) keeps all of them.

SUBSET. The first TIMING_GENES genes of the order are run on both inputs, JOBS R processes at a time, and timed; all
genes are run if max-parallel wall time (both inputs' process-seconds per gene x genes / JOBS) fits BUDGET_H, else the
longest prefix of whole BLOCK-gene units that does (the order is referee_replication.py's seeded random order, so a
prefix is a seeded random subset). The scoring applies the same prefix to every arm. Written to
discovery/trecase_subset.json.

SALMON INPUT DROPPED (user decision 2026-09-28). After the timing block fixed the subset, the full run was stopped
with 689 Salmon-input genes finished and restarted on the native input alone (RUN_INPUTS), JOBS R processes, every
gene of the subset submitted in descending order of its tested variants so the largest genes do not straggle at the
end. The Salmon input costs about five times the native one (530 against 107 process-seconds per gene in the timing
block) and the plasmode benchmark already compares TReCASE on Salmon-derived input. The timing block's Salmon results
(discovery/trecase_salmon/) and every finished per-gene checkpoint stay on disk; they are not scored.

FAILURES. A gene whose Rscript exits non-zero is recorded (<tag>_failed.txt, with the log's last lines) and skipped
on a rerun; it has no rows. A gene whose <tag>_status.tsv exists is not rerun.

Output: discovery/trecase_<input>/nominal_<lo>_<hi>.parquet (05_run_trecase.convert's columns; parquet metadata the
sha256 of the input files and 'log2'), discovery/trecase_summary.json (counts per input), discovery/trecase_subset.json;
trecase_work/ (inputs, asSeq files, one trace log per gene; native_edger/ the native library normalization).
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'plasmode'))
import common as C                                               # noqa: E402
import referee_replication as RR                                 # noqa: E402
import run_hapmixqtl_from_salmon as H                            # noqa: E402
from tensorqtl.hapmixqtl import summaries_from_point_estimates   # noqa: E402

T = C.module('05_run_trecase')   # the plasmode's TReCASE input writers, R call and output conversion
CM = C.CM

OUT = RR.OUT
DISC = OUT / 'discovery'
WORK = OUT / 'trecase_work'
ORDER = OUT / 'genes' / 'referee_order.tsv'   # referee_replication.py
VARIANT_MAP = OUT / 'variant_map.tsv.gz'      # referee_replication.py: store_id set = tested
NATIVE = C.D / 'native_counts_wasp_20260928'   # native_counts.py: totals / hap_a / hap_b parquet, featurecounts/<donor>.txt
RESTRICT = RR.CACHE / 'point_estimates' / 'restrict_calibration.txt'   # the gene list the Salmon eff_lib was normalized on
INPUTS = ('salmon', 'native')   # the timing block's
RUN_INPUTS = ('native',)          # the subset's full run (user decision 2026-09-28, module docstring)
TIMING_JOBS = 46         # R processes of the timing block (Rscript execs into R: one process per gene), the subset rule's
JOBS = 96                # R processes of the full run (coordinator, 2026-09-28: up to 96 once the Salmon input was dropped)
TIMING_GENES = 100       # task
BUDGET_H = 6.0           # task: at most 6 h at 48 processes, else a seeded random subset
BLOCK = 100              # genes per input directory and per output file
R_ENV = {**T.R_ENV, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}   # CLAUDE.md BLAS entry, for the R processes only (the
                         # loader's bcftools needs this process's LD_LIBRARY_PATH); one thread per R process


def native_offset(order):
    """log edgeR effective library size of the native featureCounts totals, in `order` (DNA library ids)."""
    d = WORK / 'native_edger'
    if not (d / 'edger_samples.tsv').exists():
        samples = (RR.CACHE / 'samples.txt').read_text().split()
        cols = {}
        for s in samples:
            f = pd.read_csv(NATIVE / 'featurecounts' / f'{s}.txt', sep='\t', comment='#', index_col=0)
            cols[s] = f.iloc[:, -1]
        tot = pd.DataFrame(cols)
        pool = pd.read_parquet(NATIVE / 'totals.parquet')
        if not tot.index.is_unique or not (tot.loc[pool.index, pool.columns].values == pool.values).all():
            raise SystemExit('featureCounts gene rows repeat, or differ from native_counts.py\'s totals.parquet')
        print(f'native featureCounts: {tot.shape[0]:,} genes x {tot.shape[1]} donors, equal to totals.parquet on its '
              f'{len(pool):,} genes', flush=True)
        H.edger_normalize(tot, RESTRICT.read_text().split(), d)
    eff, _ = H.read_edger_dir(d, order)
    return np.log(eff), eff


def prepare(t):
    """Inputs for the genes of table t (a prefix of the order): the loader dict, Salmon arrays, tested rows per gene."""
    genes = list(t.gene)
    reg = WORK / f'regions_{len(genes)}.bed'
    reg.parent.mkdir(parents=True, exist_ok=True)
    RR.write_regions(t, reg)
    one = WORK / 'loader_gene.txt'
    C.write_atomic(one, lambda fh: fh.write(genes[0] + '\n'), 'w')
    t0 = time.perf_counter()
    I = CM.load_point_estimate_inputs(gene_list=str(one), regions=str(reg))
    gi = {g: i for i, g in enumerate((RR.CACHE / 'genes.txt').read_text().split())}
    rows, keep = np.array([gi[g] for g in genes]), I['keep']
    P = {k: np.asarray(np.load(Path(CM.PE) / f'{k}.npy', mmap_mode='r')[rows])[:, keep] for k in ('pL', 'pR', 'pT')}
    Y = {k: np.asarray(np.load(RR.CACHE / f'{k}.npy', mmap_mode='r')[rows])[:, keep] for k in ('YL', 'YR', 'YT')}
    eff_lib = I['eff_lib'][keep]
    _, _, Va, _, _ = summaries_from_point_estimates(P['pL'], P['pR'], P['pT'], eff_lib, Y['YL'], Y['YR'], Y['YT'])
    del Y
    if not (I['gp'].loc[genes, 'pos'].values == t.tss.values).all():
        raise SystemExit('loader gene positions differ from referee_order.tsv')
    vm = pd.read_csv(VARIANT_MAP, sep='\t', usecols=['variant_id', 'store_id'], dtype=str)
    in_store = set(vm.variant_id[vm.store_id.notna()])
    vdf = I['vdf']
    af = I['dos'].mean(1) / 2.0
    cand = np.where((np.minimum(af, 1 - af) >= CM.MAF) & vdf.index.astype(str).isin(in_store))[0]
    win = RR.window_rows(vdf, cand, I['gp'], genes)
    nt = np.array([len(win[g]) for g in genes])
    if not (nt == t.n_tested.values).all():
        raise SystemExit(f'tested variants rebuilt here differ from referee_order.tsv for {int((nt != t.n_tested.values).sum())} genes')
    print(f'loaded {len(genes):,} genes x {len(I["order"])} donors; {len(vdf):,} SNPs in the regions; tested per gene equal to '
          f'referee_order.tsv for every gene ({int(nt.sum()):,} pairs); {time.perf_counter() - t0:.0f} s', flush=True)
    return I, dict(pL=P['pL'], pR=P['pR'], pT=P['pT'], Va=Va, eff_lib=eff_lib), win


def block_setup(I, win, genes):
    """common.setup over the genes' tested rows (referee_replication.block)."""
    rows = np.unique(np.concatenate([win[g] for g in genes]))
    Ib = {**I, 'genes': genes, 'vdf': I['vdf'].iloc[rows], 'dos': I['dos'][rows], 'xL': I['xL'][rows],
          'xR': I['xR'][rows], 'idx': np.arange(len(rows))}
    S = C.setup(Ib)
    if [int(S['n_tested'][g]) for g in genes] != [len(win[g]) for g in genes]:
        raise SystemExit('common.setup\'s tested counts differ from the windows')
    return S


def write_gene_genotypes(S, g):
    """<WORK>/genotypes/<gene>/chr<k>.Z.bin and .markers.tsv: the gene's tested variants with varying ALT dosage."""
    d = WORK / 'genotypes' / g
    c = T.chrom_int(S['gp'].loc[g, 'chr'])
    ids = sorted(S['scanned'][g], key=lambda v: S['rows'][v])
    if not (d / f'chr{c}.markers.tsv').exists():
        r = np.array([S['rows'][v] for v in ids])   # rows of S['I'], whose idx is the identity here
        I = S['I']
        d.mkdir(parents=True, exist_ok=True)
        T.write_bin(d / f'chr{c}.Z.bin', 3 * I['xL'][r].astype(np.int64) + I['xR'][r].astype(np.int64))
        m = pd.DataFrame(dict(variant_id=ids, chr=c, pos=I['vdf'].pos.values[r].astype(int)))
        C.write_atomic(d / f'chr{c}.markers.tsv', lambda fh: m.to_csv(fh, sep='\t', index=False), 'w')
    return d, ids


def write_native(S, nat, off, lo, hi, d):
    """Y, Y1, Y2, X, offset, genes.tsv of the native input for genes lo:hi (05.write_dataset's layout)."""
    I = S['I']
    X = np.column_stack([I['cov_df'].values, I['geno_cov_df'].values])
    d.mkdir(parents=True, exist_ok=True)
    for name, a in dict(Y=nat['Y'][lo:hi], Y1=nat['Y1'][lo:hi], Y2=nat['Y2'][lo:hi], X=X.T, offset=off).items():
        T.write_bin(d / f'{name}.bin', a.astype(np.float64))
    tab = pd.DataFrame(dict(gene=S['genes'], chr=[T.chrom_int(S['gp'].loc[g, 'chr']) for g in S['genes']],
                            pos=[int(S['gp'].loc[g, 'pos']) for g in S['genes']]))
    C.write_atomic(d / 'genes.tsv', lambda fh: tab.to_csv(fh, sep='\t', index=False), 'w')


def run_gene(ddir, gdir, g, tag):
    """One Rscript process unless a status or failed file exists: (wall seconds or None, state)."""
    if Path(f'{tag}_status.tsv').exists():
        return None, 'done'
    if Path(f'{tag}_failed.txt').exists():
        return None, 'failed'
    t0 = time.perf_counter()
    with open(f'{tag}.log', 'w') as fh:
        rc = subprocess.run(['Rscript', str(T.R_RUNNER), str(ddir), str(gdir), g, str(tag)], stdout=fh,
                            stderr=subprocess.STDOUT, env={**os.environ, **R_ENV}).returncode
    if rc != 0:
        tail = ''.join(open(f'{tag}.log').readlines()[-8:])
        C.write_atomic(Path(f'{tag}_failed.txt'), lambda fh: fh.write(f'exit {rc}\n{tail}'), 'w')
        print(f'{g}: Rscript exited {rc}; recorded in {tag}_failed.txt:\n{tail}', flush=True)
        return time.perf_counter() - t0, 'failed'
    return time.perf_counter() - t0, 'ran'


def sha_inputs(d, arm):
    h = hashlib.sha256(arm.encode())
    for f in ('Y.bin', 'Y1.bin', 'Y2.bin', 'X.bin', 'offset.bin', 'genes.tsv'):
        h.update((d / f).read_bytes())
    return h.hexdigest()


def gibbs_rows(lo, hi):
    """Tested pairs per gene in the gibbs arm's nominal file of genes lo:hi (referee_replication.py's discovery)."""
    n, _ = RR.paths('gibbs', lo, hi)
    return pd.read_parquet(n, columns=['phenotype_id']).groupby('phenotype_id').size()


def submit_blocks(ex, I, ds, nat, off, win, t, blocks, inputs):
    """Write every block's inputs, then submit every (input, gene) in descending order of the gene's markers;
    {(input, lo, hi): (ddir, [(gene, markers, expected, future)])} in block order."""
    specs = []
    for lo, hi in blocks:
        genes = list(t.gene[lo:hi])
        S = block_setup(I, win, genes)
        g_rows = gibbs_rows(lo, hi)
        if not all(g_rows[g] == S['n_tested'][g] for g in genes):
            raise SystemExit(f'genes {lo}-{hi}: tested counts differ from the gibbs arm\'s nominal rows')
        dsb = {k: (v[lo:hi] if k in ('pL', 'pR', 'pT', 'Va') else v) for k, v in ds.items()}
        dsb['perm'] = np.arange(len(I['order']))
        geno = {g: write_gene_genotypes(S, g) for g in genes}
        for inp in inputs:
            ddir = WORK / inp / f'{lo:05d}_{hi:05d}'
            if inp == 'salmon':
                T.write_dataset(S, dsb, ddir)
            else:
                write_native(S, nat, off, lo, hi, ddir)
            (ddir / 'out').mkdir(parents=True, exist_ok=True)
            specs += [(inp, lo, hi, ddir, g, geno[g][0], geno[g][1], S['scanned'][g]) for g in genes]
    fut = {(s[0], s[4]): ex.submit(run_gene, s[3], s[5], s[4], s[3] / 'out' / s[4])
           for s in sorted(specs, key=lambda s: -len(s[6]))}
    jobs = {}
    for inp, lo, hi, ddir, g, _, markers, expected in specs:
        jobs.setdefault((inp, lo, hi), (ddir, []))[1].append((g, markers, expected, fut[(inp, g)]))
    print(f'submitted {len(specs):,} (input, gene) jobs, largest first (markers per gene max / median '
          f'{max(len(s[6]) for s in specs):,} / {int(np.median([len(s[6]) for s in specs])):,})', flush=True)
    return jobs


def collect(jobs):
    """Wait for each block, convert asSeq's rows, write one parquet per (input, block); per-input counts and timings."""
    counts = {inp: dict(genes=0, genes_failed=[], genes_resumed=0, rows=0, final_joint=0, final_trec=0, final_na=0, joint_na=0,
                        trec_na=0, ase_na=0, y_fail_baseline=0, as_records_admitted=0, trace=dict.fromkeys(T.FAILS, 0),
                        seconds=[], trecase_seconds=[]) for inp in dict.fromkeys(k[0] for k in jobs)}
    t0 = time.perf_counter()
    for (inp, lo, hi), (ddir, items) in jobs.items():
        parts, c = [], counts[inp]
        for g, markers, expected, fut in items:
            secs, state = fut.result()
            c['genes_resumed'] += state in ('done', 'failed') and secs is None
            if state == 'failed':
                c['genes_failed'].append(g)
                c['seconds'] += [round(secs, 2)] if secs is not None else []
                continue
            d, st, log = T.convert(g, ddir / 'out' / g, markers, expected)
            parts.append(d)
            c['trecase_seconds'].append(round(float(st.seconds), 2))
            c['seconds'].append(round(secs if secs is not None else float(st.seconds), 2))   # Rscript wall, else asSeq's own
            c['y_fail_baseline'] += int(st.yFailBaselineModel != 0)
            for k, s in T.FAILS.items():
                c['trace'][k] += log.count(s)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        y1 = np.fromfile(ddir / 'Y1.bin')
        y2 = np.fromfile(ddir / 'Y2.bin')
        c['as_records_admitted'] += int(((y1 + y2) >= T.MIN_AS_READS).sum())
        c['genes'] += len(items)
        if len(df):
            c['rows'] += len(df)
            c['final_joint'] += int((df.final_stat == 'joint').sum())
            c['final_trec'] += int((df.final_stat == 'trec').sum())
            c['final_na'] += int(df.pval_nominal.isna().sum())
            c['joint_na'] += int((~df.joint_ok).sum())
            c['trec_na'] += int(df.pval_t.isna().sum())
            c['ase_na'] += int(df.pval_a.isna().sum())
            C.write_parquet(df, DISC / f'trecase_{inp}' / f'nominal_{lo:05d}_{hi:05d}.parquet', sha_inputs(ddir, f'trecase_{inp}'),
                            'log2')
        print(f'{inp} genes {lo}-{hi}: {len(df):,} rows from {len(parts)} genes, failed {len(items) - len(parts)}; final p '
              f'joint / TReC / NA {int((df.final_stat == "joint").sum()) if len(df) else 0} / '
              f'{int((df.final_stat == "trec").sum()) if len(df) else 0} / {int(df.pval_nominal.isna().sum()) if len(df) else 0}; '
              f'allelic records with Y1 + Y2 >= {T.MIN_AS_READS}: {int(((y1 + y2) >= T.MIN_AS_READS).sum()):,} of {y1.size:,}; '
              f'{(time.perf_counter() - t0) / 60:.1f} min', flush=True)
    return counts


def inputs_for(t):
    """Salmon arrays and native counts (genes of t x donors in loader order)."""
    I, ds, win = prepare(t)
    order = list(I['order'])
    nat = {k: pd.read_parquet(NATIVE / f'{f}.parquet').loc[list(t.gene), order].to_numpy()
           for k, f in (('Y', 'totals'), ('Y1', 'hap_a'), ('Y2', 'hap_b'))}
    if (nat['Y1'] + nat['Y2'] > nat['Y']).any() or min(v.min() for v in nat.values()) < 0:
        raise SystemExit('native counts: a + b above the total, or a negative count')
    return I, ds, win, nat


def main():
    threadpool_limits(2)
    t = pd.read_csv(ORDER, sep='\t', dtype={'chr': str})
    sub = DISC / 'trecase_subset.json'
    ex = cf.ThreadPoolExecutor(JOBS)
    try:
        if not sub.exists():   # timing block: both inputs at TIMING_JOBS processes
            tt = t.iloc[:TIMING_GENES]
            I, ds, win, nat = inputs_for(tt)
            off, eff = native_offset(list(I['order']))
            ratio = eff / ds['eff_lib']
            print(f'native / Salmon effective library size per donor min / median / max {ratio.min():.3f} / '
                  f'{np.median(ratio):.3f} / {ratio.max():.3f}', flush=True)
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(TIMING_JOBS) as tex:
                c = collect(submit_blocks(tex, I, ds, nat, off, win, tt, [(0, TIMING_GENES)], INPUTS))
            wall = time.perf_counter() - t0
            per = {inp: float(np.mean(c[inp]['seconds'])) for inp in INPUTS}
            both = sum(per.values())
            proj = len(t) * both / TIMING_JOBS / 3600
            n_run = len(t) if proj <= BUDGET_H else int(BUDGET_H * 3600 * TIMING_JOBS / both) // BLOCK * BLOCK
            rec = dict(rule=f'the first {TIMING_GENES} genes of referee_order.tsv run on both inputs with {TIMING_JOBS} R processes; all '
                            f'{len(t)} genes if process-seconds per gene (both inputs) x genes / {TIMING_JOBS} <= {BUDGET_H} h, else the '
                            f'longest prefix of whole {BLOCK}-gene units that fits',
                       process_seconds_per_gene=per, timing_wall_seconds=wall, jobs=TIMING_JOBS, projected_hours_all=proj,
                       genes_all=len(t), n_genes=n_run, subset=n_run < len(t), genes_file=str(ORDER),
                       seconds_source='Rscript wall per gene; asSeq\'s own seconds for a gene finished by an earlier call',
                       native_over_salmon_eff_lib=[float(ratio.min()), float(np.median(ratio)), float(ratio.max())])
            C.write_json(sub, rec)
            print(f'TIMING: {TIMING_GENES} genes, process-seconds per gene salmon {per["salmon"]:.0f}, native {per["native"]:.0f} '
                  f'(timing block wall {wall / 60:.1f} min at {TIMING_JOBS} processes); all {len(t):,} genes on both inputs would take '
                  f'{proj:.1f} h; budget {BUDGET_H} h -> the first {n_run:,} genes of the order', flush=True)
        n_run = json.loads(sub.read_text())['n_genes']
        tt = t.iloc[:n_run]
        I, ds, win, nat = inputs_for(tt)
        off, _ = native_offset(list(I['order']))
        t0 = time.perf_counter()
        counts = collect(submit_blocks(ex, I, ds, nat, off, win, tt, [(lo, min(lo + BLOCK, n_run)) for lo in range(0, n_run, BLOCK)],
                                       RUN_INPUTS))
    finally:
        ex.shutdown(cancel_futures=True)
    for inp, c in counts.items():
        ws = c['trecase_seconds']
        print(f'{inp}: {c["genes"]} genes ({c["genes_resumed"]} finished by an earlier call), {c["rows"]:,} rows, failed {c["genes_failed"]}, asSeq baseline model not used (allelic, total-count or both) in '
              f'{c["y_fail_baseline"]} genes; final p joint / TReC / NA {c["final_joint"]:,} / {c["final_trec"]:,} / {c["final_na"]:,}; '
              f'joint p missing {c["joint_na"]:,}; trace {c["trace"]}; asSeq seconds per gene median {np.median(ws):.0f}, total '
              f'{np.sum(ws) / 3600:.1f} h', flush=True)
    C.write_json(DISC / 'trecase_summary.json', dict(n_genes=n_run, per_input=counts, jobs=JOBS,
                                                      wall_hours=(time.perf_counter() - t0) / 3600))
    print(f'wrote {DISC / "trecase_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
