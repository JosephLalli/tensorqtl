"""HTML report of the benchmark (README: Report): one self-contained page from 06_score's summary,
the check files, the run facts, TReCASE's summary, the stored null of this pipeline (NULL) and the
mixQTL ladder, with four figures (embedded as base64 and written as PNG beside it). It computes
nothing 06_score.py did not, except the positions of points in the figures and small arithmetic on
stored values (range overlaps, differences and ratios). Section 5 also reads one earlier run
(EARLIER, the same pipeline on the log2(CPM + 1) total) and names what separates it. Every
comparative sentence is guarded where it is written (need) or in check_claims, so the page stops
when a run makes one false. For a gene set other than INTERPRETED_SET the interpretation paragraphs
of section 3, section 3.8 and sections 4 and 5 are left out, a closing limits section is added, and
a section after section 3 sets the set against the deep set's run (REF_RUN) with its selection
(SELECT_LOG, POOL), the transcriptome-wide stratum rates (STRATA), the Salmon half-depth test
(HALF_DEPTH) and the committed run's dataset blocks (COMMITTED_RUN_LOG), in three contrast figures
and their tables; every input the set lacks is skipped with a printed line.
"""
import base64
import datetime
import html
import json
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.ticker          # noqa: E402

import common as C                # noqa: E402
from tensorqtl.hapmixqtl import MIN_ALLELIC_DONORS   # noqa: E402

SC = C.module('06_score')
NULL = C.STORED_NULL         # the stored 200-permutation null of the hapmixQTL arms on this pipeline (scripts/half_read_stored_null.py)
EARLIER = C.D / 'release_closure_20261001' / 'plasmode_after_split' / 'summary.json'   # 2026-10-01, same counts, PCs and code, log2(CPM + 1) total (section 5)
OUT, PAGE = C.REPORT, C.REPORT / 'plasmode_report.html'
INTERPRETED_SET = 'corrected_null_store_20260925'   # the gene set the interpretation prose (section 3 paragraphs, 3.8, 4-6, check_claims) was written for
INTERPRETED = C.GENE_SET == INTERPRETED_SET
REF_RUN = C.D / C.GENE_SETS[INTERPRETED_SET]['root'] / 'summary.json'   # the deep set's run of this code: the contrast on any other gene set's page (task 2026-09-27)
FIRST_RUN = C.D / C.GENE_SETS[INTERPRETED_SET]['committed']            # the first simulated-effects run, on the deep set (a contrast page's section 1)
REF_GENES = C.D / C.GENE_SETS[INTERPRETED_SET]['gene_dir'] / 'genes.txt'
SELECT_LOG = C.GENE_DIR / 'select_stratum_genes.log'   # a stratum set's selection counts (select_stratum_genes.py)
POOL = C.GENE_DIR / 'pool_stratum.tsv'                 # every eQTL-filter gene's median admitted reads (select_stratum_genes.py)
STRATA = C.D / 'coupling_reach_20260925' / 'b_strata.tsv'      # transcriptome-wide allelic null rate by coverage stratum, pre-correction pipeline
HALF_DEPTH = C.D / 'salmon_half_depth_20260927' / 'summary.json'   # scripts/salmon_half_depth_check.py: the thinning rule against Salmon at half depth
COMMITTED_RUN_LOG = C.D / C.GS['committed'] / 'run_arms.log'   # the committed run's arms log: its dataset blocks (the head of a non-default page)
ARMS = C.ARMS
HAPMIX, JOINT, TQ = C.HAPMIX_ARMS, tuple(C.JOINT), C.TENSORQTL
HM = HAPMIX + tuple(C.MIXQTL_ARMS)   # the hapmixQTL and mixQTL arms, the arms BEFORE scored
ALL = ARMS + JOINT                   # the Salmon-input arms: the prose and check_claims are about these
JOINT_COV = C.D / 'cov' / 'log2cpm1_point_calibration_20260925'   # the covariates of the committed runs whose RASQUAL and TReCASE results are reused (3aac315), of the stored null runs and of their re-run under 8a06803 (01_check_inputs.REPRO_COV); all predate a2f4314
HALF_READ_SINCE = datetime.datetime(2026, 9, 30, 15, 36, 9, tzinfo=datetime.timezone.utc)   # commit a2f4314: the expression PCs move to the half-read build
LIMITS = 'section 6' if INTERPRETED else 'the limits section'   # where the page lists what would run to make a comparison like for like


def arms_cov(root):
    """The covariate build a run's hapmixQTL, mixQTL and tensorQTL arms used, and how that is known: 03_run_arms.py's record in
    run_arms_facts.json, or, for a run made before 03 recorded it, the build of the code of the date that file was written."""
    facts = root / 'results' / 'run_arms_facts.json'
    rec = json.loads(facts.read_text()).get('covariates')
    if rec:
        return Path(rec), 'recorded by 03_run_arms.py'
    ran = datetime.datetime.fromtimestamp(facts.stat().st_mtime, datetime.timezone.utc)
    return (C.COV if ran >= HALF_READ_SINCE else JOINT_COV), f'inferred from when 03_run_arms.py wrote its run_arms_facts.json, {ran.date()}'


ARMS_COV, ARMS_COV_HOW = arms_cov(C.ROOT)
REF_COV, REF_COV_HOW = arms_cov(REF_RUN.parent)
SAME_COV = ARMS_COV == JOINT_COV     # whether this run's arms used the covariates of the joint models and of the stored null runs
STORED_PCS = '' if SAME_COV else ", and its total channel was fitted on the log2(CPM + 1) build's expression principal components"


def cov_table(d):
    """covariates.tsv of a covariate build: {column: values as floats}, in donor order."""
    rows = [line.split('\t') for line in (d / 'covariates.tsv').read_text().splitlines()]
    return {name: [float(r[i]) for r in rows[1:]] for i, name in enumerate(rows[0]) if i > 0}


COV_DIFF = [] if SAME_COV else [c for c, v in cov_table(JOINT_COV).items() if cov_table(ARMS_COV)[c] != v]


def joint_run(model):
    """A joint model's results in this directory: (the covariate build they used, whether they were run here). 04 and 05
    record the build in their summary.json; results staged from the committed run carry no record and that run's build."""
    rec = json.loads((C.JOINT[model] / 'summary.json').read_text()).get('covariates')
    return (Path(rec), True) if rec else (JOINT_COV, False)


JOINT_NAME = {'rasqual': 'RASQUAL', 'trecase': 'TReCASE'}
JOINT_RUN = {m: joint_run(m) for m in JOINT}
JOINT_HERE = [m for m in JOINT if JOINT_RUN[m][1]]           # run into this directory
STAGED = [m for m in JOINT if not JOINT_RUN[m][1]]           # staged from the committed run
JOINT_OFF = [m for m in JOINT if JOINT_RUN[m][0] != ARMS_COV]   # on another covariate build than this run's arms
if set(JOINT_OFF) - set(STAGED):
    raise SystemExit(f'{JOINT_OFF}: a joint model run here on another covariate build than the arms; reword section 2')
names = lambda ms: ' and '.join(JOINT_NAME[m] for m in ms)   # noqa: E731
JOINT_DATE = {m: datetime.date.fromtimestamp((C.JOINT[m] / 'summary.json').stat().st_mtime).isoformat() for m in JOINT_HERE}
JOINT_HEAD = '; '.join(
    ([f'{names(STAGED)} of 2026-09-27, reused from {C.COMMITTED} because Meier\'s correction does not touch '
      + ('it' if len(STAGED) == 1 else 'them') + (" and on that run's covariates (section 2)" if JOINT_OFF else '')] if STAGED else [])
    + [f'{JOINT_NAME[m]} run {JOINT_DATE[m]} into {C.ROOT} on this run\'s covariates' for m in JOINT_HERE])
COV_OF = {m: 'those of the other arms' if m not in JOINT_OFF else "the committed run's (end of this paragraph)" for m in JOINT}
JOINT_COV_PARA = '' if not JOINT_OFF else (
    (' Both joint models\' results are' if len(JOINT_OFF) == 2 else f' {names(JOINT_OFF)}\'s results are')
    + f' reused from {C.COMMITTED.name}, the committed run on these datasets, made with the covariates of {JOINT_COV.name}; '
    + (f'{names(JOINT_HERE)} was run here and, like the other arms, uses' if JOINT_HERE else 'the other arms use')
    + f' {ARMS_COV.name} ({ARMS_COV_HOW}). The two builds differ only in {len(COV_DIFF)} of '
    f'the 17 columns ({COV_DIFF[0]} to {COV_DIFF[-1]}), the expression principal components, which the committed run\'s '
    'build computes on log2(CPM + 1); the clinical covariates and the genotype principal components are identical.')
JOINT_COV_NOTE = '' if not JOINT_OFF else (
    (' Not like for like: the joint models ran on the committed run\'s covariates ' if len(JOINT_OFF) == 2 else
     f' Not like for like for {names(JOINT_OFF)}, which ran on the committed run\'s covariates ')
    + ('(sections 2 and 6).' if INTERPRETED else '(section 2 and the limits section).'))
NATIVE = SC.NATIVE_ARMS              # 05b_native_arms.py: split weighting and TReCASE on native alignment counts; () where C.NATIVE does not exist
SHOWN = ALL + NATIVE                 # every arm in the section 3 tables and Figures 1-4
LABEL = {'split': 'split (shipped: 1/Va allelic, unit total)', 'gibbs': 'gibbs (1/Va allelic, 1/Vt total)',
         'unit': 'unit (weight 1 both channels)',
         'mixqtl': 'mixQTL, published cutoffs', 'mixqtl_permissive': 'mixQTL, permissive cutoffs',
         TQ: 'tensorQTL, total only, unweighted', 'rasqual': 'RASQUAL (joint model)', 'trecase': 'TReCASE, asSeq (joint model)',
         'split_native': 'split on native counts (control)', 'trecase_native': 'TReCASE, asSeq, on native counts'}
SHORT = {'gibbs': 'gibbs', 'split': 'split', 'unit': 'unit', 'mixqtl': 'mixQTL pub.',
         'mixqtl_permissive': 'mixQTL perm.', TQ: 'tensorQTL', 'rasqual': 'RASQUAL', 'trecase': 'TReCASE',
         'split_native': 'split native', 'trecase_native': 'TReCASE native'}
COLOR = dict(zip(ALL + C.NATIVE_ARMS, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#8a5a2b', '#4a3aa7',
                                       '#e34948', '#222222', '#7a1016')))   # native arms: near-black, and a darker red than TReCASE's
MARKER = dict(zip(ALL + C.NATIVE_ARMS, 'osD^vPpXh>d'))
BETA_COLOR = {'0.2': '#86b6ef', '0.4': '#2a78d6', '0.8': '#104281'}
BETAS = ('0.2', '0.4', '0.8')
BANDS = tuple(b[0] for b in SC.BANDS)   # 06_score's read bands for this gene set
BAND_HTML = ' / '.join(b.replace('>=', '&ge;').replace('<', '&lt;') for b in BANDS[1:])   # table headers and prose
NO_ONE_DF = SC.NO_ONE_DF
CHANNELS = ('combined', 'allelic', 'total')
MIX_CH = {'combined': 'meta', 'allelic': 'asc', 'total': 'trc'}
ALPHAS = ('0.05', '0.01', '0.001')
DETECT = ('0.05', '0.001', '1e-05')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
LOG_TICKS = (0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 4, 8, 16, 32)
S = SN = CG = CP = LF = JF = LD = SF = NF = None   # the inputs, set once by load()



def skipped(what, key):
    print(f'{what} skipped: gene set {C.GENE_SET} has none (common.GENE_SETS[{C.GENE_SET!r}][{key!r}] is None)', flush=True)


def load():
    global S, SN, CG, CP, LF, JF, LD, SF, NF
    S = json.loads(C.SUMMARY.read_text())
    if tuple(S['native_arms']) != NATIVE:
        raise SystemExit(f'{C.SUMMARY}: native arms {S["native_arms"]} differ from this script\'s {NATIVE}')
    if NATIVE:
        NF = dict(facts=json.loads((C.NATIVE / 'facts.json').read_text()),
                  trecase=json.loads((C.NATIVE_RESULTS['trecase_native'] / 'summary.json').read_text()),
                  counts=json.loads((C.NATIVE_COUNTS / 'facts.json').read_text()))
    else:
        print(f'native-input arms {list(C.NATIVE_ARMS)} skipped (tables, figures, their subsection and the sentences that cite '
              f'it): {C.NATIVE} does not exist', flush=True)
    SN = json.loads((NULL / 'summary.json').read_text()) if NULL else skipped('the stored null (anchor, null rates)', 'stored_null')
    if tuple(S['arms']) != ARMS or tuple(S['joint_arms']) != JOINT or tuple(S['bands']) != BANDS:
        raise SystemExit(f'{C.SUMMARY}: arms {S["arms"]} + {S["joint_arms"]} / bands {S["bands"]} differ from this script\'s')
    if SN is not None and (SN['floor'] != MIN_ALLELIC_DONORS or set(HAPMIX) - set(SN['rates']) or SN['n_draw'] != SC.ANCHOR_N_PERM):
        raise SystemExit(f'{NULL}: floor {SN["floor"]}, configurations {list(SN["rates"])} or {SN["n_draw"]} draws differ from this script\'s')
    CG, CP = (json.loads((C.CHECKS / f).read_text()) for f in ('check_generator.json', 'salmon_premise.json'))
    LD = json.loads((C.LADDER / 'ladder.json').read_text()) if C.LADDER else skipped('the mixQTL ladder (section 3.8)', 'ladder')
    LF = run_facts(json.loads((C.DATASETS / 'meta.json').read_text())['facts'],
                   json.loads((C.RESULTS / 'run_arms_facts.json').read_text()))
    JF = joint_facts(json.loads((C.JOINT['trecase'] / 'summary.json').read_text()))
    if SN is not None and LF['floor'][1] != sorted(SN['below_floor_genes']):
        raise SystemExit(f'below-floor genes {LF["floor"][1]} differ from {NULL} {sorted(SN["below_floor_genes"])}')
    SF = stratum_facts()
    if INTERPRETED:
        check_claims()
    else:
        print(f'interpretation prose and its fixed comparative claims skipped: written for the {INTERPRETED_SET} run, not gene '
              f'set {C.GENE_SET}; the page carries the tables and figures without them', flush=True)


def run_facts(mf, RF):
    """Counts of the generator and of the arms' runs (02's meta.json facts, 03's run_arms_facts.json, including tensorQTL's t
    against unit weights' total channel)."""
    runs = RF['runs'].values()
    zeroed = [v[a]['zeroed'] for v in runs for a in HAPMIX]
    floor = {(len(v[a]['below_floor']), tuple(v[a]['below_floor'])) for v in runs for a in HAPMIX}
    if INTERPRETED and len(floor) != 1:   # the interpretation names one set of genes below the floor
        raise SystemExit(f'the genes below the allelic floor differ between datasets or arms: {floor}')
    sets = sorted((n, list(names)) for n, names in floor)
    mix = {a: [(v[a]['genes_asc_ge_cutoff'], v[a]['genes_trc_ge_cutoff'], v[a]['asc_median']) for v in runs] for a in C.MIXQTL_ARMS}
    lib = mf['library_size_change']
    return dict(zeroed=(min(zeroed), max(zeroed)), mix=mix, tdiff=RF['tensorqtl_vs_unit_total_t'],
                pairs=tuple(f'{mf[k]:,}' for k in ('pairs', 'pairs_informative', 'pairs_expressible')),
                tested=[f'{x:,}' for x in mf['tested_per_gene']],
                expr=(f'{mf["expressible_share_het_nonnull"]:.3f}', lib['median'], lib['max']),
                floor=sets[0] if len(sets) == 1 else None, floor_sets=sets)


def joint_facts(TS):
    """TReCASE's run counts, pooled over datasets, from its summary.json; the allelic records the hapmixQTL arms admit
    (common.allelic_kept on each dataset), which asSeq receives."""
    n_ds = sum(S['n_datasets'].values())
    if len(TS['per_dataset']) != n_ds:
        raise SystemExit(f'TReCASE summary covers {len(TS["per_dataset"])} datasets, want {n_ds}')
    per = TS['per_dataset'].values()
    admitted = {}
    for k in TS['per_dataset']:
        sc, r = k.split(' rep ')
        ds = np.load(C.DATASETS / sc / f'rep{r}.npz')
        admitted[k] = int(C.allelic_kept(ds['pL'], ds['pR'], ds['Va']).sum())
    drop = [admitted[k] - c['as_records_admitted'] for k, c in TS['per_dataset'].items()]   # admitted records asSeq's min.AS.reads drops
    trace = lambda k: sum(c['joint_na_by_trace'][k] for c in per)   # noqa: E731
    return dict(trecase=dict(TS['pooled'], linear_dosage=trace('trec_linear_dosage'), ase_fail=trace('ase'),
                             few_het=sum(c['ase_na_few_het'] for c in per), df_not_1=sum(c['final_df_not_1'] for c in per),
                             constant=sorted({c['tested_constant_dosage'] for c in per}),
                             theta_gradient_max=max(c['joint_na_by_trace']['theta_fail_abs_gradient_max'] for c in per),
                             zeroed=(min(c['informative_zeroed_not_allelic_kept'] for c in per),
                                     max(c['informative_zeroed_not_allelic_kept'] for c in per)),
                             admitted=(min(admitted.values()), max(admitted.values())),
                             asseq_dropped=(min(drop), max(drop)),
                             df_not_1_anchor=TS['per_dataset']['beta0.0 rep 000']['final_df_not_1'],
                             trec_na=sum(c['trec_na'] for c in per)))



def f(x, d=3):
    return f'{x:.{d}f}'


def sci(x):
    """One-decimal scientific notation without a padded exponent: 0.0034235 -> 3.4e-3."""
    m, e = f'{x:.1e}'.split('e')
    return f'{m}e{int(e)}'


def ci(d, key='value', n=3):
    return f'{d[key]:.{n}f} [{d["lo"]:.{n}f}, {d["hi"]:.{n}f}]'


def ch_name(arm, ch):
    return f'{ch} ({MIX_CH[ch]})' if arm.startswith('mixqtl') else f'{ch} (its total channel)' if arm == TQ else ch


def table(head, rows):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def auc(b, a, bn='all'):
    return S['ranking'][f'beta{b}'][a]['auc'][bn]


def fdp(b, a):
    return S['ranking'][f'beta{b}'][a]['fdp_matched']


def bh(b, a):
    return S['gene_level'][f'beta{b}'][a]


def bhe(b, a):
    return S['gene_level_eigenmt'][f'beta{b}'][a]


def prec(sc, a, ch, part, key, bn='all'):
    return S['precision'][sc][a][ch][part][key][bn]


def bias(b, a, ch, key, bn='all'):
    return S['recovery'][f'beta{b}'][a][ch][key][bn]


def per_beta(fn, n=3):
    """'x / y / z' over |beta| 0.2 / 0.4 / 0.8 of a number-returning fn(b)."""
    return ' / '.join(f(fn(b), n) for b in BETAS)


def per_scen(fn, n=3):
    """'w / x / y / z' over the beta = 0 anchor and |beta| 0.2 / 0.4 / 0.8."""
    return ' / '.join(f(fn(sc), n) for sc in ('beta0.0',) + tuple(f'beta{b}' for b in BETAS))


def nul(a, ch, al, subset='all'):
    """A rate of the stored null on this pipeline: a dict with rate and its gene-clustered lo, hi."""
    return SN['rates'][a][ch][subset]['after'][al]


def rkey(a):
    """Efficiency against unit weights at the causal variant: hapmixQTL arms and total-only tensorQTL (unit weights' total
    fit) on their shared pipeline-scale truth; every other method on the count-scale truth for the arm and unit alike."""
    return 'ratio_vs_unit' if a in HAPMIX + (TQ,) else 'ratio_vs_unit_count'


def one_df_gene():
    return S['one_df_genes'][0]


def overlap(a, b):
    """The |beta| at which the ranges of the per-dataset AUCs of arms a and b overlap."""
    return [x for x in BETAS if auc(x, a)['lo'] <= auc(x, b)['hi'] and auc(x, b)['lo'] <= auc(x, a)['hi']]


def at_betas(bs):
    return 'no |beta|' if not bs else 'every |beta|' if len(bs) == len(BETAS) else '|beta| ' + ' and '.join(bs)


def range_overlap_text(a, b_arm):
    """Where the per-dataset AUC ranges of arm a and the higher arm b_arm overlap, and the gaps where they do not."""
    ov = overlap(a, b_arm)
    rest = [b for b in BETAS if b not in ov]
    if not rest:
        return 'overlap at every |beta|'
    gaps = ' and '.join(f(auc(b, b_arm)['lo'] - auc(b, a)['hi'], 3) for b in rest)
    return (f'{"overlap at " + at_betas(ov) + " and " if ov else ""}are separate at {at_betas(rest)}, by {gaps} '
            'between the ranges of three datasets each')


def all17():
    """Range over cutoffs and |beta| of the ladder's one-step all-17 total slope over the count-scale truth."""
    v = [LD['total_channel'][c][f'beta{b}']['one_step_all_trc']['all']['mean'] for c in ('published', 'permissive') for b in BETAS]
    return min(v), max(v)


# arm-level accessors used throughout the text: per-|beta| strings and anchor intervals
A_ = lambda a: per_beta(lambda b: auc(b, a)['mean'])                                              # noqa: E731
P_ = lambda a: per_beta(lambda b: fdp(b, a)['all']['power'])                                      # noqa: E731
R_ = lambda a: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['r2_high'], 2)                  # noqa: E731
dR = lambda a, b_arm: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['r2_high'] - S['lead'][f'beta{b}'][b_arm]['all']['r2_high'])   # noqa: E731
D_ = lambda a, ch='combined': per_beta(lambda b: S['detection'][f'beta{b}'][a][ch]['all']['0.001'], 2)   # noqa: E731
B_ = lambda a, ch, key='bias_count', bn='all', n=2: per_beta(lambda b: bias(b, a, ch, key, bn)['mean'], n)   # noqa: E731
E_ = lambda a, ch: per_beta(lambda b: prec(f'beta{b}', a, ch, 'nonnull', rkey(a))['value'], 2)     # noqa: E731
Ex_ = lambda a: per_beta(lambda b: prec(f'beta{b}', a, 'combined', 'nonnull', 'ratio_vs_unit_count')['value'], 2)   # noqa: E731
Z_ = lambda a, ch: per_beta(lambda b: prec(f'beta{b}', a, ch, 'nonnull', 'sd_z')['value'], 2)     # noqa: E731
En_ = lambda a, ch: ci(prec('beta0.0', a, ch, 'null', 'ratio_vs_unit'), 'value', 2)               # noqa: E731
Zn_ = lambda a, ch, n=2: ci(prec('beta0.0', a, ch, 'null', 'sd_z'), 'value', n)                   # noqa: E731
thr = lambda b, a: f'{fdp(b, a)["p_threshold"]:.1e}'                                             # noqa: E731
N_ = lambda a, ch='combined', n=3: per_scen(lambda sc: S['null'][sc][a][ch]['all']['0.05']['rate'], n)   # noqa: E731


def check_claims():
    """Claims that span sections; the others are guarded where they are written (need())."""
    for a in C.MIXQTL_ARMS:
        need(all(auc(b, 'split')['mean'] > auc(b, a)['mean'] and fdp(b, 'split')['all']['power'] > fdp(b, a)['all']['power']
                 and prec(f'beta{b}', 'split', 'combined', 'nonnull', 'ratio_vs_unit')['value']
                 < prec(f'beta{b}', a, 'combined', 'nonnull', 'ratio_vs_unit')['value'] for b in BETAS),
             f'split ahead of {a} on AUC, ranking power and combined squared error at every |beta| (section 5)')
    if LD is not None:   # section 3.8's fixed comparisons
        sq = lambda c, b, k: LD['total_channel'][c][f'beta{b}']['sq_error_vs_unit'][k]['all']   # noqa: E731
        rung = lambda c, b: LD['ladder'][f'beta{b}'][f'unit_{c}_cutoffs']['common_set']['total']['all']   # noqa: E731
        steps = lambda c: [rung(c, '0.8')['value'], *(sq(c, '0.8', k)['value'] for k in ('one_step_all_trc', 'one_step_trc', 'mixqtl_trc'))]   # noqa: E731
        fold = lambda v: [v[0]] + [v[i] / v[i - 1] for i in range(1, len(v))]   # noqa: E731
        p08, q08 = fold(steps('permissive')), fold(steps('published'))
        need(max(range(4), key=lambda i: p08[i]) == 3 and sq('permissive', '0.8', 'one_step_trc')['hi'] < sq('permissive', '0.8', 'mixqtl_trc')['lo'],
             'the two-step fit as the largest step at |beta| 0.8, permissive, with separated intervals (section 3.8)')
        need(max(range(1, 4), key=lambda i: q08[i]) == 3 and q08[0] > q08[3]
             and sq('published', '0.8', 'one_step_trc')['hi'] >= sq('published', '0.8', 'mixqtl_trc')['lo'],
             'the two-step fit as the largest increase after admission at |beta| 0.8, published, overlapping (section 3.8)')
        need(all(sq(c, '0.2', 'mixqtl_trc')['value'] < sq(c, '0.2', 'one_step_trc')['value'] for c in ('published', 'permissive')),
             'the two-step fit below the one-step fit at |beta| 0.2 under both settings (section 3.8)')
    print('fixed comparative claims of the text hold on this summary', flush=True)


def style(ax, ylabel=None):
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)


def save(fig, name):
    path = OUT / f'{name}.png'
    tmp = path.with_name(path.name + '.tmp')
    fig.savefig(tmp, dpi=130, bbox_inches='tight', format='png')
    plt.close(fig)
    os.replace(tmp, path)
    return path


def legend_below(fig, ax, y=-0.08):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, y), ncol=min(len(l), 4), fontsize=8.5, frameon=False)


def arm_points(ax, arms, xs, ys, los=None, his=None, offset=0.09, line=True):
    """One series per arm over categorical x positions, dodged, with optional intervals."""
    for j, arm in enumerate(arms):
        dx = (j - (len(arms) - 1) / 2) * offset
        x, y = [v + dx for v in xs], ys[arm]
        if line:
            ax.plot(x, y, color=COLOR[arm], lw=1.5, alpha=0.8)
        if los is not None:
            ax.errorbar(x, y, yerr=[[a - b for a, b in zip(y, los[arm])], [b - a for a, b in zip(y, his[arm])]],
                        fmt='none', ecolor=COLOR[arm], elinewidth=1.2, capsize=0)
        ax.plot(x, y, MARKER[arm], color=COLOR[arm], ms=7, mec='white', mew=0.8, label=LABEL[arm], ls='none')


def eigenmt_panels(ax_power, ax_fdp, X, xs, arms):
    """The eigenMT gene-level p held to a common error rate: power at 5% realized false-discovery proportion over the pooled
    datasets (ax_power), and the realized false-discovery proportion of each arm's Benjamini-Hochberg calls at 5% (ax_fdp)."""
    E = lambda a: [X['gene_level_eigenmt'][f'beta{b}'][a] for b in BETAS]   # noqa: E731
    arm_points(ax_power, arms, xs, {a: [e['fdp_matched']['all']['power'] for e in E(a)] for a in arms}, offset=0.08)
    arm_points(ax_fdp, arms, xs, {a: [e['false_discoveries'] / e['discoveries'] if e['discoveries'] else float('nan')
                                      for e in E(a)] for a in arms}, offset=0.08)
    ax_fdp.axhline(0.05, color=INK, lw=0.8, ls='--')
    ax_power.set_ylim(0, 1.02)


def fig_ranking():
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.6))
    axs = [*axs[0], *axs[1]]
    xs = range(len(BETAS))
    get = lambda a, k: [auc(b, a)[k] for b in BETAS]   # noqa: E731
    arm_points(axs[0], SHOWN, xs, {a: get(a, 'mean') for a in SHOWN}, {a: get(a, 'lo') for a in SHOWN}, {a: get(a, 'hi') for a in SHOWN},
               offset=0.07)
    axs[0].axhline(0.5, color=MUTED, lw=0.8, ls=':')
    style(axs[0], 'AUC, genes ranked by lead nominal p')
    axs[0].set_title('A. AUC of the gene ranking', fontsize=10, loc='left')
    arm_points(axs[1], SHOWN, xs, {a: [fdp(b, a)['all']['power'] for b in BETAS] for a in SHOWN}, offset=0.07)
    style(axs[1], 'share of non-null genes called')
    axs[1].set_title('B. Power at 5% realized false-discovery proportion', fontsize=10, loc='left')
    g = lambda a, k: [bh(b, a)['power_bh']['all'][k] for b in BETAS]   # noqa: E731
    perm = SC.CIS_ARMS   # the arms with a permutation p
    arm_points(axs[2], perm, xs, {a: g(a, 'rate') for a in perm}, {a: g(a, 'lo') for a in perm}, {a: g(a, 'hi') for a in perm}, offset=0.08)
    style(axs[2], 'share of non-null genes discovered')
    axs[2].set_title('C. Gene level, permutation p: BH at 5%', fontsize=10, loc='left')
    axs[2].set_ylim(0, 1.02)
    eigenmt_panels(axs[3], axs[4], S, xs, SHOWN)
    style(axs[3], 'share of non-null genes called')
    axs[3].set_title('D. Gene level, eigenMT p: power at 5% realized FDP', fontsize=10, loc='left')
    style(axs[4], 'null genes / genes called')
    axs[4].set_title('E. Gene level, eigenMT p: realized FDP of BH at 5%', fontsize=10, loc='left')
    axs[5].axis('off')
    for ax in axs[:5]:
        ax.set_xticks(list(xs), [f'|beta| = {b}' for b in BETAS])
    axs[1].set_ylim(0, 1.02)
    fig.tight_layout()
    legend_below(fig, axs[0], y=-0.06)
    return save(fig, 'fig_ranking')


def fig_bias():
    """Rows: the allelic and total channels of the arms that have them, then every arm's one combined slope on the
    count-scale truth, the row where TReCASE (one joint effect) and tensorQTL appear."""
    fig, axs = plt.subplots(3, 4, figsize=(15, 10.8), sharex=True)
    for i, ch in enumerate(('allelic', 'total', 'combined')):
        for k, bn in enumerate(BANDS):
            ax = axs[i, k]
            for j, arm in enumerate(SHOWN):
                key = ('combined' if ch == 'total' else None) if (arm == TQ and ch != 'combined') else ch   # tensorQTL's one slope is a total-channel slope
                if key not in S['recovery']['beta0.4'][arm]:
                    continue
                for m, b in enumerate(BETAS):
                    d = bias(b, arm, key, 'bias_count', bn)
                    ax.errorbar([j + (m - 1) * 0.24], [d['mean']], yerr=[[d['mean'] - d['lo']], [d['hi'] - d['mean']]], fmt='o',
                                color=BETA_COLOR[b], ms=5.5, mec='white', mew=0.6, elinewidth=1.2,
                                label=f'|beta| = {b}' if (j == 0 and i == 0 and k == 0) else None)
            ax.axhline(1, color=INK, lw=0.8)
            ax.axhline(0, color=MUTED, lw=0.6, ls=':')
            for x in (3.5, 5.5, 6.5) + ((8.5,) if NATIVE else ()):
                ax.axvline(x, color=GRID, lw=1)
            style(ax, f'{ch}: slope / truth' if k == 0 else None)
            ax.set_title(f'{ch}, {"all genes" if bn == "all" else bn + " reads"}', fontsize=10, loc='left')
            ax.set_xticks(range(len(SHOWN)), [SHORT[a] for a in SHOWN], rotation=40, ha='right')
    fig.tight_layout()
    legend_below(fig, axs[0, 0], y=-0.02)
    return save(fig, 'fig_bias')


def fig_lead():
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    xs = range(len(BETAS))
    for k, bn in enumerate(BANDS):
        arm_points(axs[k], SHOWN, xs, {a: [S['lead'][f'beta{b}'][a][bn]['r2_high'] for b in BETAS] for a in SHOWN}, offset=0.07)
        style(axs[k], 'share of non-null genes, lead r^2 >= 0.8' if k == 0 else None)
        n = S['lead']['beta0.4']['gibbs'][bn]['units']
        axs[k].set_title(f'{"all genes" if bn == "all" else bn + " reads"} ({n} gene units per |beta|)', fontsize=10, loc='left')
        axs[k].set_xticks(list(xs), [f'{b}' for b in BETAS])
        axs[k].set_xlabel('|beta| (log2)', color=MUTED, fontsize=9)
    axs[0].set_ylim(0, 1.02)
    legend_below(fig, axs[0], y=-0.2)
    return save(fig, 'fig_lead')


def fig_efficiency():
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6))
    for i, part in enumerate(('nonnull', 'null')):
        scen = BETAS if part == 'nonnull' else ('0.0',) + BETAS
        xs = list(range(len(scen)))
        for k, ch in enumerate(('allelic', 'total', 'combined')):
            ax = axs[i, k]
            arms = [a for a in (SHOWN if ch == 'combined' else HAPMIX if part == 'nonnull' else ARMS)
                    if a not in ('unit', 'mixqtl') and ch in S['precision']['beta0.0'][a]]   # published cutoffs: in the tables only (user decision 2026-09-28)
            rk = 'ratio_vs_unit_count' if (part, ch) == ('nonnull', 'combined') else 'ratio_vs_unit'
            get = lambda a, key: [prec(f'beta{b}', a, ch, part, rk)[key] for b in scen]   # noqa: E731
            arm_points(ax, arms, xs, {a: get(a, 'value') for a in arms}, {a: get(a, 'lo') for a in arms}, {a: get(a, 'hi') for a in arms}, offset=0.12)
            ax.axhline(1, color=INK, lw=0.8)
            ax.set_yscale('log')
            ax.set_ylim(min(min(get(a, 'lo')) for a in arms) / 1.15, max(max(get(a, 'hi')) for a in arms) * 1.15)
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(LOG_TICKS))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f'{t:g}' for t in LOG_TICKS]))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            style(ax, ('causal variant' if part == 'nonnull' else 'null genes, every tested variant') + '\nsquared error, arm / unit' if k == 0 else None)
            ax.set_xticks(xs, ['anchor' if b == '0.0' else f'|beta| {b}' for b in scen])
            ax.set_title(f'{ch} ({"non-null genes" if part == "nonnull" else "null genes"}'
                         f'{", count-scale truth" if rk == "ratio_vs_unit_count" else ""})', fontsize=10, loc='left')
    fig.tight_layout()
    legend_below(fig, axs[0, 2], y=-0.04)
    return save(fig, 'fig_efficiency')


def img(path, caption):
    alt = html.escape(caption, quote=True)
    return (f'<figure><img alt="{alt}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'
            f'<figcaption>{alt}</figcaption></figure>')


THIS_SET, REF_SET = 'low-coverage set', 'deep set'   # the two gene sets on a contrast page: this run's and the interpreted run's
RUN_COLOR = {THIS_SET: '#1f4e79', REF_SET: '#e07b00'}                 # neutral against the arm palette COLOR
RUN_MARKER = {THIS_SET: 'o', REF_SET: 's'}
ALPHA_COLOR = dict(zip(ALPHAS, ('#104281', '#2a78d6', '#86b6ef')))   # BETA_COLOR's ramp, dark to light


def log_axis(axs, lo, hi):
    """A shared log y axis over the data range with LOG_TICKS; a non-positive lower bound is cut at the floor."""
    floor = min(v for v in lo if v > 0) / 2 if min(lo) <= 0 else min(lo) / 1.15
    if min(lo) <= 0:
        print(f'{axs[0].get_title()}: a lower bound of 0 is cut at the axis floor', flush=True)
    ticks = [t for t in LOG_TICKS if floor <= t <= max(hi) * 1.15]
    for ax in axs:
        ax.set_yscale('log')
        ax.set_ylim(floor, max(hi) * 1.15)
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f'{t:g}' for t in ticks]))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_contrast_calibration(runs):
    """Anchor null-gene rate over its threshold (1 = nominal) for every arm at each threshold, one panel per gene set."""
    fig, axs = plt.subplots(1, len(runs), figsize=(13, 4.4), sharey=True)
    xs = range(len(ALL))
    lo_all, hi_all = [], []
    for ax, (name, X) in zip(axs, runs):
        for m, al in enumerate(ALPHAS):
            d = [X['null']['beta0.0'][a]['combined']['all'][al] for a in ALL]
            x = [v + (m - 1) * 0.22 for v in xs]
            y, lo, hi = ([v[k] / float(al) for v in d] for k in ('rate', 'lo', 'hi'))
            lo_all, hi_all = lo_all + lo, hi_all + hi
            ax.errorbar(x, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], fmt='o',
                        color=ALPHA_COLOR[al], ms=5.5, mec='white', mew=0.6, elinewidth=1.2, label=f'threshold {al}')
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_xticks(list(xs), [SHORT[a] for a in ALL], rotation=40, ha='right')
        style(ax, 'null-gene rate / threshold' if ax is axs[0] else None)
        ax.set_title(name, fontsize=10, loc='left')
    log_axis(axs, lo_all, hi_all)
    legend_below(fig, axs[0], y=-0.14)
    return save(fig, 'fig_contrast_calibration')


def fig_contrast_precision(runs, rows):
    """Squared error over unit weights' for the rows of the precision table, one series per gene set: at the causal
    variant (|beta| 0.4) and on the anchor's null genes."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    xs = range(len(rows))
    lo_all, hi_all = [], []
    for k, (sc, part, title) in enumerate((('beta0.4', 'nonnull', 'causal variant, |beta| 0.4'),
                                           ('beta0.0', 'null', 'null genes, beta = 0 anchor'))):
        ax = axs[k]
        for m, (name, X) in enumerate(runs):
            d = [X['precision'][sc][a][ch][part][rkey(a) if part == 'nonnull' else 'ratio_vs_unit']['all'] for a, ch in rows]
            x = [v + (m - 0.5) * 0.3 for v in xs]
            y, lo, hi = ([v[key] for v in d] for key in ('value', 'lo', 'hi'))
            lo_all, hi_all = lo_all + lo, hi_all + hi
            ax.errorbar(x, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], fmt=RUN_MARKER[name],
                        color=RUN_COLOR[name], ms=5.5, mec='white', mew=0.6, elinewidth=1.2, label=name if k == 0 else None)
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_xticks(list(xs), [f'{SHORT[a]}, {ch_name(a, ch)}' for a, ch in rows], rotation=40, ha='right')
        style(ax, 'squared error, arm / unit weights' if k == 0 else None)
        ax.set_title(title, fontsize=10, loc='left')
    log_axis(axs, lo_all, hi_all)
    legend_below(fig, axs[0], y=-0.3)
    return save(fig, 'fig_contrast_precision')


def fig_contrast_ranking(runs):
    """AUC, power at 5% realized false-discovery proportion and Benjamini-Hochberg gene-level power on the permutation p
    and on the eigenMT p by |beta|, one column per gene set (the arm colours and markers of Figure 1)."""
    fig, axs = plt.subplots(5, len(runs), figsize=(12, 18), sharex=True, sharey='row')
    xs = range(len(BETAS))
    for k, (name, X) in enumerate(runs):
        Rk = {b: X['ranking'][f'beta{b}'] for b in BETAS}
        arm_points(axs[0, k], ALL, xs, {a: [Rk[b][a]['auc']['all']['mean'] for b in BETAS] for a in ALL},
                   {a: [Rk[b][a]['auc']['all']['lo'] for b in BETAS] for a in ALL},
                   {a: [Rk[b][a]['auc']['all']['hi'] for b in BETAS] for a in ALL}, offset=0.08)
        axs[0, k].axhline(0.5, color=MUTED, lw=0.8, ls=':')
        arm_points(axs[1, k], ALL, xs, {a: [Rk[b][a]['fdp_matched']['all']['power'] for b in BETAS] for a in ALL}, offset=0.08)
        G = lambda a, v: [X['gene_level'][f'beta{b}'][a]['power_bh']['all'][v] for b in BETAS]   # noqa: E731
        arm_points(axs[2, k], ARMS, xs, {a: G(a, 'rate') for a in ARMS}, {a: G(a, 'lo') for a in ARMS},
                   {a: G(a, 'hi') for a in ARMS}, offset=0.08)
        eigenmt_panels(axs[3, k], axs[4, k], X, xs, ALL)
        for i, (lab, title) in enumerate((('AUC, genes ranked by lead nominal p', 'AUC of the gene ranking by lead nominal p'),
                                          ('share of non-null genes called', 'power at 5% realized false-discovery proportion'),
                                          ('share of non-null genes discovered', 'gene level, Benjamini-Hochberg 5% on the permutation p'),
                                          ('share of non-null genes called', 'gene level, eigenMT p, power at 5% realized FDP'),
                                          ('null genes / genes called', 'gene level, eigenMT p, realized FDP of BH at 5%'))):
            style(axs[i, k], lab if k == 0 else None)
            axs[i, k].set_title(f'{name}: {title}', fontsize=10, loc='left')
        axs[4, k].set_xticks(list(xs), [f'|beta| = {b}' for b in BETAS])
    for i in (1, 2, 3):
        axs[i, 0].set_ylim(0, 1.02)
    fig.tight_layout()
    legend_below(fig, axs[0, 0], y=-0.03)
    return save(fig, 'fig_contrast_ranking')


def tab_ranking():
    rows = [[LABEL[a]] + [ci(auc(b, a), 'mean') for b in BETAS]
            + [f'{f(fdp(b, a)["all"]["power"])} ({fdp(b, a)["discoveries"]} called, {fdp(b, a)["false"]} null)' for b in BETAS] for a in SHOWN]
    return table(['arm'] + [f'AUC, |beta| {b}' for b in BETAS] + [f'power at 5% FDP, |beta| {b}' for b in BETAS], rows)


def tab_bands(block, key):
    """Per arm and |beta|, the value in the three read bands, '<100 / 100-999 / >=1000'."""
    rows = [[LABEL[a]] + [' / '.join(f(block(b, a)[bn][key], 2) for bn in BANDS[1:]) for b in BETAS] for a in SHOWN]
    return table(['arm'] + [f'|beta| {b}: {BAND_HTML}' for b in BETAS], rows)


def tab_gene_level():
    """Per arm, Benjamini-Hochberg power and the null-gene count below 0.05 on its two gene-level p: its permutation p
    (n/a where it has none) and eigenMT's."""
    both = lambda sc, a: [S[k][sc].get(a) for k in ('gene_level', 'gene_level_eigenmt')]   # noqa: E731
    pw = lambda d: 'n/a' if d is None else (f'{ci(d["power_bh"]["all"], "rate")} ({d["discoveries"]} called, {d["false_discoveries"]} null)'   # noqa: E731
                                            + (f'<br>{f(d["fdp_matched"]["all"]["power"])} at 5% realized FDP' if d['p'] == 'eigenmt' else ''))
    nr = lambda d: 'n/a' if d is None else f'{d["null_rate"]["all"]["rejections"]} of {d["null_rate"]["all"]["tests"]}'   # noqa: E731
    rows = [[LABEL[a]] + [pw(d) for b in BETAS for d in both(f'beta{b}', a)]
            + [' / '.join(nr(d) for d in both(sc, a)) for sc in ('beta0.0',) + tuple(f'beta{b}' for b in BETAS)] for a in SHOWN]
    return table(['arm'] + [f'BH power, |beta| {b}, {k}' for b in BETAS for k in ('permutation p', 'eigenMT p')]
                 + [f'null genes with p &lt; 0.05, |beta| {b}: permutation / eigenMT' for b in ('0',) + BETAS], rows)


def tab_bias():
    n = lambda d: f' <span class="pipe">({d["units"]} / {d["excluded_nonfinite"]})</span>'   # noqa: E731
    rows = []
    for a in SHOWN:
        for ch in CHANNELS:
            if ch not in S['recovery']['beta0.4'][a]:
                rows.append([LABEL[a], ch, *['n/a'] * len(BETAS)])
                continue
            vals = []
            for b in BETAS:
                r = S['recovery'][f'beta{b}'][a][ch]
                txt = ci(r['bias_count']['all'], 'mean', 2) + n(r['bias_count']['all'])
                if 'bias_pipeline' in r:
                    txt += f'<br><span class="pipe">{ci(r["bias_pipeline"]["all"], "mean", 2)}</span>' + n(r['bias_pipeline']['all'])
                vals.append(txt)
            rows.append([LABEL[a], ch_name(a, ch)] + vals)
    return table(['arm', 'channel'] + [f'|beta| {b}: count scale<br><span class="pipe">pipeline scale</span> (units / excluded)'
                                       for b in BETAS], rows)


def tab_precision(key):
    rows = []
    for a in (SHOWN if key == 'sd_z' else tuple(a for a in SHOWN if a != 'unit')):
        k = key if key == 'sd_z' else rkey(a)
        joint_z = a in JOINT + ('trecase_native',) and key == 'sd_z'
        name = LABEL[a] + (', derived se (Wald inversion of &chi;<sup>2</sup>): causal-variant columns absorb bias; null column = '
                           'calibration of the likelihood-ratio test (its square is the mean &chi;<sup>2</sup>)' if joint_z
                           else ', count-scale truth' if k == 'ratio_vs_unit_count' else '')
        for ch in CHANNELS:
            if ch not in S['precision']['beta0.0'][a]:
                rows.append([LABEL[a], ch, *['n/a'] * (len(BETAS) + 1)])
                continue
            rows.append([name, ch_name(a, ch)] + [ci(prec(f'beta{b}', a, ch, 'nonnull', k), 'value', 2) for b in BETAS]
                        + [ci(prec('beta0.0', a, ch, 'null', key), 'value', 3 if joint_z else 2)])
    return table(['arm', 'channel'] + [f'causal variant, |beta| {b}' for b in BETAS] + ['null genes, beta 0 anchor'], rows)


def tab_cross():
    """Combined channel, every arm against unit weights, both on the count-scale truth."""
    rows = [[LABEL[a]] + [ci(prec(f'beta{b}', a, 'combined', 'nonnull', 'ratio_vs_unit_count'), 'value', 2) for b in BETAS]
            + [ci(prec('beta0.0', a, 'combined', 'null', 'ratio_vs_unit'), 'value', 2)] for a in SHOWN if a != 'unit']
    return table(['arm (combined channel)'] + [f'causal variant, |beta| {b}' for b in BETAS] + ['null genes, beta 0 anchor'], rows)


def need(ok, what):
    """Stop the page when a comparative sentence no longer holds on this summary."""
    if not ok:
        raise SystemExit(f'{C.SUMMARY}: {what} no longer holds; reword the sentence that states it')


def ivl(d, x=1.0):
    """Where the interval of d lies against x: 'above', 'below' or 'includes'."""
    return where(d, x)


def cross_note():
    """gibbs's combined excess over unit weights on the count-scale truth, against the pipeline-scale one."""
    c = [prec(f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit_count') for b in BETAS]
    return f"""
<p>The choice of truth barely moves these ratios on the half-read total: unit weights' total slope recovers
{B_('unit', 'total', n=3)} of the count-scale total truth and {B_('unit', 'total', 'bias_pipeline', n=3)} of the
pipeline-scale one (section 3.3), and gibbs's combined squared error at the causal variant is
{' / '.join(ci(d, 'value', 2) for d in c)} of unit weights' on the count-scale truth against
{' / '.join(ci(prec(f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit'), 'value', 2) for b in BETAS)} on the
pipeline-scale truth (the table above). On the anchor's null genes the truth is 0 for every arm, so the choice of truth
drops out there, but each method's slope scale does not: squared null error grows with the square of the slope scale,
so the anchor ratio is exact among the hapmixQTL weightings, which share one phenotype scale, and across methods it
carries each method's slope scale.</p>"""


def tab_conversion():
    """Each arm's published effect, its conversion to log2 aFC and where its standard error comes from."""
    truth = ('allelic: beta; total: per-gene total truth; combined: beta for bias, the inverse-variance combination '
             'of beta and the per-gene total truth for squared error')
    rows = [
        ['hapmixQTL, three weightings', 'allelic: slope of log2((L + 0.5)/(R + 0.5)) on xL &minus; xR; total: slope of '
         'the half-read log2 CPM on g/2; combined: their inverse-variance combination', 'none (log2 already)',
         'stated by the weighted least-squares fit', truth],
        ['mixQTL, two cutoff settings', 'the same three slopes on natural-log responses (asc log(L/R), trc '
         'log(total / 2 library size)); meta = inverse-variance combination', 'slope and se / ln 2',
         'stated by mixQTL\'s least-squares fits', truth],
        ['TReCASE (asSeq)', 'b = ln &kappa;, &kappa; the ALT over REF expression ratio, from the joint model, or from the '
         'total-count (TReC) model when asSeq\'s final p used it; the TReC mean is 1, (1 + &kappa;)/2, &kappa; at ALT '
         'dosage 0, 1, 2 (glmNBlog, glm.c:1577; the joint model, trecase.c:1049; genotypes recoded 3 &rarr; 1, '
         '4 &rarr; 2, R/trecase.R:205-206)',
         'b / ln 2', '<b>derived</b>: |slope| / &radic;&chi;<sup>2</sup> of the statistic used (asSeq reports none)',
         'beta, also for the total-count fallback (its model has the same dosage form)']]
    return table(['arm', 'published effect', 'conversion to log2 aFC, ALT over REF', 'standard error',
                  'count-scale truth used across methods'], rows)


def tab_lead():
    L = lambda b, a: S['lead'][f'beta{b}'][a]   # noqa: E731
    rows = [[LABEL[a]] + [f'{f(L(b, a)["all"]["lead_is_causal"], 2)} / {f(L(b, a)["all"]["r2_high"], 2)} / '
                          f'{f(L(b, a)["all"]["median_r2"], 2)}' for b in BETAS]
            + [' / '.join(str(L(b, a)['no_finite_p']) for b in BETAS)] for a in SHOWN]
    return table(['arm'] + [f'|beta| {b}: lead = causal / r<sup>2</sup> &ge; 0.8 / median r<sup>2</sup>' for b in BETAS]
                 + ['non-null gene units with no finite p (0.2 / 0.4 / 0.8)'], rows)


def tab_detection():
    D = lambda b, a, ch: S['detection'][f'beta{b}'][a].get(ch)   # noqa: E731  None: a joint arm's allelic or total channel
    rows = [[LABEL[a], ch_name(a, ch)] + [' / '.join(f(D(b, a, ch)['all'][al], 2) for al in DETECT) if D(b, a, ch) else 'n/a'
                                          for b in BETAS] for a in SHOWN for ch in CHANNELS]
    return table(['arm', 'channel'] + [f'|beta| {b}: p &lt; 0.05 / 1e-3 / 1e-5' for b in BETAS], rows)


def tab_null():
    rows = [[LABEL[a], ch_name(a, ch)] + [ci(S['null'][sc][a][ch]['all']['0.05'], 'rate', 4) if ch in S['null'][sc][a] else 'n/a'
                                          for sc in ('beta0.0',) + tuple(f'beta{b}' for b in BETAS)]
            for a in SHOWN for ch in CHANNELS]
    return table(['arm', 'channel'] + [f'|beta| {b}' for b in ('0 (anchor)',) + BETAS], rows)


def tab_anchor():
    rows = []
    for a in HAPMIX:
        for ch in CHANNELS:
            r, r3 = S['anchor'][a][ch]['0.05'], S['anchor'][a][ch]['0.001']
            rows.append([LABEL[a], ch, f(r['rate'], 6), f(r['stored'], 4), f'{f(r["perm_lo"], 6)} to {f(r["perm_hi"], 6)}',
                         f'{r["percentile"]:.1f}', 'yes' if r['passed'] else 'no', f'{f(r3["rate"], 4)} ({f(r3["stored"], 4)})'])
    return table(['arm', 'channel', 'this dataset, 0.05', 'stored mean, 200 permutations', 'central 99% of stored permutations',
                  'percentile among stored', 'inside', '0.001: this dataset (stored mean)'], rows)

CSS = '''
:root { --ink: #0b0b0b; --ink2: #52514e; --rule: #e1e0d9; --bg: #fcfcfb; --tint: #f3f2ee; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
       sans-serif; margin: 0; padding: 0 16px; }
main { max-width: 1080px; margin: 32px auto 64px; }
h1 { font-size: 26px; margin-bottom: 4px; } h2 { font-size: 20px; margin-top: 40px; border-bottom: 1px solid var(--rule);
padding-bottom: 4px; } h3 { font-size: 16px; margin-top: 28px; }
p { max-width: 900px; } .sub { color: var(--ink2); margin-top: 0; }
table { border-collapse: collapse; font-size: 12.5px; margin: 12px 0 18px; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--tint); font-weight: 600; } td { font-variant-numeric: tabular-nums; }
.pipe { color: var(--ink2); } figure { margin: 16px 0 24px; } figure img { max-width: 100%; height: auto; }
figcaption { color: var(--ink2); font-size: 13px; max-width: 900px; }
'''


def stratum_facts():
    """A stratum set's selection (SELECT_LOG), where the 100-gene run's genes sit on the same read measure (POOL), the
    transcriptome-wide allelic null rate of the stratum it was chosen for (STRATA) and the Salmon half-depth test's band
    (HALF_DEPTH); None for INTERPRETED_SET."""
    if INTERPRETED:
        return None
    log = SELECT_LOG.read_text()
    pool, lo, hi, cand, floor = re.search(
        r'([\d,]+) pass the eQTL gene filter .*?; [\d,]+ with median haplotype-informative reads over admitted donors in '
        r'\[(\d+), (\d+)\); ([\d,]+) of them with >= (\d+) admitted donors', log).groups()
    seed = re.search(r'genes selected from [\d,]+ candidates \((SeedSequence\(.*?\))\);', log).group(1)
    adm = re.search(r'median admitted reads per gene min ([\d.]+) / median ([\d.]+) / max ([\d.]+); median over all '
                    r'donors min ([\d.]+) / median ([\d.]+) / max ([\d.]+); admitted donors per gene min (\d+) / median '
                    r'(\d+) / max (\d+)', log).groups()
    rows = [x.split('\t') for x in POOL.read_text().splitlines()]
    reads = {r[0]: float(r[rows[0].index('median_admitted_reads')] or 'nan') for r in rows[1:]}   # empty: no admitted donor
    ref = [reads[g] for g in REF_GENES.read_text().split()]
    rows = [x.split('\t') for x in STRATA.read_text().splitlines()]
    cov = [dict(zip(rows[0], r)) for r in rows[1:] if r[0] == 'cov_bin']
    own = [c for c in cov if c['bin'].startswith(f'{lo}-{hi} ')]
    if len(own) != 1 or len(ref) != 100:
        raise SystemExit(f'{STRATA}: {len(own)} coverage strata named {lo}-{hi}; {REF_GENES}: {len(ref)} genes')
    other = [float(c['direct_0.05']) for c in cov if c is not own[0]]
    shared = sorted(set(REF_GENES.read_text().split()) & set(C.GENES.read_text().split()))
    A, B = (np.load(d / 'beta0.0' / 'rep000.npz') for d in (REF_RUN.parent / 'datasets', C.DATASETS))
    same_perm = all(np.array_equal(A[k], B[k]) for k in ('perm', 'swap', 'is_null'))   # the generator's streams are keyed on the replicate only
    design = [x.split('\t') for x in C.GENE_DESIGN.read_text().splitlines()]
    below = sum(float(r[design[0].index('median_allele_resolved_reads')]) < int(lo) for r in design[1:])
    # the committed run's arms log can hold datasets later dropped (stratum30_100 ran 19, scored 10): its dataset blocks, and those scored
    blocks = [m for m in (re.match(r'beta(\S+) rep (\d+) ', x) for x in COMMITTED_RUN_LOG.read_text().splitlines()) if m]
    lines = (len(blocks), sum(int(m.group(2)) < S['n_datasets'][m.group(1)] for m in blocks))
    hd = json.loads(HALF_DEPTH.read_text())
    band = f'{lo}-{int(hi) - 1}'
    if band not in hd['bands']:
        raise SystemExit(f'{HALF_DEPTH}: no band {band} among {sorted(hd["bands"])}')
    h = hd['bands'][band]
    half = dict(band=band, f=hd['f'], failed=band in hd['failed_bands'], pass_band=hd['pass_band'], va=h['va_meas_over_pred'],
                exponent=h['exponent'], became_one_sided=h['became_one_sided'], two_sided=h['two_sided_full'], attenuation=h['attenuation'])
    print(f'stratum facts: {SELECT_LOG}, {POOL}, {STRATA}, {HALF_DEPTH} (band {band}); {COMMITTED_RUN_LOG}: {lines[0]} dataset '
          f'blocks, {lines[1]} of them from the {sum(S["n_datasets"].values())} scored datasets', flush=True)
    return dict(pool=pool, lo=int(lo), hi=int(hi), cand=cand, floor=floor, seed=seed, adm=adm, half=half,
                ref_in=sum(int(lo) <= x < int(hi) for x in ref), ref_above=sum(x >= int(hi) for x in ref),
                ref_below=sum(x < int(lo) for x in ref), ref_median=float(np.median(ref)), strata=len(cov),
                own={k: float(own[0][k]) for k in ('direct_0.05', 'direct_0.05_lo', 'direct_0.05_hi', 'n_genes', 'median_med_asc')},
                other=(min(other), max(other)), total_genes=sum(int(float(c['n_genes'])) for c in cov),
                shared=shared, same_perm=same_perm, below=below, lines=lines)


def sec_head():
    run_date = datetime.date.fromtimestamp((C.RESULTS / 'run_arms_facts.json').stat().st_mtime).isoformat()   # when 03 wrote this root's arms
    dated = ('<p class="sub"><b>Configuration.</b> Every hapmixQTL arm, and total-only tensorQTL, runs on the '
             'half-read total (user decision 2026-10-01); "shipped" means half-read split, the default since '
             '2026-09-29. mixQTL mode keeps its published natural-log response.</p>')
    if not INTERPRETED:
        n_genes = S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z']['all']['genes']
        n_ds = S['n_datasets']
        ran = SF['lines'][0] * sum(n_ds.values()) // SF['lines'][1]   # datasets the arms ran on: blocks per scored dataset are the same for every dataset
        return (f'<h1>Simulated-effects eQTL benchmark: the {THIS_SET} ({SF["lo"]}-{SF["hi"]} reads)</h1>' + dated +
                f'<p class="sub">This page is the {THIS_SET}: {n_genes} genes drawn at random '
                f'({SF["seed"]}) from the {SF["cand"]} of {SF["pool"]} eQTL-filter genes whose median haplotype-informative '
                f'reads over admitted allelic donors lie in [{SF["lo"]}, {SF["hi"]}) and that have at least {SF["floor"]} '
                f'admitted allelic donors (median admitted reads per gene {SF["adm"][0]} to {SF["adm"][2]}, median '
                f'{SF["adm"][1]}; admitted allelic donors {SF["adm"][6]} to {SF["adm"][8]}, median {SF["adm"][7]}). It '
                f'holds {n_ds["0.0"]} beta = 0 anchor dataset and {n_ds["0.2"]} / {n_ds["0.4"]} / {n_ds["0.8"]} replicate '
                f'datasets at |beta| = 0.2 / 0.4 / 0.8, each with half the genes non-null, so every effect-size '
                f'comparison rests on {n_ds["0.4"]} replicates. hapmixQTL weightings, mixQTL mode, total-only tensorQTL '
                f'and TReCASE on the BrainVar cohort\'s own Salmon output with injected effects, {n_genes} genes x 92 donors '
                f'({C.GENES}); hapmixQTL arms with commit 8a06803\'s per-channel t references and {MIN_ALLELIC_DONORS}-donor '
                f'allelic floor and with Meier\'s correction of the combined standard error for estimated channel weights '
                f'(commit a1b2ef4, section 2); this run\'s directory {C.ROOT}, with the hapmixQTL, mixQTL and tensorQTL arms '
                f'run there on {run_date}, and {JOINT_HEAD}; units log2 aFC (beta = 1 is a twofold effect). The '
                f'section "The {THIS_SET} against the {REF_SET}", after section 3, sets it against the {REF_SET}\'s delivered '
                f'run ({REF_RUN}), each set with its own intervals. The read bands of sections 2 and 3 use each gene\'s median '
                f'over all donors, on which {SF["below"]} of these genes fall below {SF["lo"]} reads; the set\'s own measure is '
                f'the median over admitted donors. In {C.COMMITTED.name}, the committed run whose joint-model results '
                f'{"are" if STAGED else "were"} reused here, {COMMITTED_RUN_LOG.name} holds {SF["lines"][0]} dataset blocks from {ran} '
                f'datasets: its hapmixQTL and mixQTL arms first ran on {(ran - n_ds["0.0"]) // (len(n_ds) - 1)} replicates '
                f'per |beta|, the datasets were then regenerated at {n_ds["0.4"]} (user decision 2026-09-27; every generator '
                f'stream is keyed on the replicate index, so the kept replicates are unchanged), and its joint arms and '
                f'scoring used those {sum(n_ds.values())}. This run\'s hapmixQTL, mixQTL and tensorQTL arms ran once, on its own '
                f'{sum(n_ds.values())} datasets ({C.DATASETS}). Made by '
                f'benchmark/simulated_effects/08_report.py from {C.SUMMARY}, {REF_RUN}, {SELECT_LOG}, {POOL}, {STRATA}, {HALF_DEPTH}, '
                f'{COMMITTED_RUN_LOG}, the check files that 01_check_inputs.py wrote into {C.CHECKS} on this run '
                f'({C.ROOT / "01_check_inputs.log"}), the run facts of {C.DATASETS} and {C.RESULTS}, and the '
                f'TReCASE\'s summary in {C.JOINT["trecase"]}'
                + (f', and the native-input arms\' {C.NATIVE / "facts.json"} and {C.NATIVE_RESULTS["trecase_native"] / "summary.json"} '
                   f'(TReCASE and split weighting on alignment counts from the same BAMs, section {native_sec()})' if NATIVE else '')
                + '; figures also written as PNG in '
                f'{OUT}. The {REF_SET} page\'s interpretation paragraphs, its mixQTL ladder section '
                f'and its critique and meaning sections are not made for this set; the contrast section '
                f'carries this set\'s comparisons, its limit and what it settles'
                + (', and the limits section what the run cannot establish' if sec_closing_limits() else '') + '.</p>')
    return ('<h1>Simulated-effects eQTL benchmark: recovering known cis effects</h1>' + dated +
            '<p class="sub">Three hapmixQTL weightings on the half-read total (split, the shipped default; gibbs; unit), mixQTL '
            'mode, total-only tensorQTL and TReCASE on the BrainVar cohort\'s own Salmon output with injected effects'
            + (', and TReCASE and split weighting also on alignment counts from '
               f'the same BAMs (section {native_sec()}, run into {C.NATIVE})' if NATIVE else '')
            + '; 100 genes x 92 donors; datasets of 2026-09-26, regenerated with the half-read total; hapmixQTL, mixQTL and '
            f'tensorQTL arms and the mixQTL ladder run {run_date} into {C.ROOT}, with per-channel t references, a 15-donor '
            'allelic admission floor and Meier\'s correction of the combined standard error for estimated channel weights '
            f'(section 2); {JOINT_HEAD}; units log2 aFC (beta = 1 is a twofold effect). Made by '
            f'benchmark/simulated_effects/08_report.py from {C.SUMMARY}, the stored null {NULL} and its draws, the check files '
            f'that 01_check_inputs.py wrote into {C.CHECKS} on this run ({C.ROOT / "01_check_inputs.log"}), the run facts of '
            f'{C.DATASETS} and {C.RESULTS}, TReCASE\'s summary in {C.JOINT["trecase"]}, '
            + (f'{C.LADDER}, and the native-input arms\' {C.NATIVE / "facts.json"} and '
               f'{C.NATIVE_RESULTS["trecase_native"] / "summary.json"}' if NATIVE else f'and {C.LADDER}')
            + f'; figures also written as PNG in {OUT}.</p>')

def sec_why():
    first = '''
<p>On 2026-09-29 half-read split became the shipped weighting (docs/pipeline_rules.md, "Decision, 2026-09-29:
half-read split is the default weighting configuration"; the evidence is recorded in
brainvar_hapmix_deploy/half_read_default_adoption_20260929/): the allelic contrast weighted by its Gibbs variance, the
total on the half-read log CPM with a unit working variance. Until today this benchmark ran on the earlier log2(CPM + 1)
total, so its arms did not include the configuration that shipped. On 2026-10-01 every hapmixQTL arm,
and total-only tensorQTL, moved onto the half-read total, and RASQUAL was set aside (user decisions). A null says whether
an arm's p values can be trusted; it cannot say how well the shipped default finds a real effect, how close its slope
comes to the truth or what the Gibbs draws buy, because real data carry no known effect. The datasets are built from the
cohort's own Salmon output, keeping its depth, noise and donor structure and adding a known effect.</p>''' if SF is None else f'''
<p>The first simulated-effects run ({FIRST_RUN.name}) built datasets with known cis effects from the cohort's own Salmon
output on 100 genes that are mostly deeper than this set: on the read measure that defines it (median
haplotype-informative reads over admitted allelic donors) {SF["ref_in"]} of those genes lie in [{SF["lo"]},
{SF["hi"]}), {SF["ref_below"]} below and {SF["ref_above"]} above it, median {SF["ref_median"]:.0f} reads
({POOL.name}); both sets hold {SF["ref_in"] + SF["ref_above"] + SF["ref_below"]} genes, so the two are told apart here
by depth, not by count. Transcriptome-wide, on the pre-correction pipeline (natural-log Gibbs-mean phenotype, synthetic
Hardy-Weinberg variants, records permutation), the {SF["lo"]}-{SF["hi"]}-read coverage bin had the highest
allelic nominal-p rate at 0.05 of {SF["strata"]} coverage bins: {SF["own"]["direct_0.05"]:.4f}
[{SF["own"]["direct_0.05_lo"]:.4f}, {SF["own"]["direct_0.05_hi"]:.4f}] over {SF["own"]["n_genes"]:,.0f} genes, against
{SF["other"][0]:.4f} to {SF["other"][1]:.4f} in the others ({STRATA.parent.name}/{STRATA.name}). That bin is the
pre-correction pipeline's analogue of this set, not its definition: scripts/coupling_reach.py bins genes on the
median Gibbs-mean haplotype-informative reads over the donors it admits, over {SF["total_genes"]:,} genes with at least
20 of them, where this gene set is drawn on point-estimate reads under the zero-haplotype admission rule, at least
{SF["floor"]} admitted donors, over {SF["pool"]} eQTL-filter genes; the bin's median is {SF["own"]["median_med_asc"]:.1f}
reads against this set's {SF["adm"][1]}. The choice of weighting therefore rested on genes where that rate was lower.</p>'''
    return '''
<h2>1. Why the analysis was needed</h2>''' + first + '''
<p>The question: on data with the real cohort's structure, does the shipped default, half-read split, rank non-null
genes above null ones, discover them at a controlled false-discovery rate, estimate the injected slope without bias,
state its standard error correctly and place the lead variant on the causal one better than its control without the
Gibbs draws (unit weights), than weighting the total by its Gibbs variance as well (gibbs), than mixQTL mode (the
published estimator, which never sees the draws), than a total-only tensorQTL scan, and than TReCASE, a published model
that fits total and allele-specific counts in one likelihood?</p>'''


def sec_run():
    n_genes = S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z']['all']['genes']
    th, rc, rp = CG['thinning'], CG['recovery'], CG['reproduction']
    mr = rc['min_reads']
    n_ds, mp = S['n_datasets'], S['mixqtl_permutation']
    set_desc = ('the 100 genes of the corrected null store' if INTERPRETED
                else f'the {n_genes} genes of the {C.GENE_SET} gene set ({C.GENES})')
    band_desc = 'fewer than 100, 100-999, at least 1,000' if INTERPRETED else BAND_HTML.replace(' / ', ', ')
    if rp is not None:
        repro = ('Check (d), exact reproduction of the stored null: given the stored null run\'s own permutation 0 and '
                 'haplotype swaps, the beta = 0 path through map_nominal must reproduce that run\'s first permutation for '
                 'every hapmixQTL arm in every column: the channel and combined slopes, their standard errors, the three p '
                 'values and their calls, the degrees of freedom and the allelic admission. The stored null is made by the '
                 'same code on the same inputs (below), so the check tests the plumbing that connects the generator to it.')
    else:
        repro = (f'Check (d), exact reproduction of a stored null permutation, needs the stored 200-permutation null '
                 f'run, which exists for the {INTERPRETED_SET} gene set only, and is skipped for this set.')
    mix_perm_txt = f'''mixQTL's own permutation scan ran on every dataset at both cutoff settings, {mp["nperm"]:,}
permutations on CPU, under mixQTL's published null: the phenotype bundle (the two haplotype counts, the total
and the library size) and the RNA-tied covariates move with the donor record, the genotype principal components stay,
the covariate offset is refitted on each permuted dataset, and no haplotype labels are swapped. Its gene-level p is the
empirical permutation p of the gene's largest |meta statistic|, (1 + the number of permuted maxima at least as
large) / (1 + the number of finite permuted maxima), without a Beta approximation, which mixQTL's port does not have.
<b>eigenMT</b> (Davis et al. 2016) gives every arm, TReCASE included, a second gene-level p that needs no
permutation: a gene's effective number of independent tests, M<sub>eff</sub>, is the number of eigenvalues of its
tested variants' genotype correlation matrix (Ledoit-Wolf shrunk: the sample correlation pulled toward the identity by
a weight estimated from the data; in windows of 200 consecutive variants) needed to
explain 99% of their variance, and its gene-level p is min(1, M<sub>eff</sub> x the gene's smallest nominal p), a
Bonferroni correction over M<sub>eff</sub> tests. M<sub>eff</sub> depends on the genotypes alone, so it is the same
for every arm and dataset.'''
    return f'''
<h2>2. What was run</h2>
<p><b>Generator.</b> Each dataset starts from the real cohort's Salmon output for the donor-gene pairs of {set_desc}
(92 donors). Three steps turn it into a dataset with a known answer. First, the real
associations are broken: donor records are permuted against fixed genotypes. A record's point estimates,
Gibbs draws, library size and RNA-tied covariates move together; the genotype principal components stay with
the genotypes; each moved record's L and R labels are swapped with probability one half. Second, an effect is
injected. For each non-null gene one causal variant is drawn among its tested variants, with |beta| = 0.2, 0.4 or 0.8 log2
units and a random sign (beta = 1 would be a twofold allelic effect). On each donor, the haplotype carrying the
lower-expressed allele keeps each of its reads with probability f = 2<sup>-|beta|</sup>. This is binomial thinning
(Gerard 2020, BMC Bioinformatics): the data keep their own noise, depth and donor-to-donor structure and gain
only the chosen signal. Null genes are thinned by the same average factor, so null and non-null genes sit at
the same depth. Third, the Gibbs variance of a thinned record is set. For the total channel, every Gibbs draw
is thinned like the point estimate. For the allelic channel, the real record's Gibbs variance is scaled by the
ratio of the counting term q = 1/(pL + 0.5) + 1/(pR + 0.5) at the thinned over the real counts, where pL and pR are
Salmon's point-estimate read counts on the L and R haplotypes. That rule is
read from Salmon 1.10.3's Gibbs sampler (CollapsedGibbsSampler.cpp lines 149, 257-265 and 507): reads shared by
both haplotypes carry no allelic information, so the allelic Gibbs variance scales with one over the
haplotype-specific reads, and thinning scales those reads by f. Library sizes are not recomputed after thinning;
section 3 gives the share of reads it removes.</p>
<p><b>Datasets.</b> {n_ds["0.0"]} beta = 0 anchor dataset (every gene null, no thinning) and
{n_ds["0.2"]} / {n_ds["0.4"]} / {n_ds["0.8"]} datasets at |beta| = 0.2 / 0.4 / 0.8, each with {n_genes // 2} of the {n_genes} genes
non-null. Dataset r uses the same permutation, causal variants, signs and null genes at every |beta|, so the
effect sizes are paired, not independent replicates. A gene is non-null in about {n_ds["0.4"] / 2:g} of the {n_ds["0.4"]} datasets of a scenario, so every statistic is pooled
over gene-dataset units (a <i>causal unit</i> is one non-null gene in one dataset, at its causal variant).
Genes are grouped into three <i>read bands</i> by their real median haplotype-informative reads over donors
({band_desc}).{"" if SF is None else " This run repeats the benchmark, unchanged, on the genes of the coverage bin named in section 1."}</p>
<p><b>Arms.</b> Three hapmixQTL weightings, all in default mode and all on the same two phenotypes: the allelic
contrast A = log2((pL + 0.5)/(pR + 0.5)) and the <i>half-read total</i> T = log2((pT + 0.5)/(L + 1) x 10<sup>6</sup>),
pT the total point-estimate reads of the gene (every transcript) and L the donor's effective library size, the
shipped default's total since 2026-09-29 (user decision 2026-10-01: every arm on it). A is also the difference of the two
haplotypes' half-read log2 CPM, the library cancelling. Each record's residual has variance sigma<sup>2</sup> w, w the
arm's working variance and sigma<sup>2</sup> the residual scale fitted per variant. All run after the zero-haplotype
admission rule (an allelic record with exactly one haplotype below 0.5 reads is excluded):
<b>split</b>, the shipped default, w = Va in the allelic channel (Va the Gibbs variance of A over Salmon's 200 Gibbs
draws plus its counting term) and w = 1 in the total channel; <b>gibbs</b>, w = Va in the allelic channel and w = Vt in
the total channel, Vt the Gibbs variance of T over the draws plus its counting term 1/((pT + 0.5) ln<sup>2</sup>2), so
the total channel too is weighted by its measurement variance; <b>unit</b>, w = 1 in both channels, the control without
the Gibbs draws. Each channel's p is referred to t with that channel's own residual degrees of
freedom (informative donors minus the fitted columns), and the combined p to the <i>Welch-Satterthwaite</i> degrees of
freedom of the inverse-variance combination: the degrees of freedom of the scaled &chi;<sup>2</sup> whose first two
moments match those of a fixed weighted sum of independent variance estimates, here
(w<sub>a</sub> + w<sub>t</sub>)<sup>2</sup> / (w<sub>a</sub><sup>2</sup>/&nu;<sub>a</sub> +
w<sub>t</sub><sup>2</sup>/&nu;<sub>t</sub>) with w = 1/se<sup>2</sup> and &nu; each channel's degrees of freedom. The
allelic channel enters the combined statistic only for a gene with at least {MIN_ALLELIC_DONORS} informative allelic donors, mixQTL's
own cutoff for combining its two channels; below that the combined slope, se and p are the total channel's.
The combined standard error carries <i>Meier's correction</i> (Meier 1953) for channel
weights estimated from the residuals they combine: the plug-in variance 1/(w<sub>a</sub> + w<sub>t</sub>) is
multiplied by M = 1 + 4 f<sub>a</sub> f<sub>t</sub> (1/&nu;<sub>a</sub> + 1/&nu;<sub>t</sub>), f being each
channel's share of the weight, so the combined t falls by &radic;M on the same Welch-Satterthwaite degrees of
freedom; M is 1 wherever one channel carries all the weight, and slopes and per-channel statistics are unchanged. It
applies in map_nominal and in map_cis's scan and every permutation alike (docs/hapmixqtl_methods.md, Section 4.5, has
the derivation and the reference's measured cost).
Two mixQTL-mode arms run on the thinned point estimates, never
on the draws: <b>published cutoffs</b> (total reads 100, allelic reads 50 to 1,000, weight cap 10) and
<b>permissive cutoffs</b> (20, 5 to 5,000, cap 100). The realized fold cap is min(weight cap, floor(n/10)) for n
admitted donors, at most 9 with 92 donors, so the two weight-cap settings act identically and only the count
cutoffs differ between the mixQTL arms. mixQTL's combined estimate, its <i>meta statistic</i>, is the
inverse-variance combination of its allelic (asc) and total (trc) estimates. Its natural-log slopes and standard
errors are divided by ln 2. The hapmixQTL arms were also run through
map_cis for gene-level p (1,000 permutations of donor records with haplotype-label swaps, GPU), with the Beta
approximation: a Beta distribution fitted to the permuted minimum p values, used to smooth the gene-level p. {mix_perm_txt}</p>
<p><b>Total-only tensorQTL.</b> tensorQTL's own cis scan (tensorqtl.cis map_nominal and map_cis) on the total phenotype T
alone, the half-read total, unweighted and with no allelic channel, with the same 17 covariates (the genotype principal
components among them as ordinary covariates) and the same tested variants per gene; map_cis permutes the
covariate-residualized phenotype 1,000 times and fits the Beta approximation. It regresses on ALT dosage g, so its
slope and standard error are doubled to put them on g/2, the scale of the hapmixQTL total channel and of the truth.
It is the standard total-expression eQTL scan, and its least-squares fit is unit weights' total channel. It is scored
as a one-test arm: its one slope is the combined row, held to the total truth (pipeline scale at the causal variant, as
for the hapmixQTL arms; count scale across methods), and its squared error is compared with unit weights' combined
slope, so that ratio measures what the allelic channel adds to a total-only scan.</p>
<p><b>Joint model.</b> TReCASE, a published method that fits the total and allele-specific counts in one likelihood,
was run on every dataset as a further comparator, nominal only (no permutation p; eigenMT gives it a gene-level p). It
gives one test per variant, scored here as its combined channel; its allelic and total rows read n/a. RASQUAL, the
other joint model of earlier versions of this page, is left out for now (user decision 2026-10-01; 04_run_rasqual.py is
kept). <b>TReCASE</b> (asSeq 0.99.501) models total counts as negative binomial (TReC), a count model that allows
<i>overdispersion</i>, variance beyond that of a Poisson count, and allele-specific counts as beta-binomial (ASE), its
binomial counterpart; it fits both jointly and runs a cis/trans test of whether the total and
allelic effects agree; asSeq's final p is the joint p when that test does not reject at 0.05 and the total-count p
otherwise, which is also what it reports when the joint fit fails. Each p comes from a <i>likelihood-ratio
statistic</i> &chi;<sup>2</sup>, twice the gain in log-likelihood when the variant's effect is added. Its inputs are the
donor-gene pairs the hapmixQTL arms admit to the allelic channel as allele-specific
records (counts rounded per haplotype, because its beta-binomial needs integers), fractional totals, the log effective
library size as offset, and the 17 covariates, {COV_OF['trecase']}; asSeq's defaults are kept except the p cutoff for writing a row
(05_run_trecase.py).{JOINT_COV_PARA}</p>
{native_method()}
<p><b>One scale for every method.</b> Every slope on this page is a log2 allelic fold change (aFC), ALT over REF,
where beta = 1 is a twofold effect. The table gives each arm's published effect, its conversion, and where its
standard error comes from. TReCASE reports no standard error: it is derived as |slope| / &radic;&chi;<sup>2</sup>
(the Wald inversion of &chi;<sup>2</sup>: the standard error at which (slope / se)<sup>2</sup> equals
&chi;<sup>2</sup>), so under truth 0 (null genes) z = slope / se is &plusmn;&radic;&chi;<sup>2</sup>
by construction, and its realized-over-stated standard error on null genes (section 3.4) measures the calibration of
its likelihood-ratio test, not a reported standard error. At the causal variant z = (slope &minus; beta) / se instead
measures how well the derived se describes the slope's spread around beta, and absorbs bias. For
comparisons across methods every arm is held to the count-scale truth (defined below); the pipeline-scale
truth is a hapmixQTL-only diagnostic and never ranks methods. In TReCASE the total mean at ALT dosage
0, 1, 2 is proportional to 1, (1 + &kappa;)/2, &kappa; for an ALT over REF ratio &kappa;. That is the expected form,
averaged over donors, of the total fold the generator injects: thinning one haplotype's reads and the shared reads by
different factors gives exactly this form only for a donor whose haplotype-specific reads are balanced, and on average
over the random direction of real imbalance. Its estimand is therefore log2 &kappa; = beta, also when asSeq falls back
to its total-count test, whose model has the same dosage form (glm.c:1577); the per-gene total truth, a straight-line fit
of that fold on g/2, is the estimand of the linear total channels of hapmixQTL and mixQTL. One asSeq fallback cannot be
identified per test: where its total-count dosage model fails it refits the dosage as a linear covariate, whose slope
is a log fold per ALT allele, about half of ln &kappa;, so about half of beta after the conversion. Such a row at a
causal variant lowers TReCASE's bias ratio and raises its squared error; the rows are not flagged.</p>
{tab_conversion()}
<p><b>Missing rows.</b> TReCASE's rows with &chi;<sup>2</sup> &le; 0 (p = 1), whose derived standard error is undefined,
are left out of the standard-error statistics only, and stay in the ranking, the null rates and squared error. TReCASE has no row for a tested pair whose ALT dosage is the same in every
donor. A causal unit without a row is left out of that arm's causal-variant detection share, and in bias and precision
it is non-finite and so excluded and counted like any other non-finite unit.</p>
{ladder_method() if LD is not None else ''}
<p><b>Checks before the run.</b> 01_check_inputs.py runs its checks on this run's datasets and writes them to
{C.CHECKS}. The premise of the allelic rule: on donor {CP["sample"]}'s dumped equivalence classes, over the genes with at
least {CP["min_u"]} haplotype-specific reads on each haplotype, the observed allelic Gibbs variance beyond counting noise is
set against the variance predicted from the shared-read share s, and the prediction's rank correlation with the observed
excess against the counting term's; its pass thresholds were set after its first result, so it guards the derivation
against regression rather than testing it independently. Check (a), identity: with every thinning factor 1, the
generator must reproduce the pipeline's inputs exactly: A, the allelic log2 ratio log2((pL + 0.5)/(pR + 0.5)); T, the
half-read total; and Va and Vt, their Gibbs variances, also after a permutation and swap. Check (b), thinning: the
Fano factor (across-draw variance over mean) of the total Gibbs draws must stay at its real value after thinning by
f = {th["f"]}, and the allelic rule's arithmetic must hold on every informative record. Check (c), recovery: over
{rc["n_datasets"]} all-non-null datasets at |beta| = {rc["beta"]}, unit weights must recover the pipeline-scale truth of the
allelic slope{f' (genes with at least {mr} reads)' if mr else ''}; the same datasets give the allelic slope under 1/Va
weights evaluated three ways. Its gene-clustered se is the standard deviation of the per-gene means over the square root
of the number of genes, as 01_check_inputs.py computes it, not the resampling interval used elsewhere. {repro}</p>
{earlier_runs_method()}
{sec_scoring()}'''


def earlier_runs_method():
    """Section 2's account of the stored runs the page reads besides this run: design only, no outcome."""
    if not INTERPRETED:
        return f'''
<p><b>Other runs this page reads.</b> The contrast section reads the {REF_SET}\'s run ({REF_RUN.parent.name}), whose arms
used {REF_COV.name} ({REF_COV_HOW}); this run\'s arms used {ARMS_COV.name} ({ARMS_COV_HOW}). Where the two differ, the
contrast section says so and {LIMITS} says what would have to run to make the comparison like for like.</p>'''
    return f'''
<p><b>The stored null.</b> The anchor dataset is one permutation; to place it, and for the null rates of 200
permutations, the page reads a stored null run on this pipeline ({C.STORED_NULL.name}, scripts/half_read_stored_null.py):
the three hapmixQTL weightings on the real records of these 100 genes, with no effect injected and no thinning, under
200 permutations of donor records against genotypes with haplotype-label swaps, on the same code, phenotypes, covariates
({ARMS_COV.name}) and Meier's correction as this run. Its comparisons with this run are like for like in every channel.
Check (d) reproduces its first permutation from this run's generator.</p>'''


def sec_scoring():
    """How each statistic of section 3 is defined: section 2's last part, so that section 3 holds results only."""
    genes = S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z']['all']['genes']
    n_rep = S['n_datasets']['0.4']
    if INTERPRETED:
        auc_txt = '''It is computed
per dataset and averaged over the 3 datasets. Its interval is the range of the three per-dataset AUCs: with
three datasets, the 2.5% and 97.5% quantiles of the mean over datasets resampled with replacement are the smallest
and largest dataset values. It is not a 95% interval, and it carries no gene-to-gene variation, because the three
datasets hold the same 100 genes.'''
        anchor_txt = f'''For the
three hapmixQTL arms its rate is compared with the stored null on this pipeline (section 2): the percentile of this
dataset's rate among the 200 stored per-permutation rates, and whether it lies inside their central 99%. The stored null
holds the same records under 200 permutations, the first of them this one, so this is descriptive: it says where one
permutation fell, and check (d), not this comparison, tests the plumbing.'''
    else:
        auc_txt = (f'It is computed per dataset and averaged over the {n_rep} datasets. Its interval is the 2.5% and 97.5% '
                   f'quantiles of the mean over datasets resampled with replacement ({n_rep} datasets); it carries no '
                   f'gene-to-gene variation, because every dataset holds the same {genes} genes.')
        anchor_txt = (f'No stored 200-permutation null run exists for this gene set, so where its one permutation falls '
                      f'among permutations is not known here; it is the same permutation as the {REF_SET}\'s anchor, and '
                      f'the contrast section places it against that set\'s stored null.')
    return f'''
<h3>How the results are scored</h3>
<p><b>Intervals.</b> Unless stated, an interval is a <i>gene-clustered 95% interval</i>: the
genes are resampled with replacement {S["n_boot"]:,} times, each gene carrying all its units from the
scenario's datasets, the statistic is recomputed each time, and the 2.5% and 97.5% quantiles are reported. It
carries gene-to-gene spread, the main source of uncertainty when the same genes recur across datasets. All arms are
run on the same datasets, so their errors are correlated, and the summary carries no interval for the
difference between two arms: a gap between arms is read against each arm's own interval. Separated intervals
are then good evidence of a difference; overlapping ones do not show that two arms are equal.</p>
<p><b>Gene ranking (section 3.1).</b> Within each dataset the {genes} genes are ordered by the nominal p of their
<i>lead variant</i> (the tested variant with the smallest p; the combined statistic for hapmixQTL, the meta statistic for
mixQTL, TReCASE's one joint test). The
<b>AUC</b> (area under the receiver operating characteristic curve) is the probability that a randomly chosen
non-null gene ranks above a randomly chosen null gene: 0.5 is chance, 1 is perfect separation. {auc_txt} <b>Power at 5% realized
false-discovery proportion</b>: the gene units of the {n_rep} datasets are pooled and walked down the ranking; the
realized false-discovery proportion (FDP) at depth k is the share of the top k that are truly null, known here
from the truth; the walk stops at the deepest k with FDP &le; 0.05, and power is the share of non-null gene
units above that point. The summary gives it no interval. A within-dataset ranking uses no threshold, so a
miscalibration shared by every gene does not move it, but one specific to a gene does. Since commit 8a06803 each
hapmixQTL pair has its own t reference, so ranking hapmixQTL genes by lead p is no longer the same as
ranking them by |t|. The ranking is also confounded by the number of tested variants per gene (a null gene with many
variants has a smaller lead p by chance), a confounding every arm shares.</p>
<p><b>Gene-level power (section 3.2).</b> map_cis gives each gene a gene-level p, <b>pval_beta</b>: the lead variant's
nominal p is compared with the smallest p of each of 1,000 permutations of donor records (with L/R swaps) under the arm's
own weights, and the comparison is smoothed by fitting a Beta distribution to those permuted minima. Because each arm is
referred to its own permutation null, pval_beta absorbs whatever miscalibration of that arm's nominal p the
permutation reproduces. Total-only tensorQTL's pval_beta is the same construction on its own null (its
covariate-residualized phenotype permuted), and mixQTL's gene-level p is the empirical p of its own permutation
scan, without the Beta smoothing. The eigenMT p is a second gene-level p for every arm,
TReCASE included; it needs no permutation. The <b>Benjamini-Hochberg</b> procedure at 5% then calls genes within each dataset:
the {genes} gene-level p are sorted and the k smallest are called, k being the largest rank with
p<sub>(k)</sub> &le; 0.05 k / {genes}; with valid p values the expected share of null genes among the calls is at
most 5%. Power is the share of non-null gene units called. The table's last four columns count null gene units with
gene-level p below 0.05, permutation p / eigenMT p (a gene-level false-positive rate before any multiple-testing
correction).</p>
<p><b>Bias at the causal variant (section 3.3).</b> The <b>bias ratio</b> is the mean, over causal units, of the
estimated slope divided by the true slope at the causal variant: 1 means the injected effect is recovered on average,
0.9 means 90% of it. Two truths are
used. On the <i>count scale</i> the allelic truth is beta itself, and the total truth is the least-squares
slope, with intercept, of the exact log2 total fold on half the ALT dosage (g/2) over the dataset's 92 donors
(near beta but not equal to it, because genotype counts are asymmetric). On the <i>pipeline scale</i> (hapmixQTL
arms only) the truth is the
slope the pipeline's own transformed phenotypes would show with no noise, after the +0.5 pseudocount of
log2((L + 0.5)/(R + 0.5)) and the 0.5-read offset of the half-read total, which compress a fold at low depth. It is an unweighted
slope, so a weighted arm's bias against it still contains how the weights re-target a shift that varies with
depth. mixQTL's response has neither pseudocount nor +1, so its count-scale truth is already its own scale.
The combined slope mixes the two channels' estimands and is reported against beta only. Units whose ratio is
not finite are excluded and counted. A unit whose allelic channel has no admitted heterozygous donor returns
slope 0 with an infinite standard error; its count-scale ratio 0 / beta is finite, so it enters the count-scale
mean as 0, while its pipeline-scale truth is undefined and it is excluded there. In the table the first line is
the count scale, the second (grey) the pipeline scale, each followed by (units / excluded). The figure draws every
arm against the count-scale truth; the table's grey line carries the pipeline-scale diagnostic for hapmixQTL.</p>
<p><b>Precision (section 3.4).</b> <b>Realized over stated standard error</b>, written sd(z): z = (slope &minus; truth) /
stated se, and sd(z) is its standard deviation over units. It is 1 when the stated se equals the realized spread of the
slope, 1.2 when the realized spread is 20% larger than the se says (se too small, p too small), and below 1 when the se
is too large. This is the reciprocal of a stated-over-true ratio; it is reported as 06_score.py computes it.
Non-null units use the causal variant and the pipeline-scale truth (mixQTL: count scale; TReCASE: beta;
combined: the same inverse-variance combination of the two channel truths); in the allelic channel only the causal
units that have allelic data. Null units use every tested variant of the null genes, with truth 0. In the allelic
channel these include variants of null genes with no admitted heterozygous donor, whose output is p = 1 and
z = 0: they cannot reject and enter sd(z) as zeros, which lowers both the allelic null rate and the
allelic null sd(z). The summary does not count them. Since commit 8a06803 a gene whose allelic channel is switched off
altogether (fewer than two informative allelic donors) has a NaN allelic p and leaves the allelic null rate, where
before it counted at p = 1. The <b>mean squared error ratio against unit weights</b> (efficiency) is the sum of squared
errors under the arm divided by the same sum under unit weights, over the same units; below 1 means more
precise than unit weights. split and unit share the total channel's weights, so split's total-channel ratio is
1 by construction. Real data have no oracle variance, so this is efficiency relative to unit weights, never
against the best possible weights. For mixQTL the ratio is a comparison of methods as run: mixQTL admits a
different donor set (its count cutoffs; under the published allelic cap of 1,000 reads admission even depends
on the injected effect, because thinning pulls records down into the band), so its ratio mixes weighting with
admission. For mixQTL and TReCASE the ratio at the causal variant holds the arm and unit weights both to the
count-scale truth, so no method is ranked on the pipeline scale.</p>
<p><b>Lead-variant recovery (section 3.5).</b> For each non-null gene unit the lead variant is compared with the causal
one. <b>LD r<sup>2</sup></b> is the squared Pearson correlation (the ordinary correlation coefficient) of ALT allele
dosages over the 92 donors between the two variants (1 when they are the same variant). Reported: the share of units
whose lead is the causal variant, the share with r<sup>2</sup> &ge; 0.8, and the median r<sup>2</sup>. A unit without a
finite p counts as not recovered. The summary gives these shares no interval.</p>
<p><b>Causal-variant detection (section 3.6).</b> The share of non-null gene units whose nominal p at the causal
variant falls below 0.05, 1e-3 and 1e-5, per channel. This is power at a fixed nominal threshold, so it rewards an arm
whose p values are too small, and it is read together with the null rates of section 3.7.</p>
<p><b>Null genes and the beta = 0 anchor (section 3.7).</b> The <b>null-gene rate</b> is the share of tested variants of
null genes whose nominal p is below a threshold. In the allelic channel it includes the tests with no allelic data
(p = 1), which cannot reject. At |beta| &gt; 0 the null genes are thinned too, which adds binomial noise of the model's
own kind and is expected to dilute the real data's coupling between weights and residuals, so those rates are not
calibration results. The beta = 0 anchor is one dataset, that is ONE record permutation, with no thinning. {anchor_txt}</p>'''


def sec_run_facts():
    """What the run itself produced before any arm is compared: the datasets' facts, the arms' admission counts, the
    joint models' rows, eigenMT's structure and the input checks' outcomes. The first part of section 3."""
    n_genes = S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z']['all']['genes']
    genes = {bn: S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z'][bn]['genes'] for bn in BANDS}
    th, idn, rc, rp = CG['thinning'], CG['identity'], CG['recovery'], CG['reproduction']
    pr, tn, rl, mr = rc['primary'], th['thinned'], th['real'], rc['min_reads']
    fano = table(['haplotype-informative reads (pL + pR)', 'donor-gene pairs, thinned (real)', 'median Fano factor, thinned',
                  'median Fano factor, real'],
                 [[b, f'{tn[b]["pairs"]:,} ({rl[b]["pairs"]:,})'] + ['n/a (no pairs)' if x is None else f(x) for x in (tn[b]['fano'], rl[b]['fano'])]
                  for b in ('1-9', '10-99', '100-999', '1000+')])
    rec = table([f'allelic slope estimate ({f"genes >= {mr} reads" if mr else "every gene"})', 'mean slope / truth', 'gene-clustered se'],
                [[name, f(pr[k]['mean']), f(pr[k]['gene_clustered_se'])] for name, k in (
                    ('weights 1/Va from the unthinned record (do not depend on the effect)', 'inv_va_real_beta'),
                    ('weights 1/Va at the expected thinned counts', 'inv_va_exp_beta'),
                    ('weights 1/Va\' at the realized thinned counts (the arms\' weights), vs beta', 'inv_va_beta'),
                    ('unit weights, vs beta', 'unit_beta'),
                    ('weights 1/Va\', vs pipeline-scale truth', 'inv_va_pipeline'),
                    ('unit weights, vs pipeline-scale truth (the pass rule)', 'unit_pipeline'))])
    if rp is not None:
        if not all(v['passed'] for v in rp.values()):
            raise SystemExit(f'{C.CHECKS}: check (d) did not pass; reword section 3, run facts and checks')
        calls = lambda v: sum(sum(x['calls_differ'].values()) for x in v['pvalues'].values())   # noqa: E731
        repro = ('Check (d): the beta = 0 path reproduced the stored null\'s first permutation for ' + ', '.join(rp) + ' over '
                 f'{next(iter(rp.values()))["tests"]:,} tests each: slopes within '
                 f'{max(x["max_slope_diff_se"] for v in rp.values() for x in v["pinned"].values()):.1e} se and standard errors '
                 f'within {max(x["max_se_rel"] for v in rp.values() for x in v["pinned"].values()):.1e} relative of the stored '
                 f'ones, the three p within {max(x["max_rel"] for v in rp.values() for x in v["pvalues"].values()):.1e} relative '
                 f'with {sum(calls(v) for v in rp.values())} calls differing at 0.05, 0.01 and 0.001, dof_a, dof_t and the '
                 'allelic admission equal, and the Welch-Satterthwaite dof within '
                 f'{max(v["dof_nominal"]["max_rel"] for v in rp.values()):.1e} relative (the stored statistics are float32).')
    else:
        repro = 'Check (d) was skipped (check_generator.json has no reproduction entry).'
    prem_by_s = '; '.join(f's in {k}: {f(v["ratio_median"], 2)} ({v["genes"]:,} genes)' for k, v in CP['by_ambiguous_share'].items())
    mix = LF['mix']
    rng = lambda arm, i: '-'.join(dict.fromkeys(str(g(x[i] for x in mix[arm])) for g in (min, max)))   # noqa: E731
    tr, mp, em, td = JF['trecase'], S['mixqtl_permutation'], S['eigenmt'], LF['tdiff']
    if [td['finite_one']] != tr['constant']:
        raise SystemExit(f'tensorQTL and unit weights disagree on {td["finite_one"]} pairs, not the {tr["constant"]} of constant '
                         f'dosage; reword section 3, run facts and checks')
    miss = lambda a: ' / '.join(str(S['missing_causal'][f'beta{b}'][a]) for b in BETAS)   # noqa: E731
    if LF['floor'] and LF['floor'][0] == 0:
        floor_txt = 'No gene falls below the allelic admission floor in any dataset or arm (run_arms_facts.json).'
    elif LF['floor']:
        floor_txt = (f'In every dataset and arm the same {LF["floor"][0]} genes fall below the allelic admission floor '
                     f'({", ".join(LF["floor"][1])}; run_arms_facts.json).')
    else:
        ns = [n for n, _ in LF['floor_sets']]
        floor_txt = (f'Between {min(ns)} and {max(ns)} genes fall below the allelic admission floor, the set differing '
                     f'between datasets and arms as thinning moves genes across it (run_arms_facts.json).')
    vp = em['vs_permutation']
    calibrated = [a for a in (*HAPMIX, TQ) if a != 'gibbs']
    shape = [vp[a]['m_eff_over_shape2'] for a in HAPMIX]
    ratio = [vp[a]['eigenmt_over_pval_beta'] for a in calibrated]
    null05 = lambda a: S['null']['beta0.0'][a]['combined']['all']['0.05']   # noqa: E731
    if not (null05('gibbs')['lo'] > 0.05 and all(null05(a)['lo'] <= 0.05 for a in calibrated) and min(ratio) > 1):
        raise SystemExit('eigenMT sentence: gibbs is not the one arm above 0.05 on the beta = 0 null, or an eigenMT / pval_beta '
                         'median is at most 1; reword section 3, run facts and checks')
    em_txt = f'''M<sub>eff</sub> is {em["m_eff_min"]:,} to {em["m_eff_max"]:,} per gene (median {em["m_eff_median"]:,.0f}),
{f(em["share_min"], 2)} to {f(em["share_max"], 2)} of the tested variants. On this design shrinkage, not linkage
disequilibrium, sets M<sub>eff</sub>: with {em["donors"]} donors,
fewer than the {em["window"]} variants of a window, the window's sample correlation has rank at most {em["donors"] - 1},
and unshrunk it reaches 99% of its variance with {f(em["unshrunk_share_min"], 2)} to {f(em["unshrunk_share_max"], 2)} of
the tested variants; the Ledoit-Wolf weight, a median {f(em["lw_weight_min"], 2)} to {f(em["lw_weight_max"], 2)} per gene,
puts every eigenvalue of the shrunk matrix at about that weight or more, so 99% of the variance needs most of them. M<sub>eff</sub> is therefore a
median {f(min(shape), 1)} to {f(max(shape), 1)} times the permutation's own count of independent tests (the shape2 of the
Beta distribution map_cis fits to the hapmixQTL arms' permuted minimum p, Beta(1, M) for M independent tests), and where
pval_beta lies between {vp["unit"]["band"][0]:g} and {vp["unit"]["band"][1]:g} (the range of a dataset's Benjamini-Hochberg
thresholds) the eigenMT p is a median {f(min(ratio), 1)} to {f(max(ratio), 1)} times pval_beta for
{", ".join(SHORT[a] for a in calibrated)} ({f(vp["gibbs"]["eigenmt_over_pval_beta"], 2)} for gibbs, whose nominal p is
anticonservative, section 3.7): for an arm whose nominal p is not anticonservative the eigenMT column is conservative
relative to the permutation p (06_score.py, eigenmt_structure and eigenmt_vs_permutation).'''
    if INTERPRETED:
        constant_txt = f'''TReCASE has no row for the
{"/".join(f"{x:,}" for x in tr["constant"])} tested pairs per dataset whose ALT dosage is the same in every donor, among
them TPPP's causal variant in dataset 2 of each |beta| &gt; 0 scenario.'''
        b10 = rc['by_band']['10-99']
        rec_txt = f'''Write Va for the allelic Gibbs variance of an unthinned record
and Va' for that of a thinned record. The 1/Va' weights the arms use recover
{f(pr["inv_va_beta"]["mean"])} of beta, against {f(pr["inv_va_real_beta"]["mean"])} when the weights come from the
unthinned record's Va. Almost all of that shortfall appears when the weights are evaluated at the expected thinned
counts ({f(pr["inv_va_exp_beta"]["mean"])}), before any binomial noise. On one truth, the pipeline scale, 1/Va'
weights recover {f(pr["inv_va_pipeline"]["mean"])} (gene-clustered se {f(pr["inv_va_pipeline"]["gene_clustered_se"])})
against {f(pr["unit_pipeline"]["mean"])} ({f(pr["unit_pipeline"]["gene_clustered_se"])}) for unit weights. So 1/v
weights that follow the thinned counts attenuate the allelic slope by about 5%, and unit weights do not. One
explanation, consistent with these numbers but not tested separately: a donor whose effect thinned its already
smaller haplotype gets a larger v and a smaller weight, and it is the donor whose allelic ratio already lay in the
effect's direction before thinning; down-weighting those donors leaves the weighted mean of their pre-existing
imbalances pointing against the effect. The attenuation is a property of 1/v weighting when v tracks the counts,
which the premise check says Salmon's Gibbs variance does, not a generator defect; it applies to the allelic
channel of gibbs and split. In the 10-99 read band both weightings fall short of the
pipeline-scale truth: {f(b10["inv_va_pipeline"]["mean"])} (gene-clustered se
{f(b10["inv_va_pipeline"]["gene_clustered_se"])}) for 1/Va' and {f(b10["unit_pipeline"]["mean"])}
({f(b10["unit_pipeline"]["gene_clustered_se"])}) for unit weights, over {b10["unit_pipeline"]["genes"]} genes.
That shortfall is not decomposed; 01_check_inputs.py names one untested candidate, the zero-haplotype admission
rule, which conditions on the thinned outcome.'''
    else:
        constant_txt = (f'TReCASE has no row for the {"/".join(f"{x:,}" for x in tr["constant"])} tested pairs per dataset '
                        f'whose ALT dosage is the same in every donor.')
        nd = pr['unit_nodrop_beta']
        rec_txt = (f"The 1/Va' weights the arms use (Va' the allelic Gibbs variance of a thinned record) recover "
                   f"{f(pr['inv_va_beta']['mean'])} of beta (gene-clustered se "
                   f"{f(pr['inv_va_beta']['gene_clustered_se'])}), against {f(pr['inv_va_real_beta']['mean'])} when the "
                   f"weights come from the unthinned record's Va and {f(pr['inv_va_exp_beta']['mean'])} at the expected "
                   f"thinned counts; on the pipeline scale 1/Va' weights recover {f(pr['inv_va_pipeline']['mean'])} "
                   f"({f(pr['inv_va_pipeline']['gene_clustered_se'])}) against {f(pr['unit_pipeline']['mean'])} "
                   f"({f(pr['unit_pipeline']['gene_clustered_se'])}) for unit weights. Unit weights over every record with "
                   f"allelic information, the zero-haplotype drop not applied, recover {f(nd['mean'])} "
                   f"({f(nd['gene_clustered_se'])}) of beta.")
    return f'''
<h3>Run facts and checks</h3>
<p><b>Datasets.</b> The datasets hold {LF["pairs"][0]} donor-gene pairs, {LF["pairs"][1]} of them with
haplotype-informative reads; the genes have {LF["tested"][0]} to {LF["tested"][2]} tested variants (median
{LF["tested"][1]}), and the read bands hold {" / ".join(str(genes[b]) for b in BANDS[1:])} of the {genes["all"]} genes.
Among heterozygous donor-gene pairs of non-null genes, a share of {LF["expr"][0]} had both haplotypes at 0.5 reads or
more before thinning. Thinning removed at most {LF["expr"][2]:.1e} of the cohort's median effective library size from a
donor (median {LF["expr"][1]:.1e}).</p>
<p><b>Admission.</b> The zero-haplotype rule excluded {LF["zeroed"][0]}-{LF["zeroed"][1]} donor-gene pairs per dataset.
{floor_txt} Under the published mixQTL cutoffs {rng("mixqtl", 0)} of {n_genes} genes had at least 15 allelic
donors per dataset (median allelic donors per gene {rng("mixqtl", 2)}); under the permissive cutoffs
{rng("mixqtl_permissive", 0)} (median {rng("mixqtl_permissive", 2)}). TReCASE received the {tr["admitted"][0]:,} to
{tr["admitted"][1]:,} allele-specific records per dataset that the hapmixQTL arms admit. mixQTL's permutation scan took a median
{' and '.join(f'{v:.0f}' for v in mp["seconds_per_dataset"].values())} s per dataset with the published and permissive
cutoffs.</p>
<p><b>eigenMT.</b> {em_txt}</p>
<p><b>Total-only tensorQTL against unit weights.</b> On dataset {td["dataset"]} tensorQTL's t differs from unit
weights' total-channel t by at most {td["max_abs_diff"]:.1e} over {td["finite_both"]:,} pairs (largest |t|
{td["max_abs_t"]:.1f}; tensorQTL computes in single precision), and the {td["finite_one"]:,} pairs whose ALT dosage is
the same in every donor have no tensorQTL statistic, where unit weights' total channel returns p = 1.</p>
<p><b>Missing rows.</b> {constant_txt} asSeq refitted the dosage as a linear covariate in {tr["linear_dosage"]:,} of
{tr["tests"]:,} tests ({100 * tr["linear_dosage"] / tr["tests"]:.1f}%); how many fall at causal variants is not known.
Causal units without a TReCASE row, at |beta| = 0.2 / 0.4 / 0.8 (of
{S['lead']['beta0.4']['gibbs']['all']['units']} each): {miss('trecase')}.</p>
<p><b>Checks before the run.</b> The premise of the allelic rule: the observed excess over the predicted one has median
{f(CP["ratio_median"])} (interquartile range {f(CP["ratio_iqr"][0])} to {f(CP["ratio_iqr"][1])}) over
{CP["genes_retained"]:,} of the {CP["genes_min_u"]:,} genes ({CP["genes_excess_le_0"]} dropped for non-positive
excess, {CP["genes_s_eq_0"]} for s = 0). The ratio is not flat in s ({prem_by_s}), so the median is set
mainly by the largest group. The rule under-predicts the excess where informative reads are a larger share and over-predicts it
where almost all reads are shared; the generator's rule does not depend on s, because thinning leaves s
unchanged. The prediction ranks genes' excess with Spearman correlation (the Pearson correlation of the ranks)
{f(CP["spearman_excess_s2H"])} against {f(CP["spearman_excess_counting"])} for the counting term alone. Check (a): the
generator reproduces the pipeline's inputs exactly (largest difference in A after a permutation and swap
{idn["max_abs_dA"]:.1e}). Check (b): the allelic rule's arithmetic holds to {th["allelic_rule"]["max_rel_dev"]:.1e}
relative over {th["allelic_rule"]["records_checked"]:,} records, and the Fano factors are:</p>
{fano}
<p>Check (c), over {pr["unit_pipeline"]["genes"]} genes ({pr["unit_pipeline"]["units"]:,} units): unit weights recover the
pipeline-scale truth ({f(pr["unit_pipeline"]["mean"])}, gene-clustered se {f(pr["unit_pipeline"]["gene_clustered_se"])}),
which passes. {rec_txt}</p>
{rec}
<p>{repro}</p>'''


def gibbs_low():
    """What gibbs's pooled ranking did at |beta| 0.2 (sections 3.1 and 4)."""
    d = fdp('0.2', 'gibbs')
    if d['p_threshold'] is None:
        return ('For gibbs at |beta| 0.2 no depth of the pooled ranking kept the null share at or below 5%: null genes '
                'sat at the top of its ranking, so it called nothing.')
    return (f'For gibbs at |beta| 0.2 the null share stayed at or below 5% only down to a lead p of '
            f'{d["p_threshold"]:.1e}, so it called {f(d["all"]["power"])} of non-null units: null genes sat near the '
            f'top of its ranking.')


def interp_ranking():
    hm_ov = [b for b in BETAS if all(b in overlap(x, y) for x, y in (('split', 'unit'), ('split', 'gibbs'), ('unit', 'gibbs')))]
    need(hm_ov == list(BETAS), 'the three hapmixQTL arms\' AUC ranges overlapping at every |beta|')
    beat = [a for a in ('mixqtl', 'mixqtl_permissive', TQ, *JOINT) if auc('0.8', a)['hi'] < auc('0.8', 'split')['lo']]
    beat_txt = ('' if not beat else f' At |beta| 0.8 split\'s lowest dataset AUC ({f(auc("0.8", "split")["lo"])}) exceeds the '
                'highest of ' + ', '.join(f'{SHORT[a]} ({f(auc("0.8", a)["hi"])})' for a in beat)
                + ', so split ranks higher in each of the three datasets.')
    return f"""
<p>At |beta| = 0.2 / 0.4 / 0.8 the AUC is {A_('split')} for split, {A_('unit')} for unit and {A_('gibbs')} for gibbs;
mixQTL reaches {A_('mixqtl')} with the published cutoffs and {A_('mixqtl_permissive')} with the permissive ones, and
total-only tensorQTL {A_(TQ)}. The ranges of the three hapmixQTL arms' per-dataset AUCs overlap at every |beta|, so the
AUC does not separate the weightings.{beat_txt} A narrow range such as unit's {ci(auc('0.4', 'unit'), 'mean')} at
|beta| 0.4 means three datasets happened to agree, not that the estimate is precise.</p>
<p>Power at 5% realized FDP spreads the arms further: split {P_('split')}, unit {P_('unit')}, gibbs {P_('gibbs')};
mixQTL {P_('mixqtl')} (published) and {P_('mixqtl_permissive')} (permissive); tensorQTL {P_(TQ)}. {gibbs_low()} At 0.4
and 0.8 its cut fell at a lead p of {thr('0.4', 'gibbs')} and {thr('0.8', 'gibbs')}, where split's fell at
{thr('0.4', 'split')} and {thr('0.8', 'split')}: null genes' lead p reached below split's cut, so to keep them out
gibbs had to stop at smaller p. Section 4 takes up whether that reflects null p values that are too small.</p>"""


def interp_gene_level():
    P = lambda a: per_beta(lambda b: bh(b, a)['power_bh']['all']['rate'])   # noqa: E731
    N = lambda a: ' / '.join(f'{bh(b, a)["false_discoveries"]} of {bh(b, a)["discoveries"]}' for b in BETAS)   # noqa: E731
    pw = lambda b, a: bh(b, a)['power_bh']['all']   # noqa: E731
    ov = all(pw(b, x)['lo'] <= pw(b, y)['hi'] and pw(b, y)['lo'] <= pw(b, x)['hi']
             for b in BETAS for x, y in (('split', 'unit'), ('split', 'gibbs'), ('unit', 'gibbs')))
    need(ov, 'the three hapmixQTL arms\' Benjamini-Hochberg power intervals overlapping at every |beta|')
    E = lambda a: per_beta(lambda b: bhe(b, a)['power_bh']['all']['rate'])   # noqa: E731
    EF = lambda a: per_beta(lambda b: bhe(b, a)['fdp_matched']['all']['power'])   # noqa: E731
    return f"""
<p>With each arm referred to its own permutation null, Benjamini-Hochberg power at |beta| = 0.2 / 0.4 / 0.8 is
{P('split')} for split, {P('gibbs')} for gibbs and {P('unit')} for unit. The gene-clustered intervals of the three arms
overlap at every |beta| (table), so at {S['lead']['beta0.4']['split']['all']['units']} non-null gene units per effect size
the weightings are not distinguished on gene-level power. Null genes among the calls: split {N('split')}, gibbs
{N('gibbs')}, unit {N('unit')}; these are small counts without an interval. The generator's null and map_cis's null are
the same record permutation with label swaps, so the permutation p of a null gene is valid by construction and the
share of null gene units with pval_beta below 0.05 (last columns of the table) checks the plumbing and the Beta
approximation, not calibration on real data. On its own permutation p total-only tensorQTL reaches {P(TQ)} and mixQTL
{P('mixqtl')} (published) and {P('mixqtl_permissive')} (permissive); their permutation nulls are their own (section 2).</p>
<p><b>eigenMT.</b> On the eigenMT p, which multiplies an arm's smallest nominal p by M<sub>eff</sub> and so inherits that
nominal p's calibration, Benjamini-Hochberg power is split {E('split')}, gibbs {E('gibbs')}, unit {E('unit')}, TReCASE
{E('trecase')}. Held instead to 5% realized FDP on the same p (Figure 1 D) it is split {EF('split')}, gibbs {EF('gibbs')},
unit {EF('unit')}, TReCASE {EF('trecase')}: an arm whose nominal p is anticonservative gains on the first and not on the
second, and Figure 1 E gives each arm's realized false-discovery proportion of its eigenMT calls.</p>"""


def interp_bias():
    tot = lambda a, key='bias_count': [bias(b, a, 'total', key) for b in BETAS]   # noqa: E731
    need(all(ivl(d) == 'includes' for a in HAPMIX for d in tot(a)), 'every hapmixQTL total slope\'s count-scale interval including 1')
    al_ov = all(bias(b, 'split', 'allelic', 'bias_pipeline')['lo'] <= bias(b, 'unit', 'allelic', 'bias_pipeline')['hi']
                and bias(b, 'unit', 'allelic', 'bias_pipeline')['lo'] <= bias(b, 'split', 'allelic', 'bias_pipeline')['hi'] for b in BETAS)
    need(al_ov, 'split\'s and unit\'s pipeline-scale allelic bias intervals overlapping at every |beta|')
    pr = CG['recovery']['primary']
    n_c = S['recovery']['beta0.4']['split']['allelic']['bias_count']['all']['units']
    n_p = S['recovery']['beta0.4']['split']['allelic']['bias_pipeline']['all']['units']
    mq, mp = (S['recovery']['beta0.4'][a]['allelic']['bias_count']['all'] for a in ('mixqtl', 'mixqtl_permissive'))
    lo17, hi17 = all17()
    return f"""
<p><b>Total channel.</b> On the half-read total split and unit, which share one total-channel fit, recover
{B_('split', 'total', n=3)} of the count-scale truth and {B_('split', 'total', 'bias_pipeline', n=3)} of the pipeline-scale
truth at |beta| = 0.2 / 0.4 / 0.8, and gibbs {B_('gibbs', 'total', n=3)} and {B_('gibbs', 'total', 'bias_pipeline', n=3)};
every count-scale interval includes 1. The two truths differ little because the half-read total's offset is half a read,
which compresses a fold only where a donor has few reads.</p>
<p><b>Allelic channel.</b> gibbs and split fit the allelic channel identically (every allelic row of the two arms is the
same fit). Their allelic slope recovers {B_('split', 'allelic', 'bias_pipeline', n=3)} of the pipeline-scale truth,
against {B_('unit', 'allelic', 'bias_pipeline', n=3)} for unit weights; the intervals overlap at every |beta|, so the
benchmark does not resolve a bias difference between 1/Va and unit weights. The roughly 5% attenuation from 1/Va weights
rests on check (c) (section 3, run facts and checks): {f(pr['inv_va_pipeline']['mean'])} against
{f(pr['unit_pipeline']['mean'])} for unit weights on the pipeline-scale truth.</p>
<p><b>Two denominators in the allelic rows.</b> In {n_c - n_p} of the {n_c} causal units per |beta| the allelic channel
has no admitted heterozygous donor: map_nominal returns slope 0 with an infinite standard error. Their pipeline-scale
truth is undefined, so the pipeline line drops them ({n_p} units), while the count line keeps them as exact zeros
({n_c} units), which lowers the count-scale mean by the factor {n_p}/{n_c}; the allelic sd(z) and efficiency at the
causal variant (section 3.4) use the {n_p} units with data. mixQTL's count-scale allelic rows drop their units without
an estimate, so they are on a different filter.</p>
<p><b>Combined slope.</b> Against beta it reads {B_('split', 'combined', n=3)} for split, {B_('unit', 'combined', n=3)}
for unit and {B_('gibbs', 'combined', n=3)} for gibbs. It mixes the two channels' estimands, so it is not a measure of
estimator bias on its own.</p>
<p><b>mixQTL.</b> Its total slope recovers {B_('mixqtl', 'total', n=3)} (published) and
{B_('mixqtl_permissive', 'total', n=3)} (permissive) of the count-scale truth. Section 3.8 explains most of it: mixQTL
fits its covariate offset from the covariates alone, before the variant enters, and then regresses on the genotype
without adjusting it for those covariates, which multiplies the slope by one minus the share of the genotype's variance
they explain, and it chooses those covariates on the outcome; a one-step fit with all 17 covariates on mixQTL's response
recovers {f(lo17, 2)} to {f(hi17, 2)} of the truth. Its allelic slope recovers {B_('mixqtl', 'allelic', n=3)} (published)
and {B_('mixqtl_permissive', 'allelic', n=3)} (permissive); at |beta| 0.4 the published cutoffs leave
{mq['excluded_nonfinite']} of {mq['units'] + mq['excluded_nonfinite']} allelic units without a finite estimate and the
permissive cutoffs {mp['excluded_nonfinite']}.</p>"""


def interp_precision():
    tz = lambda a: [prec(f'beta{b}', a, 'total', 'nonnull', 'sd_z') for b in BETAS]   # noqa: E731
    az = lambda a: [prec(f'beta{b}', a, 'allelic', 'nonnull', 'sd_z') for b in BETAS]   # noqa: E731
    need(all(ivl(d) == 'above' for d in tz('gibbs')) and all(ivl(d) == 'includes' for d in tz('split')),
         'gibbs\'s total sd(z) above 1 and split\'s including 1 at every |beta|')
    need(all(ivl(d) == 'includes' for a in HAPMIX for d in az(a)), 'every hapmixQTL allelic sd(z) interval including 1')
    cs = [prec(f'beta{b}', 'split', 'combined', 'nonnull', 'ratio_vs_unit') for b in BETAS]
    cg = [prec(f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit') for b in BETAS]
    tg = [prec(f'beta{b}', 'gibbs', 'total', 'nonnull', 'ratio_vs_unit') for b in BETAS]
    need(all(ivl(d) == 'below' for d in cs) and ivl(prec('beta0.0', 'split', 'combined', 'null', 'ratio_vs_unit')) == 'below',
         'split\'s combined squared error below unit weights\' at every |beta| and on the anchor')
    need(all(ivl(d) == 'above' for d in cg + tg), 'gibbs\'s combined and total squared error above unit weights\' at every |beta|')
    tq = [prec(f'beta{b}', TQ, 'combined', 'nonnull', 'ratio_vs_unit') for b in BETAS]
    need(all(ivl(d) == 'includes' for d in tq) and ivl(prec('beta0.0', TQ, 'combined', 'null', 'ratio_vs_unit')) == 'above',
         'tensorQTL\'s causal-variant ratio including 1 at every |beta| and its anchor ratio above 1')
    band = lambda a, ch, sc, part: ' / '.join(ci(prec(sc, a, ch, part, 'ratio_vs_unit', bn), 'value', 2) for bn in BANDS[1:])   # noqa: E731
    one = one_df_gene()
    w1 = lambda a: ci(prec('beta0.0', a, 'allelic', 'null', 'sd_z', NO_ONE_DF), 'value', 3)   # noqa: E731
    return f"""
<p><b>Stated standard error.</b> In the total channel at the causal variant sd(z) is {Z_('split', 'total')} for split and
unit, whose intervals include 1, and {Z_('gibbs', 'total')} for gibbs, whose intervals lie above 1 (lower bounds
{' / '.join(f(d['lo'], 3) for d in tz('gibbs'))}); on the anchor's null genes gibbs reads {Zn_('gibbs', 'total')} and unit
{Zn_('unit', 'total')}. So weighting the half-read total by its Gibbs variance makes the total slope vary more than its
stated standard error says, and unit working variance states it correctly. In the allelic channel the point value of
sd(z) at the causal variant is above 1 for all three arms (gibbs and split {Z_('split', 'allelic')}, unit
{Z_('unit', 'allelic')}), but every interval includes 1; at the causal variant sd(z) absorbs bias whose sign follows the
random sign of beta, so the excess may be bias rather than a stated error that is too small, and the two were not
separated. On the anchor's null genes the allelic channel reads {Zn_('split', 'allelic')} for gibbs and split and
{Zn_('unit', 'allelic')} for unit; much of that excess is one null gene, {one}, whose allelic scale is fitted on two
donors, and without it they read {w1('split')} and {w1('unit')}.</p>
<p><b>Efficiency.</b> In the allelic channel the gibbs and split squared error is {E_('split', 'allelic')} of unit
weights' at the causal variant and {En_('split', 'allelic')} on the anchor's null genes; by read band
({BAND_HTML}, |beta| 0.4) it is {band('split', 'allelic', 'beta0.4', 'nonnull')}. In the total channel gibbs's squared
error is {E_('gibbs', 'total')} of unit weights' at the causal variant and {En_('gibbs', 'total')} on the anchor's null
genes, with intervals above 1 at every |beta|; by band at |beta| 0.4 it is {band('gibbs', 'total', 'beta0.4', 'nonnull')}.
For the combined slope split's squared error is {' / '.join(ci(d, 'value', 2) for d in cs)} of unit weights' at the
causal variant, every interval below 1, and {En_('split', 'combined')} on the anchor's null genes; gibbs's is
{' / '.join(ci(d, 'value', 2) for d in cg)}, every interval above 1, and {En_('gibbs', 'combined')} on the anchor. So the
Gibbs draws buy precision in the allelic channel and cost it in the total channel, which is the split the shipped
default makes.</p>
<p><b>Total-only tensorQTL</b> against unit weights' combined slope: {' / '.join(ci(d, 'value', 2) for d in tq)} at the
causal variant, every interval including 1, and {En_(TQ, 'combined')} on the anchor's null genes. The gap between the
two was not decomposed.</p>
<p><b>mixQTL</b> is compared as run (section 2). The published arm's combined ratio is {E_('mixqtl', 'combined')} at the
causal variant and {En_('mixqtl', 'combined')} on the anchor; the permissive arm's {E_('mixqtl_permissive', 'combined')}
and {En_('mixqtl_permissive', 'combined')}. Section 3.8 decomposes the total channel's part of this gap.</p>"""


def interp_lead():
    lo = lambda a: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['lead_is_causal'], 2)   # noqa: E731
    low = [b for b in BETAS if S['lead'][f'beta{b}']['mixqtl']['all']['r2_high']
           < min(S['lead'][f'beta{b}'][a]['all']['r2_high'] for a in HAPMIX + ('mixqtl_permissive', TQ) + JOINT)]
    return f"""
<p>The share of non-null gene units whose lead is within r<sup>2</sup> &ge; 0.8 of the causal variant is {R_('split')}
for split, {R_('unit')} for unit and {R_('gibbs')} for gibbs; mixQTL reaches {R_('mixqtl')} (published) and
{R_('mixqtl_permissive')} (permissive), and tensorQTL {R_(TQ)}. The lead is the causal variant itself in {lo('split')}
of units for split. The summary carries no interval for these shares, so differences among the arms are not
interpreted{', except that mixQTL with the published cutoffs has the lowest point value at ' + at_betas(low) if low else ''}.</p>"""


def interp_detection():
    st = lambda a, ch: ci(nul(a, ch, '0.001'), 'rate', 4)   # noqa: E731
    return f"""
<p>At p &lt; 1e-3 the combined statistic detects the causal variant in {D_('split')} of non-null gene units for split,
{D_('gibbs')} for gibbs and {D_('unit')} for unit; mixQTL {D_('mixqtl')} (published) and {D_('mixqtl_permissive')}
(permissive); tensorQTL {D_(TQ)}. The allelic channel shows the weights most directly: gibbs and split
{D_('split', 'allelic')} against unit {D_('unit', 'allelic')}. Detection at a fixed nominal threshold rewards a p that is
too small, and gibbs's is: on the stored null its combined rate at 1e-3 is {st('gibbs', 'combined')} and its total
channel's {st('gibbs', 'total')}, against {st('split', 'combined')} and {st('unit', 'combined')} for split and unit
(section 3.7), so its detections are not comparable at face value.</p>"""


def interp_null():
    sv = lambda a, ch, al='0.05': nul(a, ch, al)   # noqa: E731
    r3 = {a: nul(a, 'combined', '0.001') for a in HAPMIX}
    need(r3['split']['lo'] <= 0.001 <= r3['split']['hi'] and not (r3['unit']['lo'] <= 0.001 <= r3['unit']['hi']),
         'the stored null\'s pass rule holding for split and failing for unit')
    out = [(a, ch, S['anchor'][a][ch]['0.05']) for a in HAPMIX for ch in CHANNELS if not S['anchor'][a][ch]['0.05']['passed']]
    pct = lambda ch: [S['anchor'][a][ch]['0.05']['percentile'] for a in HAPMIX]   # noqa: E731
    rng = lambda v: f'{min(v):.1f}-{max(v):.1f}th'   # noqa: E731
    bench = lambda a, ch: [S['null'][f'beta{b}'][a][ch]['all']['0.05'] for b in BETAS]   # noqa: E731
    covers = lambda a, ch: all(d['lo'] <= sv(a, ch)['rate'] <= d['hi'] for d in bench(a, ch))   # noqa: E731
    outside = '; '.join(f'{a} {ch}, {f(r["rate"], 4)} at the {r["percentile"]:.1f}th percentile, against a central 99% of '
                        f'{f(r["perm_lo"], 4)} to {f(r["perm_hi"], 4)}' for a, ch, r in out) or 'none'
    g1 = one_df_gene()
    need(all(S['null'][sc]['gibbs'][ch]['all']['0.05']['lo'] > 0.05 for sc in S['null'] for ch in ('combined', 'total')),
         'gibbs\'s combined and total null-rate intervals above 0.05 in every scenario')
    return f"""
<p>At 0.05, over the anchor and |beta| = 0.2 / 0.4 / 0.8: split combined {N_('split')}; unit combined {N_('unit')} and
allelic {N_('unit', 'allelic')}; gibbs combined {N_('gibbs')} and total {N_('gibbs', 'total')}, with gene-clustered
intervals above 0.05; mixQTL combined {N_('mixqtl')} (published) and {N_('mixqtl_permissive')} (permissive); tensorQTL
{N_(TQ)}.</p>
<p><b>The stored null.</b> Over its 200 permutations the combined rate at 0.05 is {ci(sv('split', 'combined'), 'rate', 4)}
for split, {ci(sv('unit', 'combined'), 'rate', 4)} for unit and {ci(sv('gibbs', 'combined'), 'rate', 4)} for gibbs, and the
total channel {ci(sv('split', 'total'), 'rate', 4)} under unit working variance (split and unit) against
{ci(sv('gibbs', 'total'), 'rate', 4)} under gibbs's 1/Vt. The rule stated before it ran (the combined rate at 0.001 within
its gene-clustered interval of 0.001, {g1} included) holds for split, {ci(r3['split'], 'rate', 5)}, and fails narrowly for
unit, {ci(r3['unit'], 'rate', 5)}. The two arms share one total channel, at {ci(nul('unit', 'total', '0.001'), 'rate', 5)},
and differ in the allelic one, at {ci(nul('split', 'allelic', '0.001'), 'rate', 5)} for split's 1/Va-weighted fit against
{ci(nul('unit', 'allelic', '0.001'), 'rate', 5)} for unit's. gibbs's combined rate at 0.001 is {ci(r3['gibbs'], 'rate', 5)}, its
total channel's {ci(nul('gibbs', 'total', '0.001'), 'rate', 5)}. {g1}, with two allelic donors, is below the
{MIN_ALLELIC_DONORS}-donor floor and tested on its total channel alone.</p>
<p><b>Rates at |beta| &gt; 0.</b> Thinning adds binomial noise of the model's own kind and is expected to dilute the real
data's coupling between weights and residuals, pulling these rates toward nominal; they are not calibration results.
gibbs's total-channel rates at 0.05 at |beta| &gt; 0 ({per_beta(lambda b: S['null'][f'beta{b}']['gibbs']['total']['all']['0.05']['rate'], 4)})
{'include' if covers('gibbs', 'total') else 'do not all include'} the stored null's {f(sv('gibbs', 'total')['rate'], 4)}
in their intervals, so the dilution is {'not seen' if covers('gibbs', 'total') else 'not resolved'} here.</p>
<p><b>The anchor.</b> Its one permutation sits at the {rng(pct('total'))} percentile of the stored per-permutation
total-channel rates in the three arms, the {rng(pct('allelic'))} of the allelic and the {rng(pct('combined'))} of the
combined ones, and outside the central 99% for: {outside}. The anchor is permutation 0 of the stored stream, so this
says where one permutation fell, a low draw for the total channel in all three arms; check (d), not this placement,
tests the plumbing.</p>"""


def where(d, x):
    """Where an interval d lies against x."""
    return 'above' if d['lo'] > x else 'below' if d['hi'] < x else 'includes'


def sec_contrast():
    """This gene set against the 100-gene run (REF_RUN): every gene, each run with its own interval."""
    R = json.loads(REF_RUN.read_text())
    if tuple(R['arms']) != ARMS or tuple(R['joint_arms']) != JOINT or R['n_boot'] != S['n_boot']:
        raise SystemExit(f'{REF_RUN}: not scored as {C.SUMMARY} is')
    runs = ((THIS_SET, S), (REF_SET, R))
    n_rep = {n: X['n_datasets']['0.4'] for n, X in runs}
    au = lambda X, b, a: X['ranking'][f'beta{b}'][a]['auc']['all']   # noqa: E731  auc, fdp and overlap of either run
    fd = lambda X, b, a: X['ranking'][f'beta{b}'][a]['fdp_matched']   # noqa: E731
    ov = lambda X, a, b0: [x for x in BETAS if au(X, x, a)['lo'] <= au(X, x, b0)['hi'] and au(X, x, b0)['lo'] <= au(X, x, a)['hi']]   # noqa: E731
    null = lambda X, a, al: X['null']['beta0.0'][a]['combined']['all'][al]   # noqa: E731
    t_null = table(['arm, combined statistic'] + [f'{n}, {al}' for n, _ in runs for al in ALPHAS],
                   [[LABEL[a]] + [ci(null(X, a, al), 'rate', 4) for _, X in runs for al in ALPHAS] for a in ALL])
    lst = lambda X, al, w: ', '.join(SHORT[a] for a in ALL if where(null(X, a, al), float(al)) == w) or 'none'   # noqa: E731
    cal = '; '.join(f'at {al}, the interval lies above {al} for {lst(S, al, "above")} in the {THIS_SET} and for '
                    f'{lst(R, al, "above")} in the {REF_SET}, and below it for {lst(S, al, "below")} in the {THIS_SET} '
                    f'and {lst(R, al, "below")} in the {REF_SET}' for al in ALPHAS)
    P = lambda X, sc, a, ch, part, k: X['precision'][sc][a][ch][part][k]['all']   # noqa: E731
    rows = [(a, ch) for a in ('gibbs', 'split') for ch in CHANNELS if (a, ch) != ('split', 'total')]
    rows += [(a, 'combined') for a in ('mixqtl', 'mixqtl_permissive', TQ) + JOINT]
    t_prec = table(['arm', 'channel'] + [f'{n}, {c}' for n, _ in runs for c in (
        'causal variant, |beta| 0.4', 'null genes, beta 0 anchor')],
                   [[LABEL[a], ch_name(a, ch)] + [x for _, X in runs for x in (
                       ci(P(X, 'beta0.4', a, ch, 'nonnull', rkey(a)), 'value', 2),
                       ci(P(X, 'beta0.0', a, ch, 'null', 'ratio_vs_unit'), 'value', 2))] for a, ch in rows])
    sp = lambda X, sc, part, k: P(X, sc, 'split', 'combined', part, k)   # noqa: E731
    t_rank = table(['arm'] + [f'{n}, |beta| {b}' for n, _ in runs for b in BETAS],
                   [[LABEL[a]] + [f'{ci(au(X, b, a), "mean")}<br>{f(fd(X, b, a)["all"]["power"])}'
                                  for _, X in runs for b in BETAS] for a in ALL])
    def gl(X, k, b, a):   # a gene-level power entry; an eigenMT entry also with its calls and the null gene units among them
        d = X[k][f'beta{b}'].get(a)
        if d is None:
            return 'n/a'
        return ci(d['power_bh']['all'], 'rate') + (f' ({d["discoveries"]} called, {d["false_discoveries"]} null)'
                                                   f'<br>{f(d["fdp_matched"]["all"]["power"])} at 5% realized FDP'
                                                   if k == 'gene_level_eigenmt' else '')
    efd = lambda X, a: X['gene_level_eigenmt']['beta0.4'][a]   # noqa: E731
    fdp_e = lambda X, a: efd(X, a)['false_discoveries'] / efd(X, a)['discoveries']   # noqa: E731  realized FDP of the eigenMT calls
    fdp_txt = lambda X, a: f'{f(fdp_e(X, a), 2)} ({efd(X, a)["false_discoveries"]} of {efd(X, a)["discoveries"]})'   # noqa: E731
    high = ('gibbs', 'mixqtl', 'mixqtl_permissive', 'trecase')   # the arms whose eigenMT calls carry a high null share (review 2026-09-27)
    other = lambda X: max((a for a in ALL if a not in high), key=lambda a: fdp_e(X, a))   # noqa: E731
    fdp_sentence = (' Benjamini-Hochberg at 5% bounds the expected false-discovery proportion by 5% only when the gene-level '
                    'p is not anticonservative: at |beta| 0.4 the realized false-discovery proportion of the eigenMT calls '
                    '(null gene units among the gene units called, over the datasets) is '
                    + ', '.join(f'{SHORT[a]} {fdp_txt(S, a)}' for a in high) + f' in the {THIS_SET} and '
                    + ', '.join(f'{SHORT[a]} {fdp_txt(R, a)}' for a in high) + f' in the {REF_SET}, against at most '
                    f'{fdp_txt(S, other(S))} and {fdp_txt(R, other(R))} for the other arms; the eigenMT entries of the '
                    f'second table give the calls and null gene units at every |beta|, and on a second line the power when '
                    f'every arm is held to the same 5% realized false-discovery proportion on its eigenMT p, which is the '
                    f'comparison across arms.')
    t_gene = table(['arm'] + [f'{n}, |beta| {b}, {c}' for n, _ in runs for b in BETAS for c in ('permutation p', 'eigenMT p')],
                   [[LABEL[a]] + [gl(X, k, b, a) for _, X in runs for b in BETAS for k in ('gene_level', 'gene_level_eigenmt')]
                    for a in ALL])
    auc_iv = ('the 2.5% and 97.5% quantiles of the mean over the datasets resampled with replacement'
              + (', which with three datasets are the smallest and largest of the three per-dataset values, not a 95% '
                 'interval' if set(n_rep.values()) == {3} else ''))

    def h2h(a, X):
        vs = lambda al: f'{ci(null(X, a, al), "rate", 4)} against {ci(null(X, "split", al), "rate", 4)}'   # noqa: E731
        sep = lambda al: ('separated' if null(X, a, al)['lo'] > null(X, 'split', al)['hi']   # noqa: E731
                          or null(X, a, al)['hi'] < null(X, 'split', al)['lo'] else 'overlapping')
        return (f'AUC {per_beta(lambda b: au(X, b, a)["mean"])} against split\'s '
                f'{per_beta(lambda b: au(X, b, "split")["mean"])}, the per-dataset AUC ranges overlapping at '
                f'{at_betas(ov(X, a, "split"))}; power at 5% realized FDP {per_beta(lambda b: fd(X, b, a)["all"]["power"])} '
                f'against {per_beta(lambda b: fd(X, b, "split")["all"]["power"])} (no interval); squared error against unit '
                f'weights on the count-scale truth at |beta| 0.8 {ci(P(X, "beta0.8", a, "combined", "nonnull", "ratio_vs_unit_count"), "value", 2)} '
                f'against split\'s {ci(sp(X, "beta0.8", "nonnull", "ratio_vs_unit_count"), "value", 2)}; anchor null rate at '
                f'0.05 {vs("0.05")} ({sep("0.05")}) and at 0.001 {vs("0.001")} ({sep("0.001")})')
    b_max = max(BETAS, key=float)
    rec = lambda X, a: X['recovery'][f'beta{b_max}'][a]['combined']['bias_count']['all']['mean']   # noqa: E731  recovered share of the count-scale truth

    def scaled(X):   # the anchor ratio against unit weights, put on unit weights' slope scale
        return ', '.join(f'{LABEL[a]} {P(X, "beta0.0", a, "combined", "null", "ratio_vs_unit")["value"] * (rec(X, "unit") / rec(X, a)) ** 2:.2f} '
                         f'(table {ci(P(X, "beta0.0", a, "combined", "null", "ratio_vs_unit"), "value", 2)}; recovered share '
                         f'{f(rec(X, a), 2)} against unit weights\' {f(rec(X, "unit"), 2)})' for a in JOINT)
    pct = ' / '.join(f'{R["anchor"][a]["total"]["0.05"]["percentile"]:g}' for a in HAPMIX) + f' ({" / ".join(SHORT[a] for a in HAPMIX)})'
    joint = ''.join(f'<p><b>{LABEL[a]} against split.</b> In the {THIS_SET}: {h2h(a, S)}. In the {REF_SET}: {h2h(a, R)}.</p>'
                    for a in JOINT) + (
        f'<p>On the count-scale truth unit weights\' squared error contains their own attenuation of the total slope '
        f'(section 3.3), which grows with |beta|, so a ratio there is squared error, not precision. On the anchor\'s null '
        f'genes the truth is 0, so that attenuation is gone, but each method\'s slope scale is not: squared null error grows '
        f'with the square of the slope scale, so the anchor ratio is exact among the hapmixQTL weightings, which share one '
        f'phenotype scale, and not across methods. Multiplying each cross-method anchor ratio by (unit weights\' recovered '
        f'share of the count-scale truth at |beta| {b_max} over the arm\'s)<sup>2</sup> puts it on unit weights\' scale: in '
        f'the {THIS_SET} {scaled(S)}; in the {REF_SET} {scaled(R)}.</p>')
    H = SF['half']
    limit = (f'<p><b>Limit, from the Salmon half-depth test.</b> This set\'s genes lie in the {H["band"]}-read band of the test of the '
             f'thinning rule against Salmon itself (donor 100 re-quantified from a fraction f = {H["f"]} of its reads; {HALF_DEPTH}), '
             f'the band where the rule was least faithful'
             f'{" and failed its pass band " + str(H["pass_band"]) if H["failed"] else ""}: the allelic Gibbs variance Salmon '
             f'produced was {H["va"]["median"]:.2f} of what the rule predicts (median over {H["va"]["n"]:,} donor-gene pairs; 95% '
             f'interval of the median {H["va"]["median_ci95"][0]:.2f} to {H["va"]["median_ci95"][1]:.2f}); as depth fell the Gibbs '
             f'variance grew as (1/depth)<sup>{H["exponent"]["median"]:.2f}</sup> (median over {H["exponent"]["n"]:,} pairs) rather '
             f'than (1/depth)<sup>1</sup>; '
             f'{H["became_one_sided"]["half"]} of {H["two_sided"]:,} two-sided pairs became one-sided at half depth against '
             f'{H["became_one_sided"]["thinned"]} under thinning; and the half-depth allelic ratio regressed on the full-depth ratio '
             f'with slope {H["attenuation"]["half"]["slope"]:.2f} against {H["attenuation"]["thinned"]["slope"]:.2f} under thinning. '
             f'The thinned records of these datasets therefore carry more allelic Gibbs variance, fewer zero-haplotype records and less '
             f'attenuated allelic ratios than Salmon would produce at the same depths, most for the haplotypes thinned hardest '
             f'(f = 2<sup>-{b_max}</sup> = {2 ** -float(b_max):.2f}), so in the |beta| &gt; 0 datasets (causal-variant '
             f'precision and bias, ranking, power, and the thinned null genes of section 3.7) every arm\'s allelic-channel figures '
             f'are optimistic relative to real data of this set by an amount this run does not measure. The beta = 0 anchor is '
             f'not thinned (f = 1 on both haplotypes), so the calibration table above is free of this limit, though not of the one '
             f'shared permutation. The comparison among arms, which share the same thinned input, is affected less than any arm\'s '
             f'absolute figures.</p>')
    anc = lambda X, a, ch: P(X, 'beta0.0', a, ch, 'null', 'ratio_vs_unit')   # noqa: E731
    apart = lambda x, y: 'separated' if x['lo'] > y['hi'] or x['hi'] < y['lo'] else 'overlapping'   # noqa: E731
    gap = per_beta(lambda b: au(S, b, 'split')['mean'] - au(S, b, 'trecase')['mean'], 3)
    lower = [al for al in ('0.05', '0.001') if all(null(X, 'split', al)['hi'] < null(X, 'trecase', al)['lo'] for X in (S, R))]
    cal_vs = (f'split\'s anchor rate is below TReCASE\'s with separated intervals at {" and ".join(lower)} in both runs'
              if lower else 'split\'s and TReCASE\'s anchor rates are not separated at 0.05 or 0.001 in both runs')
    settled = (f'<p><b>What the {THIS_SET} settles, and what it cannot.</b> Under the one shared permutation the combined nominal '
               f'p has an interval that includes nominal for {lst(S, "0.05", "includes")} at 0.05, {lst(S, "0.01", "includes")} at '
               f'0.01 and {lst(S, "0.001", "includes")} at 0.001, and one above nominal for {lst(S, "0.05", "above")} at 0.05, '
               f'{lst(S, "0.01", "above")} at 0.01 and {lst(S, "0.001", "above")} at 0.001; a stored null for this gene set is '
               f'needed before "nominal" means more than "nominal on this permutation". split\'s gain over unit weights is smaller '
               f'here than in the {REF_SET}: {ci(anc(S, "split", "combined"), "value", 2)} against '
               f'{ci(anc(R, "split", "combined"), "value", 2)} on the anchor ({apart(anc(S, "split", "combined"), anc(R, "split", "combined"))} '
               f'intervals), its allelic channel alone {ci(anc(S, "split", "allelic"), "value", 2)} against '
               f'{ci(anc(R, "split", "allelic"), "value", 2)}, because at these depths the total channel carries most of the '
               f'combined slope. Against TReCASE, {cal_vs} (the paragraphs above); its AUC ranges overlap TReCASE\'s at '
               f'{at_betas(ov(S, "trecase", "split"))} with a mean '
               f'difference (split minus TReCASE) of {gap} at |beta| {" / ".join(BETAS)}, and the precision comparison rests on '
               f'the scale correction above. More |beta| replicates would sharpen the ranking comparison only; they cannot '
               f'replace the stored null. The AUC intervals are the range of {n_rep[THIS_SET]} datasets.</p>')
    fc = dict(cal=fig_contrast_calibration(runs), prec=fig_contrast_precision(runs, [r for r in rows if r[0] != 'mixqtl']),
              rank=fig_contrast_ranking(runs))   # published cutoffs: in the table only (user decision 2026-09-28)
    return f'''
<h2>The {THIS_SET} against the {REF_SET}</h2>
<p>Every gene of each set, each value with its own set's interval; no interval of the difference is computed. The
{REF_SET} is its delivered run ({REF_RUN}) on the datasets of the first simulated-effects run ({FIRST_RUN.name}), with Meier's
correction as here{'' if REF_COV == ARMS_COV else f', but with its arms on {REF_COV.name} ({REF_COV_HOW}) where this run' + "'" + f's used {ARMS_COV.name}: the two sets' + "'" + ' total and combined channels are compared across covariate builds, so those comparisons are not like for like (' + LIMITS + ')'}: the same benchmark on the {INTERPRETED_SET} genes, {SF["ref_above"]} of
which lie above this set's read range, scored by the same 06_score.py with {n_rep[REF_SET]} datasets per |beta|; the
{THIS_SET} has {n_rep[THIS_SET]}. The two runs are not independent: the generator's streams are keyed on the replicate
index alone, so both anchors carry {'the same' if SF['same_perm'] else 'DIFFERENT'} record permutation, label swaps and
null assignment by gene index (checked on the two beta = 0 files), the |beta| &gt; 0 replicates likewise, and
{len(SF['shared'])} gene{'s' if len(SF['shared']) != 1 else ''} ({', '.join(SF['shared']) or 'none'})
{'are' if len(SF['shared']) != 1 else 'is'} in both sets. Intervals that do not overlap are evidence of a difference
between the gene sets under one shared permutation. Unless stated, an interval is a <i>gene-clustered 95% interval</i>:
the genes are resampled with replacement {S["n_boot"]:,} times, each gene carrying all its units from the scenario's
datasets, the statistic is recomputed each time, and the 2.5% and 97.5% quantiles are reported; it carries the
gene-to-gene spread. Every statistic is defined at its first use below; section 3 gives the {THIS_SET}'s values by read
band.</p>
<p><b>Calibration of the nominal p on the beta = 0 anchor.</b> The <i>nominal p</i> is each arm's per-variant p under
its own reference distribution (section 2: t references for hapmixQTL's channels and their combination, a normal
reference for mixQTL's meta statistic, a chi-squared likelihood-ratio reference for TReCASE). The
null-gene rate is the share of the null genes' tested variants whose combined nominal p falls below the threshold; on
null genes it should equal the threshold. The anchor is one dataset, that is ONE record
permutation with no thinning, so its interval is gene-clustered only (genes resampled with replacement) and carries no
permutation-to-permutation spread; the thinned null genes of the |beta| &gt; 0 datasets are in section 3.7. On the 100
genes, where 200 stored permutations exist, this same permutation's total-channel rate at 0.05 sat at percentile
{pct} of theirs, so a low draw there may be a low draw here, and this set's rates that include nominal may be low by a
margin only a stored null for this gene set can measure. By the intervals: {cal}.</p>
{t_null}
{img(fc['cal'], f'Contrast figure A. Null-gene rate of the combined nominal p on the beta = 0 anchor divided by its threshold '
                f'(1 = nominal; log scale), every arm at 0.05 / 0.01 / 0.001, {THIS_SET} left and {REF_SET} right; bars are '
                f'gene-clustered 95% intervals under the one record permutation both anchors share; a lower whisker reaching '
                f'the axis floor is a lower bound of 0.')}
<p><b>Precision against unit weights.</b> The <i>squared error</i> of an arm is the squared difference between its slope
estimate and the truth, summed over gene units; the ratio reported is the arm's squared error over unit weights' on
the same units, paired, with the gene-clustered interval; below 1 is more precise than unit weights. At the causal variant the hapmixQTL rows
and tensorQTL's (its total truth) use their pipeline-scale truth and every other row the count-scale truth for the arm and unit alike; on the anchor's
null genes the truth is 0 for every arm. split's combined ratio is
{ci(sp(S, 'beta0.4', 'nonnull', 'ratio_vs_unit'), 'value', 2)} at the causal variant (|beta| 0.4) and
{ci(sp(S, 'beta0.0', 'null', 'ratio_vs_unit'), 'value', 2)} on the anchor in the {THIS_SET}, against
{ci(sp(R, 'beta0.4', 'nonnull', 'ratio_vs_unit'), 'value', 2)} and {ci(sp(R, 'beta0.0', 'null', 'ratio_vs_unit'), 'value', 2)}
in the {REF_SET}. The other effect sizes are in section 3.4.</p>
{t_prec}
{img(fc['prec'], f"Contrast figure B. The precision table as points: squared error under the arm over squared error under unit "
                 f"weights (log scale; below 1 is more precise than unit weights), {THIS_SET} circles and {REF_SET} squares, at "
                 f"the causal variant (|beta| 0.4; hapmixQTL and tensorQTL rows on their pipeline-scale truth, every other row on the "
                 f"count-scale truth for arm and unit alike) and on the anchor's null genes (truth 0); bars are gene-clustered "
                 f"95% intervals. Across methods the anchor ratio still carries each method's slope scale (text below the "
                 f"TReCASE paragraphs). mixQTL with published cutoffs is left out of the figure: its anchor ratio "
                 f"is {P(S, 'beta0.0', 'mixqtl', 'combined', 'null', 'ratio_vs_unit')['value']:.0f}x in the {THIS_SET} and "
                 f"{P(R, 'beta0.0', 'mixqtl', 'combined', 'null', 'ratio_vs_unit')['value']:.0f}x in the {REF_SET} (table above), "
                 f"which would compress every other arm onto one line.")}
<p><b>Ranking and gene-level power.</b> Each arm gives every tested variant a nominal p; a gene's <i>lead variant</i> is
the tested variant in its cis window with the smallest nominal p, and that p ranks the genes within a dataset. The
<b>AUC</b> (area under the receiver operating characteristic curve of that ranking) is the probability that a randomly
chosen non-null gene, one with an injected effect, ranks above a randomly chosen null gene: 0.5 is chance, 1 is
perfect separation. It is computed per dataset and averaged, with an interval that is {auc_iv}. <b>Power at 5%
realized false-discovery proportion</b> walks down the pooled ranking of gene units over the datasets and reports the
share of non-null units above the deepest rank at which at most 5% of the units called are null; it uses the truth,
so it has no interval. The lead nominal p is not corrected for the number of variants tested, so both are ranking
instruments, not calls. The <b>gene-level p</b> is the call a scan makes, and each arm has up to two. The first is its
own permutation p for the whole cis window: pval_beta, the Beta-approximated permutation p of the lead, for the
hapmixQTL arms (map_cis) and total-only tensorQTL (tensorQTL's map_cis); for mixQTL the empirical p of its own
permutation scan, which has no Beta approximation; TReCASE has none here. The second, for every arm, is
the <b>eigenMT</b> p (Davis et al. 2016), which needs no permutation: a gene's effective number of independent tests,
M<sub>eff</sub>, is the number of eigenvalues of its tested variants' genotype correlation matrix (Ledoit-Wolf shrunk:
the sample correlation pulled toward the identity by a weight estimated from the data; in windows of 200 consecutive
variants) needed to explain 99% of their variance, and the gene-level p is min(1,
M<sub>eff</sub> x the gene's smallest nominal p), a Bonferroni correction over M<sub>eff</sub> tests that inherits any
miscalibration of the arm's nominal p. With 92 donors in 200-variant windows the shrinkage, not linkage disequilibrium,
sets M<sub>eff</sub>, so for an arm whose nominal p is not anticonservative the eigenMT p is conservative relative to
the permutation p (section 3 measures by how much on this set). <b>Benjamini-Hochberg at 5%</b> keeps the k smallest gene-level p values of a
dataset for the largest k with p<sub>(k)</sub> &le; 0.05 k / (genes tested); <b>gene-level power</b> is the share of
non-null gene units so discovered, with the gene-clustered interval. In the first table each first line is the AUC and
each second line the power at 5% realized false-discovery proportion; the second table is the gene-level power, a
permutation-p column and an eigenMT column per effect size.{fdp_sentence}</p>
{t_rank}
{t_gene}
{img(fc['rank'], f'Contrast figure C. The ranking and power tables as points, {THIS_SET} left and {REF_SET} right, arm colours '
                 f'and markers as in Figure 1. Top: AUC of the within-dataset gene ranking by lead nominal p (bar: the range of '
                 f'the {n_rep[THIS_SET]} per-dataset values, not a 95% interval; dotted line: chance). Second row: power at 5% '
                 f'realized false-discovery proportion over the pooled datasets (no interval). Third row: gene-level power, the share '
                 f"of non-null gene units discovered by Benjamini-Hochberg at 5% on each arm's "
                 f'permutation p (every arm but TReCASE; gene-clustered 95% intervals). Fourth row: the eigenMT p '
                 f'of every arm held to a common error rate, power at 5% realized false-discovery proportion as in the second '
                 f"row (no interval). Bottom: the realized false-discovery proportion of each arm's Benjamini-Hochberg calls at "
                 f'5% on the eigenMT p (dashed line: 0.05); an arm above the line calls null genes beyond the 5% the procedure '
                 f"promises, so its Benjamini-Hochberg power on this p (table) is not comparable with the others'. The "
                 f'statistics are defined in the paragraph above the tables.')}
{joint}
{limit}
{settled}'''


def interp_joint(part):
    """TReCASE's paragraph of one results section."""
    tz = prec('beta0.0', 'trecase', 'combined', 'null', 'sd_z')
    n0 = lambda al, g='all': S['null']['beta0.0']['trecase']['combined'][g][al]   # noqa: E731
    comp = {k: v['all']['0.05'] for k, v in S['trecase_components'].items()}
    name = dict(trec='total-count (TReC)', joint='joint', ase='allele-specific (ASE)')
    ex = [prec(f'beta{b}', 'trecase', 'combined', 'nonnull', 'ratio_vs_unit_count') for b in BETAS]
    tb = [bias(b, 'trecase', 'combined', 'bias_count') for b in BETAS]
    if part == 'ranking':
        need(all(auc(b, 'trecase')['mean'] < auc(b, 'split')['mean'] and fdp(b, 'trecase')['all']['power'] < fdp(b, 'split')['all']['power']
                 for b in BETAS), 'TReCASE below split on AUC and power at 5% FDP at every |beta|')
        return f"""
<p><b>TReCASE.</b> Its AUC is {A_('trecase')} and its power at 5% realized FDP {P_('trecase')}, against {A_('split')} and
{P_('split')} for split: below split in point estimate on both at every |beta|; the per-dataset AUC ranges overlap at
{at_betas(overlap('trecase', 'split'))}. At |beta| 0.2 its pooled cut fell at a lead p of {thr('0.2', 'trecase')},
against {thr('0.2', 'split')} for split: null genes' lead p reached below split's cut, which agrees with its null-gene
rates (section 3.7).</p>"""
    if part == 'bias':
        need(all(ivl(d) == 'includes' for d in tb), 'TReCASE\'s bias interval including 1 at every |beta|')
        return f"""
<p><b>TReCASE.</b> Its one slope has estimand beta (section 2), so its ratio to beta measures estimator bias: it recovers
{B_('trecase', 'combined', n=3)} of beta, intervals {', '.join(ci(d, 'mean', 3) for d in tb)}, each including 1. Where
asSeq refits the dosage as a linear covariate (section 3, run facts) the slope is about half of beta, so those
unflagged rows pull the ratio down.</p>"""
    if part == 'precision':
        need(ivl(tz) == 'above' and ivl(ex[0]) == 'above' and ivl(ex[2]) == 'above',
             'TReCASE\'s anchor sd(z) above 1 and its squared error above unit weights\' at |beta| 0.2 and 0.8')
        return f"""
<p><b>TReCASE.</b> Its standard error is derived by the Wald inversion of &chi;<sup>2</sup> (section 2), so on null genes
sd(z)<sup>2</sup> is about the mean of &chi;<sup>2</sup>, 1 for a &chi;<sup>2</sup> with one degree of freedom: on the
anchor it is {ci(tz, 'value', 3)}, above 1, which agrees with its null-gene rate (section 3.7). At the causal variant its
sd(z) is {Z_('trecase', 'combined')}, a measure that absorbs bias. Its squared error, against unit weights on the
count-scale truth, is {' / '.join(ci(d, 'value', 2) for d in ex)} at |beta| = 0.2 / 0.4 / 0.8 and
{En_('trecase', 'combined')} on the anchor's null genes, against {Ex_('split')} and {En_('split', 'combined')} for
split.</p>"""
    if part == 'lead':
        return f"""
<p>TReCASE's share of leads within r<sup>2</sup> &ge; 0.8 of the causal variant is {R_('trecase')}, against {R_('split')}
for split, with no interval.</p>"""
    if part == 'detection':
        return f"""
<p>TReCASE detects the causal variant at p &lt; 1e-3 in {D_('trecase')} of non-null gene units, against {D_('split')} for
split; a causal unit without a row is left out of its share (section 2). Its test rejects at 1e-3 in
{ci(n0('0.001'), 'rate', 4)} of the anchor's null-gene tests (section 3.7), so its detections are not comparable at
face value.</p>"""
    if part == 'null':
        need(n0('0.05')['lo'] > 0.05, 'TReCASE\'s anchor rate at 0.05 above 0.05')
        fin = n0('0.05')['rate']
        return f"""
<p><b>TReCASE.</b> Its likelihood-ratio p, referred to &chi;<sup>2</sup> with one degree of freedom, rejects at 0.05 on
null genes in {N_('trecase', 'combined', 4)} of tests (anchor, then |beta| = 0.2 / 0.4 / 0.8); on the anchor the
gene-clustered intervals are {ci(n0('0.05'), 'rate', 4)} at 0.05 and {ci(n0('0.001'), 'rate', 5)} at 0.001. No stored
null run exists for it. Scored alone on the anchor's null genes, each of asSeq's component tests rejects at 0.05 in
{'; '.join(f'{name[k]} {ci(v, "rate", 4)}' for k, v in comp.items())}, against {f(fin, 4)} for the final p, so each
component is above nominal on its own and the per-test choice between them adds to that.</p>"""
    raise SystemExit(f'interp_joint: unknown part {part}')


def sec_results(figs):
    interp = lambda fn: fn() if INTERPRETED else ''   # noqa: E731
    joint = lambda part: interp_joint(part) if INTERPRETED else ''   # noqa: E731
    if INTERPRETED:
        off = [g for g, v in SN['below_floor_genes'].items() if v['n_a'] < 2]
        n_all, n_al = (S['null']['beta0.0']['gibbs'][ch]['all']['0.05']['tests'] for ch in ('combined', 'allelic'))
        if len(off) != 1:
            raise SystemExit(f'{NULL}: allelic channel switched off in {off}; reword section 3.4')
        exc_txt = f'''The one gene whose allelic channel is switched off altogether, {off[0]} (one informative allelic
donor), has no allelic p, so its {n_all - n_al:,} tests on the anchor leave the allelic null rates ({n_al:,} of
{n_all:,} tests); the stored null treats it the same way.'''
        fig1_bar = 'range of the three per-dataset AUCs, not a 95% interval'
        anchor_tab = f'''<p>The anchor against the stored null on this pipeline (hapmixQTL arms only; like for like in every
channel, section 2):</p>
{tab_anchor()}'''
    else:
        exc_txt = ''
        fig1_bar = 'interval of the mean over datasets resampled with replacement'
        anchor_tab = ''
    units_a = S['precision']['beta0.4']['gibbs']['allelic']['nonnull']['sd_z']['all']['units']
    units_c = S['recovery']['beta0.4']['gibbs']['allelic']['bias_count']['all']['units']
    ladder = sec_ladder() if LD is not None else ''   # the ladder was run for the interpreted set only
    return f'''
<h2>3. Results</h2>
{sec_run_facts()}

<h3>3.1 Gene ranking: AUC and power at a realized false-discovery proportion</h3>
{interp(interp_ranking)}
{joint('ranking')}
{tab_ranking()}
<p>By read band ({BAND_HTML} reads), AUC and then power at 5% FDP:</p>
{tab_bands(lambda b, a: S['ranking'][f'beta{b}'][a]['auc'], 'mean')}{tab_bands(fdp, 'power')}
{img(figs['ranking'], 'Figure 1. A: AUC of the within-dataset gene ranking by lead nominal p (bar: '
     + fig1_bar + '). B: share of non-null gene units called at the deepest point of the pooled '
     'ranking where at most 5% of calls are null genes. C: share of non-null gene units discovered by '
     "Benjamini-Hochberg at 5% on each arm's permutation p (map_cis pval_beta for the hapmixQTL arms"
     + (', split on native counts' if NATIVE else '')
     + " and tensorQTL, mixQTL's own pval_perm; "
     + ('the two TReCASE arms have' if NATIVE else 'TReCASE has')
     + " none; gene-clustered intervals). D: the eigenMT p of every "
     'arm held to a common error rate, the share of non-null gene units called at the deepest point of the pooled '
     'ranking by that p where at most 5% of calls are null genes (as B; no interval). E: the realized '
     "false-discovery proportion of each arm's Benjamini-Hochberg calls at 5% on the eigenMT p, null gene units "
     'among the units called (dashed line: 0.05); an arm above the line calls null genes beyond the 5% the procedure '
     'promises, which is why its Benjamini-Hochberg power on this p (table below) is not comparable with the '
     "others'. Points are offset sideways within each |beta| so that intervals do not overlap.")}

<h3>3.2 Gene-level power</h3>
{interp(interp_gene_level)}
{tab_gene_level()}

<h3>3.3 Bias at the causal variant</h3>
{interp(interp_bias)}
{joint('bias')}
{tab_bias()}
{img(figs['bias'], 'Figure 2. Bias ratio (mean slope / truth at the causal variant, gene-clustered interval) by '
     'arm and read band, every arm against the count-scale truth. Top two rows: the allelic and total channels of the '
     "arms that have them (hapmixQTL, mixQTL; tensorQTL's one slope is drawn in the total row). Bottom row: every "
     "arm's one combined slope, the row where TReCASE appears, fitting one joint effect for both kinds "
     "of count, beside hapmixQTL's combined slope (section 3.3). " 'The hapmixQTL allelic points keep as zeros the units with no allelic data, which '
     'the mixQTL points drop (section 3.3). mixQTL channels: allelic = asc, total = trc. Colour shade = |beta|. The '
     'y axes differ between panels.')}

<h3>3.4 Precision: stated standard error and efficiency against unit weights</h3>
<p>In the allelic channel the causal-variant statistics use the {units_a} of {units_c} causal units per |beta| that have
allelic data (section 3.3). {exc_txt}</p>
{interp(interp_precision)}
{joint('precision')}
<p>sd(z), realized over stated standard error:</p>
{tab_precision('sd_z')}
<p>Mean squared error ratio against unit weights. At the causal variant the hapmixQTL rows hold the arm and unit
weights to their shared pipeline-scale truth (a comparison among weightings of one method), and total-only tensorQTL's
row holds its slope to the pipeline-scale total truth against unit weights' combined slope on theirs; every other row
holds both to the count-scale truth. The null-gene column has truth 0 for every arm.</p>
{tab_precision('ratio_vs_unit')}
<p>The cross-method comparison: combined channel, every arm and unit weights both against the count-scale truth at
the causal variant (for the hapmixQTL and mixQTL arms and for unit weights, the inverse-variance combination of beta
and the per-gene total truth; for tensorQTL, the per-gene total truth; for TReCASE, beta).</p>
{tab_cross()}
{cross_note()}
{img(figs['efficiency'], 'Figure 3. Mean squared error ratio against unit weights (log scale; below 1 = more '
     'precise than unit weights), gene-clustered interval. Top: causal variant of non-null genes. Bottom: every '
     'tested variant of null genes ("anchor" is the beta = 0 dataset). Allelic and total panels: hapmixQTL arms '
     '(top, pipeline-scale truth for arm and unit alike) and mixQTL (bottom only). Combined panels: every arm, '
     'TReCASE included, and at the causal variant the count-scale truth for arm and unit alike, so the '
     'top combined panel is the cross-method comparison and its hapmixQTL points differ from the pipeline-scale '
     'values in the text. unit is 1 by definition and not drawn; the total channel of split is 1 by construction. '
     'mixQTL with published cutoffs is left out of the figure (its ratios are in the tables above; on the anchor '
     f"{prec('beta0.0', 'mixqtl', 'combined', 'null', 'ratio_vs_unit')['value']:.0f}x unit weights', which would "
     'compress every other arm onto one line). The y range of each panel covers every plotted interval.')}

<h3>3.5 Lead-variant recovery</h3>
<p>Each share is over {S['lead']['beta0.4']['gibbs']['all']['units']} gene units per |beta|.</p>
{interp(interp_lead)}
{joint('lead')}
{tab_lead()}
<p>Share with r<sup>2</sup> &ge; 0.8 by read band:</p>
{tab_bands(lambda b, a: S['lead'][f'beta{b}'][a], 'r2_high')}
{img(figs['lead'], 'Figure 4. Share of non-null gene units whose lead variant is in LD r^2 >= 0.8 with the '
     'causal variant, by arm, |beta| and read band. No interval.')}

<h3>3.6 Causal-variant detection</h3>
{interp(interp_detection)}
{joint('detection')}
{tab_detection()}

<h3>3.7 Null genes and the beta = 0 anchor</h3>
{interp(interp_null)}
{joint('null')}
<p>Null-gene rate at 0.05 (gene-clustered interval):</p>
{tab_null()}
{anchor_tab}
{ladder}
{sec_native()}'''


def native_sec():
    """The native-input subsection's number: after the mixQTL ladder (3.8) where the gene set has one."""
    return f'3.{9 if LD is not None else 8}'


def native_method():
    """The native-input arms' design (05b_native_arms.py), part of section 2; '' where 06 scored none (no C.NATIVE)."""
    if not NATIVE:
        return ''
    return f'''
<p><b>Native-input arms.</b> RASQUAL and TReCASE are written for integer alignment counts, and every other arm reads
Salmon's point estimates. Two further arms read counts from the same STAR alignments instead ({C.NATIVE_COUNTS},
scripts/native_counts.py): the total is featureCounts' count of fragments on the gene's exons (primary, uniquely mapped,
reverse-stranded read pairs; a fragment on exons of two genes is not counted), and the allele-specific counts a and b
are phASER's counts of fragments over heterozygous SNPs on the haplotypes carrying the analysis VCF's first and second
allele (counted per transcript strand on strand-split BAMs, at SNVs in exon stretches that one gene owns on its strand,
the GTEx-style collapsed gene model; HLA genes and CHM13 short-read-inaccessible regions excluded; reads WASP-filtered for
allele-dependent mapping, scripts/phaser_wasp.py; scripts/phaser_stranded.py), set to a = b = 0 where a + b exceeds the
total. Each dataset's record permutation, label swaps and thinning factors are applied to them (05b_native_arms.py;
exact binomial thinning of integers in 02's order), so the null genes, causal variants, injected effects and truth are
the Salmon-input arms'. The count-scale total truth depends only on the causal genotypes and beta, so it is the truth of
both inputs; no pipeline-scale truth is computed for the native arms, whose standard-error and squared-error figures in
sections 3.3 and 3.4 use the count-scale truth. The effective library size is recomputed from the native totals by the
Salmon run's edgeR rule (filterByExpr, edgeR's filter keeping genes with enough counts in enough samples; protein-coding
autosomal genes; TMM, the trimmed mean of log expression ratios against a reference sample, as the scale factor).
<b>{SHORT['trecase_native']}</b> is the same TReCASE runner on the native total and on a and b as they are (no
zero-haplotype rule; asSeq's own floor of five allele-specific reads applies), with the log native effective library size
as offset and the same 17 covariates. <b>{SHORT['split_native']}</b> is split weighting on the same counts, the control
that separates the model from the quantifier: the allelic log2((a + 0.5)/(b + 0.5)) weighted by one over its counting
variance (1/(a + 0.5) + 1/(b + 0.5))/ln2<sup>2</sup>, the total log2(CPM + 1) unweighted. TReCASE's component tests are
scored on both inputs: the share of tests whose joint fit is missing (asSeq's final p is then its total-count test), the
same at each gene unit's reported lead, the power at 5% realized FDP when genes are ranked by the lead p of one component
test alone (06_score.py, trecase_parts), and each component's null-gene rate on the anchor over the tests where its p is
finite.</p>'''


def native_limits():
    """What the native arms cannot show: in section 6 on the interpreted page, at the end of the native results elsewhere."""
    if not NATIVE:
        return ''
    T = NF['trecase']
    notrun = sorted({g for v in T['per_dataset'].values() for g in v['genes_not_run_constant_total']})
    return f'''
<p><b>What the native arms cannot show.</b> The 14 RNA-tied covariates are unchanged, and their expression
principal components were computed from Salmon's log2 CPM. The two inputs admit different donors to the allelic channel
(phASER needs a fragment over a heterozygous SNP, and only the Salmon-input arms apply the zero-haplotype rule);
featureCounts drops multimapping fragments and fragments on two genes' exons{f", which leaves {', '.join(notrun)} with a total of 0 in every donor and no allele-specific counts: asSeq stops on a constant total and tensorQTL's input generator drops a constant phenotype, so neither native arm tests it and both rank it last in every dataset" if notrun else ''}.
The inputs therefore differ in which reads are counted, not only in integer against fractional values, and a difference
between a native and a Salmon-input arm combines those differences. The permutation, thinning, truth and design are
shared with the Salmon-input arms, so this compares inputs to the same models, not TReCASE's or RASQUAL's own read
pipelines, and RASQUAL was not run on native counts; {SHORT['split_native']} against {SHORT['trecase_native']} is the
comparison in which both methods see the same counts. Each |beta| rests on {S['n_datasets']['0.4']} datasets, and power at
realized FDP has no interval.</p>'''


def sec_native():
    """The native-input arms' results beside their Salmon-input counterparts: the inputs' facts, the donors each admits,
    the ranking, the calibration, the main table, then TReCASE's component statistics. Empty where 06 scored no native
    arms (no C.NATIVE)."""
    if not NATIVE:
        return ''
    F, T = NF['facts'], NF['trecase']
    gs = NF['counts']['remainder']['gene_sets'][C.GS['gene_dir'].split('/')[0]]
    anc, (lib_lo, lib_med, lib_hi) = F['datasets']['beta0.0 rep 000'], F['library']['native_over_salmon']
    cf = F['counts']   # the cohort's own counts, before any dataset's permutation or thinning (05b main)
    if cf['informative'] != anc['informative']:
        raise SystemExit(f'{C.NATIVE / "facts.json"}: the anchor has {anc["informative"]:,} native informative pairs against '
                         f'{cf["informative"]:,} before thinning; section 3 (native-input arms) reads the unthinned counts as the anchor\'s')
    notrun = sorted({g for v in T['per_dataset'].values() for g in v['genes_not_run_constant_total']})
    if notrun != sorted({g for v in F['split_native'].values() for g in v['not_tested']}):
        raise SystemExit(f'{C.NATIVE}: the two native arms left different genes untested; reword section 3 (native-input arms)')
    arms = ('split', 'split_native', 'trecase', 'trecase_native', TQ)
    tp = lambda b, a: S['trecase_parts'][f'beta{b}'][a]   # noqa: E731
    lf = lambda a, ws: (sum(tp(b, a)['lead_joint_failed'][w]['failed'] for b in BETAS for w in ws)   # noqa: E731  pooled over |beta| > 0
                        / sum(tp(b, a)['lead_joint_failed'][w]['units'] for b in BETAS for w in ws))
    lead_fail = lambda a, w: lf(a, (w,))   # noqa: E731
    both = lambda a: lf(a, ('null', 'nonnull'))   # noqa: E731
    tests = {'trecase': JF['trecase'], 'trecase_native': T['pooled']}
    n0 = lambda a, al: ci(S['null']['beta0.0'][a]['combined']['all'][al], 'rate', 4)   # noqa: E731
    nc = lambda a, ch: ci(S['null']['beta0.0'][a][ch]['all']['0.05'], 'rate', 4)   # noqa: E731
    comp = {'trecase': S['trecase_components'], 'trecase_native': S['trecase_native_components']}
    name = dict(trec='total-count (TReC)', joint='joint', ase='allele-specific (ASE)')
    pw = lambda b, a: fdp(b, a)['all']['power']   # noqa: E731
    moved = ', '.join(f'{w} at {at_betas(bs)}' for w, bs in (
        ('higher', [b for b in BETAS if pw(b, 'trecase_native') > pw(b, 'trecase')]),
        ('lower', [b for b in BETAS if pw(b, 'trecase_native') < pw(b, 'trecase')]),
        ('equal', [b for b in BETAS if pw(b, 'trecase_native') == pw(b, 'trecase')])) if bs)
    t1 = table(['arm'] + [f'AUC, |beta| {b}' for b in BETAS] + [f'power at 5% FDP, |beta| {b}' for b in BETAS]
               + ['anchor null-gene rate, 0.05', 'anchor null-gene rate, 0.001']
               + [f'recovered share of the count-scale truth, |beta| {b}' for b in BETAS],
               [[LABEL[a]] + [f(auc(b, a)['mean']) for b in BETAS] + [f(fdp(b, a)['all']['power']) for b in BETAS]
                + [n0(a, '0.05'), n0(a, '0.001')] + [ci(bias(b, a, 'combined', 'bias_count'), 'mean', 2) for b in BETAS] for a in arms])
    t2 = table(['arm', 'tests without a joint fit', 'reported lead without a joint fit: null / non-null gene units, |beta| &gt; 0']
               + [f'power at 5% FDP by the {name[k]} p alone, |beta| 0.2 / 0.4 / 0.8' for k in SC.TRECASE_PARTS]
               + [f'anchor null-gene rate at 0.05, {name[k]} p' for k in SC.TRECASE_PARTS],
               [[LABEL[a], f'{f(tests[a]["joint_na_share"])} of {tests[a]["tests"]:,}',
                 f'{f(lead_fail(a, "null"))} / {f(lead_fail(a, "nonnull"))}']
                + [per_beta(lambda b: tp(b, a)['fdp_power'][k]) for k in SC.TRECASE_PARTS]
                + [ci(comp[a][k]['all']['0.05'], 'rate', 4) for k in SC.TRECASE_PARTS] for a in SC.TRECASE_ARMS])
    return f'''
<h3>{native_sec()} Native-input arms</h3>
<p><b>The native counts.</b> phASER's a + b exceeded the featureCounts total, and both were set to 0, in
{gs['negative']:,} of this gene set's {gs['pairs_with_reads']:,} donor-gene pairs with phASER reads, holding
{gs['allelic_fragments_in_negative']:,} of its {gs['allelic_fragments']:,} allele-specific fragments. The edgeR rule kept
{F['library']['genes_kept']:,} genes, and the native effective library size is {f(lib_lo, 2)} to {f(lib_hi, 2)} times the
Salmon one across donors (median {f(lib_med, 2)}).</p>
<p><b>Which donors each input admits to the allelic channel.</b> In the cohort's own counts, which the anchor keeps (its
thinning factors, 2<sup>-|beta|</sup>, are 1), {cf['informative']:,} of the {cf['pairs']:,} donor-gene pairs have native allele-specific counts
(a + b &gt; 0; {anc['one_sided_zero']:,} of them with one haplotype at 0) and {cf['salmon_informative']:,} have Salmon
haplotype estimates (pL + pR &gt; 0). Of those Salmon pairs {anc['salmon_admitted']:,} pass hapmixQTL's zero-haplotype
rule (common.allelic_kept: a pair with exactly one haplotype below {C.EXPRESSIBLE_MIN:g} is left out of the allelic
channel; docs/pipeline_rules.md), so the rule removes {cf['salmon_informative'] - anc['salmon_admitted']:,} of them, while
the two inputs' counts of pairs with any allele-specific signal differ by {cf['informative'] - cf['salmon_informative']:+,}
(native minus Salmon). The Salmon-input TReCASE arm carried that rule (05_run_trecase.allelic_counts gives asSeq
Y1 = Y2 = 0 on every pair it removes); the native arms do not. The median gene has {anc['admitted_median']:g} allelic
donors on native counts against {anc['salmon_admitted_median']:g} admitted Salmon donors. The allele-specific depth
differs too: over the pairs with any, the median a + b is {cf['median_hap']:g} fragments and the median pL + pR
{f(cf['salmon_median_hap'], 1)} (over every pair with pL + pR &gt; 0, before the rule). phASER's a + b counts fragments
over a heterozygous SNP; Salmon's pL + pR is its estimate for both haplotype copies of the transcripts whose copies
differ, fragments over no heterozygous site included.</p>
<p><b>What changes in the ranking.</b> At |beta| = 0.2 / 0.4 / 0.8 TReCASE's power at 5% realized FDP is
{P_('trecase')} on Salmon's inputs and {P_('trecase_native')} on native counts ({moved}), and split's
{P_('split')} and {P_('split_native')}, against total-only tensorQTL's {P_(TQ)} (AUC in the first table below). The
change in TReCASE's power between its two inputs mixes three differences that this benchmark does not separate: integer
alignment counts against Salmon's fractional estimates, the zero-haplotype rule that the Salmon-input arm carried and
the native arm does not, and the allele-specific depth above. A Salmon-input TReCASE arm without the rule, not run,
would separate the rule from the input format.</p>
<p><b>Calibration, a finding separate from the ranking.</b> On the anchor TReCASE's null-gene rate at 0.05 is
{n0('trecase', '0.05')} on Salmon's inputs and {n0('trecase_native', '0.05')} on native counts, and at 0.001
{n0('trecase', '0.001')} and {n0('trecase_native', '0.001')}. split's anchor rate at 0.05 is {n0('split', '0.05')} on
Salmon's inputs and {n0('split_native', '0.05')} on native counts, and at 0.001 {n0('split', '0.001')} and
{n0('split_native', '0.001')}; in split on native counts the allelic channel rejects at 0.05 in
{nc('split_native', 'allelic')} (split {nc('split', 'allelic')}) and the total channel in {nc('split_native', 'total')}
(split {nc('split', 'total')}). These rates describe each arm's nominal p; they do not explain its ranking. Power at 5%
realized FDP ranks genes by their lead p and counts the calls with the true null labels, so a p made smaller for every
gene by the same increasing map leaves it unchanged, and null genes whose p is too small can only move above non-null
genes and lower it: an anticonservative null cannot raise that power.</p>
{t1}
<p><b>TReCASE's component tests</b> (second table). TReCASE's joint fit is missing at the reported lead in
{f(both('trecase'), 2)} of the gene units at |beta| &gt; 0 on Salmon's inputs and {f(both('trecase_native'), 2)} on
native counts, and in {f(tests['trecase']['joint_na_share'], 2)} and {f(tests['trecase_native']['joint_na_share'], 2)}
of all tests. Its total-count test alone reaches {per_beta(lambda b: tp(b, 'trecase')['fdp_power']['trec'])} power at
5% realized FDP on Salmon's inputs and {per_beta(lambda b: tp(b, 'trecase_native')['fdp_power']['trec'])} on native
counts, against tensorQTL's {P_(TQ)} on Salmon's totals.</p>
{t2}'''


def ladder_method():
    """The mixQTL ladder's design (07_mixqtl_ladder.py), part of section 2; its results are section 3.8."""
    return f'''
<p><b>The mixQTL ladder.</b> A separate run (benchmark/simulated_effects/07_mixqtl_ladder.py, output
{C.LADDER / "ladder.json"}) went from the unit-weight arm to mixQTL one change at a time, in the total channel, on the
causal units where every step has a finite error (the <i>common set</i>). mixQTL fits its total channel in two steps.
First the <i>covariate offset</i>: the natural-log total, log(total reads / 2 / library size), is regressed on the 17
covariates without the genotype; the covariates whose t statistic exceeds 2 in absolute value are kept (the
<i>selected covariates</i>) and the regression is refitted on them. Second, the offset is subtracted from the response and
the result is regressed on x = (h1 + h2)/2, half the ALT dosage, without first removing from x the part that the
selected covariates explain (mixQTL's own R code: the offset in rlib_covariate.R:27-40, the genotype regression on
the residual in rlib_matrix_ls.R:26-46; ported as covariate_offset and trc_channel). The <b>Frisch-Waugh-Lovell identity</b>
states that in a least-squares fit of y on an intercept, x and covariates C, the slope on x equals the slope of y on x
after x is replaced by its residual from a regression on the intercept and C. It follows that when both steps use the
same donors, mixQTL's slope is (1 &minus; R<sup>2</sup>) times the <i>one-step</i> slope, the slope on x when y is
regressed on the intercept, x and the selected covariates together, where R<sup>2</sup> is the share of the variance of
x over donors that the selected covariates explain. The ladder checks this where the offset's donors (total reads above
0) and the total channel's donors (total reads at the cutoff) are the same. The one-step slope with all 17 covariates
is fitted without any selection on the outcome. The chance level of R<sup>2</sup> is the same quantity at the null
genes' causal variants, where selection sees no genotype effect but the genotype principal components among the
covariates can still correlate with x. The squared error of each step at the causal variant is divided by that of the
unit-weight arm, on the common set. The <i>cutoff rung</i> is the unit-weight arm (unit weights, the half-read total, all 17
covariates in one fit) with only the donors mixQTL's count cutoffs admit. The next step changes the response to
mixQTL's natural-log total over 2 x library size and the truth to the count scale, together; then the covariates become
the selected ones; then the fit becomes mixQTL's two steps. As the ladder defines it, the unit-weight arm's squared error
in the denominator is against its pipeline-scale truth, so these ratios are not the count-scale cross-method ratios of
section 3.4. All columns share that one denominator, but the step to mixQTL's response also moves the numerator from the
pipeline-scale to the count-scale truth: it mixes a change of response with a change of the truth it is measured
against, so the ladder as built cannot assign it a cost. The other steps each hold the truth fixed and compare with
each other.</p>'''


def sec_ladder():
    T, CUT = LD['total_channel'], ('published', 'permissive')
    tc = lambda c, b, k: T[c][f'beta{b}'][k]['all']   # noqa: E731
    sel = lambda c, b: T[c][f'beta{b}']['selection']   # noqa: E731
    sq = lambda c, b, k: T[c][f'beta{b}']['sq_error_vs_unit'][k]['all']   # noqa: E731
    rung = lambda c, b: LD['ladder'][f'beta{b}'][f'unit_{c}_cutoffs']['common_set']['total']['all']   # noqa: E731
    own = lambda b: LD['ladder'][f'beta{b}']['mixqtl']['own_set']['total']['all']['value']   # noqa: E731
    share = per_beta(lambda b: 1 - sq('published', b, 'mixqtl_trc')['value'] / own(b), 2)
    ov = sum(tc(c, b, 'one_step_trc')['hi'] >= tc(c, b, 'one_step_all_trc')['lo']
             and tc(c, b, 'one_step_all_trc')['hi'] >= tc(c, b, 'one_step_trc')['lo'] for c in CUT for b in BETAS)
    lo17, hi17 = all17()
    x = lambda c, b, k0, k1: sq(c, b, k1)['value'] / (sq(c, b, k0)['value'] if k0 else 1.0)   # noqa: E731
    rq = lambda c, b: rung(c, b)['value']   # noqa: E731
    ub = per_beta(lambda b: bias(b, 'unit', 'total', 'bias_count')['mean'], 2)
    PB = lambda c, k: per_beta(lambda b: tc(c, b, k)['mean'])   # noqa: E731
    Q = lambda c, k: per_beta(lambda b: sq(c, b, k)['value'], 2)   # noqa: E731
    R2 = lambda c: per_beta(lambda b: sel(c, b)['mean_r2'])   # noqa: E731
    N2 = lambda c: ' / '.join(ci(sel(c, f'{b}')['null_gene_r2'], 'mean') for b in BETAS)   # noqa: E731
    above = lambda c: ' / '.join('above' if sel(c, b)['mean_r2'] > sel(c, b)['null_gene_r2']['hi'] else 'inside' for b in BETAS)   # noqa: E731
    nsel = sorted({int(sel(c, b)['n_selected_min_median_max'][1]) for c in CUT for b in BETAS})
    E, cu, idn = LD['estimability']['published'], LD['common_units'], LD['identity']
    t1 = table(['cutoffs', '|beta|', 'mixQTL total slope', '1 &minus; R<sup>2</sup> (prediction for the ratio of '
                'mixQTL to one-step selected)', 'one-step, selected covariates', 'one-step, all 17 covariates'],
               [[c, b] + [ci(tc(c, b, k), 'mean') for k in ('mixqtl_trc', 'predicted_1_minus_r2', 'one_step_trc', 'one_step_all_trc')]
                for c in CUT for b in BETAS])
    t2 = table(['cutoffs', '|beta|', 'cutoff rung', 'one-step, all 17, mixQTL response', 'one-step, selected', 'mixQTL two-step'],
               [[c, b, ci(rung(c, b), 'value', 2)] + [ci(sq(c, b, k), 'value', 2) for k in ('one_step_all_trc', 'one_step_trc', 'mixqtl_trc')]
                for c in CUT for b in BETAS])
    return f'''
<h3>3.8 Why mixQTL trails unit weights</h3>
<p>Sections 3.3 and 3.4 found mixQTL's total slope attenuated at every depth and its squared error above that of unit
weights; the ladder of section 2 takes the two apart. Every number in this section is read from {C.LADDER / "ladder.json"}.
The common set holds {per_beta(lambda b: cu[f"beta{b}"]["total"], 0)} of {cu["beta0.4"]["non_null"]} non-null units at
|beta| = 0.2 / 0.4 / 0.8. The selected covariates number a median {"-".join(map(str, nsel))} of 17 per gene and dataset.
Where both steps use the same donors, the largest relative difference between mixQTL's slope and
(1 &minus; R<sup>2</sup>) times the one-step slope is {idn["max_rel_dev"]:.1e} over {idn["units"]} non-null units, so the
attenuation is arithmetic, not noise.</p>
<p>Mean slope over the count-scale total truth at the causal variant (gene-clustered interval), common set:</p>
{t1}
<p>At the published cutoffs mixQTL's total slope recovers {PB("published", "mixqtl_trc")} of the truth at
|beta| = 0.2 / 0.4 / 0.8, the one-step slope with the selected covariates {PB("published", "one_step_trc")}, and the
one-step slope with all 17 covariates {PB("published", "one_step_all_trc")}; the permissive cutoffs give
{PB("permissive", "mixqtl_trc")}, {PB("permissive", "one_step_trc")} and {PB("permissive", "one_step_all_trc")}. The
attenuation has two parts: fitting the covariates before the genotype, which multiplies the one-step slope by
1 &minus; R<sup>2</sup> ({PB("published", "predicted_1_minus_r2")} published, {PB("permissive", "predicted_1_minus_r2")}
permissive), and choosing the covariates on the outcome, the gap between the one-step slopes with the selected and
with all covariates. Those two slopes' intervals are not paired, and they overlap in {ov} of the 6 rows of the table,
so these intervals resolve the selection part only where they separate; a paired interval was not computed. That the
gap comes from choosing covariates on the outcome, rather than from using fewer of them, follows from the design (the
RNA-tied covariates move with the record, independently of the genotypes) and was not tested against a random subset
of covariates of the same size. The one-step slope with all 17 covariates recovers {f(lo17, 2)} to {f(hi17, 2)} of
the truth; that remainder is not decomposed here.</p>
<p><b>The selection follows the injected effect.</b> The covariates are chosen on a response that carries the genotype
effect, so a covariate that happens to correlate with the genotype passes |t| &gt; 2 more often as the effect grows,
and R<sup>2</sup> grows with it. The mean R<sup>2</sup> of x on the selected covariates at the non-null genes' causal
variants is {R2("published")} at |beta| = 0.2 / 0.4 / 0.8 with the published cutoffs and {R2("permissive")} with the
permissive ones, against a chance level of {N2("published")} (published) and {N2("permissive")} (permissive). The
non-null mean lies {above("published")} the null interval (published) and {above("permissive")} it (permissive) at the
three effect sizes.</p>
<p><b>Squared error, one change at a time</b> (each column over the unit-weight arm's squared error, common set,
gene-clustered interval):</p>
{t2}
<p>Donor admission alone gives {per_beta(lambda b: rung("published", b)["value"], 2)} at the published cutoffs and
{per_beta(lambda b: rung("permissive", b)["value"], 2)} at the permissive ones, which admit nearly every donor. The step to
mixQTL's response, which also changes the truth, gives {Q("published", "one_step_all_trc")} published and
{Q("permissive", "one_step_all_trc")} permissive; for the reason given in section 2 it is not read as a cost. On bias the
change of response runs the other way: the one-step fit with all 17 covariates on mixQTL's response recovers
{f(lo17, 2)} to {f(hi17, 2)} of the count-scale total truth (first table), where unit weights' total slope on
the half-read total recovers {ub} (section 3.3). Selecting the covariates on the outcome gives {Q("published", "one_step_trc")} and
{Q("permissive", "one_step_trc")}, and mixQTL's two-step fit {Q("published", "mixqtl_trc")} and
{Q("permissive", "mixqtl_trc")}. At |beta| 0.8 the two-step fit is the largest single step with the permissive
cutoffs: it takes the ratio from {ci(sq("permissive", "0.8", "one_step_trc"), "value", 2)} to
{ci(sq("permissive", "0.8", "mixqtl_trc"), "value", 2)} ({f(x("permissive", "0.8", "one_step_trc", "mixqtl_trc"), 2)}-fold),
with separated intervals. With the published ones it takes the ratio from
{ci(sq("published", "0.8", "one_step_trc"), "value", 2)} to {ci(sq("published", "0.8", "mixqtl_trc"), "value", 2)}
({f(x("published", "0.8", "one_step_trc", "mixqtl_trc"), 2)}-fold), with overlapping intervals: the largest increase
of any step, while donor admission multiplies the ratio by more ({f(rq("published", "0.8"), 2)}-fold). At |beta| 0.2 it is lower than the one-step fit under both settings, with
overlapping intervals; shrinking every slope by 1 &minus; R<sup>2</sup> also shrinks its variance, which can outweigh
the bias it adds when the effect is small. That trade was not tested separately.</p>
<p><b>What the common set leaves out.</b> At the published cutoffs the rung has no total-channel estimate in
{E["rung_empty"]} of {E["gene_datasets"]} gene-datasets, all with at most {E["rung_empty_max_donors"]:.0f} admitted
donors, where a joint fit of an intercept, 17 covariates and the genotype has no residual degree of freedom; mixQTL,
whose total fit has only an intercept and x, still has a causal-variant slope in {E["rung_empty_mixqtl_trc_finite"]} of
them, from as few as {E["rung_empty_mixqtl_trc_min_donors"]:.0f} donors. Paired with unit weights over its own units
rather than the common set, mixQTL's published total ratio is {per_beta(own, 2)}, against
{Q("published", "mixqtl_trc")} on the common set. The own set contains the common set, so unit weights' squared error
over it is at least as large, and the units outside the common set, mostly few-donor fits, carry at least {share} of
mixQTL's own-set total squared error at |beta| = 0.2 / 0.4 / 0.8. So mixQTL trails unit weights in the total channel
through its donor admission at the published cutoffs and the few-donor fits that admission leaves it, and, growing with
the effect, through the attenuation of its two-step covariate adjustment. The ladder cannot cost its change of response,
which on bias favours mixQTL.</p>'''


def sec_critique():
    an = S['anchor']['gibbs']['total']['0.05']
    gl = lambda a: per_beta(lambda b: bh(b, a)['power_bh']['all']['rate'])   # noqa: E731
    lowest = {ch: [a for a in HAPMIX if S['anchor'][a][ch]['0.05']['percentile'] == 0] for ch in CHANNELS}
    return f"""
<h2>4. The strongest critique, and what it changed</h2>
<p><b>The ranking is not free of calibration.</b> A within-dataset ranking uses no threshold, but the cut at 5% realized
FDP is set by where the null genes land, and an arm whose null genes get too-small p pushes them up its ranking. gibbs's
total channel does this: its null-gene rate at 0.05 is {f(an['rate'], 4)} on the anchor and {f(an['stored'], 4)} over the
stored 200 permutations, and in section 3.1 it {'made no call' if fdp('0.2', 'gibbs')['p_threshold'] is None else f'called {f(fdp("0.2", "gibbs")["all"]["power"])} of non-null units'}
at |beta| 0.2. TReCASE, whose null rate at 0.05 is above 0.05 too, is open to the same objection. If those null p values
are too small, part of each one's ranking deficit is calibration, not a lack of signal.</p>
<p><b>What addressing it changed.</b> Gene-level Benjamini-Hochberg on each arm's own permutation p (section 3.2) refers
each arm to its own null: there gibbs reads {gl('gibbs')} against split's {gl('split')}, with overlapping intervals, so
gibbs's ranking deficit is not resolved at gene level. What survives the critique is the signal-side cost, which needs
no reference distribution: gibbs's total-channel squared error is {E_('gibbs', 'total')} of unit weights' at the causal
variant and {En_('gibbs', 'total')} on the anchor's null genes, and its combined squared error {E_('gibbs', 'combined')}.
For TReCASE no permutation p exists here, so the critique stands for it: its ranking is measured, its calibration is not
corrected.</p>
<p><b>A second objection: the allelic gain could be made by the generator.</b> The allelic Gibbs variance of a thinned
record follows its thinned counts by the generator's own rule, so 1/Va weights might track the true error on thinned
records by construction. The anchor answers this: nothing is thinned there and Va is Salmon's own, and the gibbs and
split allelic squared error on the anchor's null genes is {En_('split', 'allelic')} of unit weights', against
{E_('split', 'allelic')} at the causal variants. The total channel's loss under gibbs is on the anchor too
({En_('gibbs', 'total')}). Neither comes from the generator's rule.</p>
<p><b>A third: one anchor permutation.</b> {', '.join(f'{" and ".join(v)} {ch}' for ch, v in lowest.items() if v) or 'No arm'}
sits at the 0th percentile of the stored permutations at 0.05. The anchor is the stored stream's permutation 0, a low
draw for the total channel in every arm (section 3.7), so the anchor's rates are low by an amount the stored null
measures; the conclusions above that rest on the anchor are its squared-error ratios, which compare arms on the same
permutation, not its rates.</p>
<p><b>What is not established.</b> At the causal variant the allelic sd(z) is above 1 in point estimate for all three
hapmixQTL arms, unit weights included, but every interval includes 1, and sd(z) there absorbs bias. Because unit
weights show the same point excess, it does not bear on the choice among weightings.</p>"""


def sec_meaning():
    pr = CG['recovery']['primary']
    genes = {bn: S['precision']['beta0.0']['split']['combined']['null']['sd_z'][bn]['genes'] for bn in BANDS}
    E0 = json.loads(EARLIER.read_text())
    eb = lambda a: per_beta(lambda b: E0['recovery'][f'beta{b}'][a]['total']['bias_count']['all']['mean'])   # noqa: E731
    em = lambda a: per_beta(lambda b: E0['precision'][f'beta{b}'][a]['combined']['nonnull']['ratio_vs_unit']['all']['value'], 2)   # noqa: E731
    ep = lambda a: per_beta(lambda b: E0['ranking'][f'beta{b}'][a]['fdp_matched']['all']['power'])   # noqa: E731
    rb = lambda a: SN['rates'][a]['combined']['all']['before']['0.001']['rate']   # noqa: E731
    return f"""
<h2>5. What it means for the open decisions</h2>
<p><b>The shipped default.</b> Half-read split was adopted on 2026-09-29 (section 1). On known effects its
combined squared error is {E_('split', 'combined')} of its no-draws control's (unit weights) at the causal variant and
{En_('split', 'combined')} on the anchor's null genes, every interval below 1; its total slope recovers
{B_('split', 'total', n=3)} of the count-scale truth; its total-channel standard error matches the slope's spread; and
its combined statistic passes the stored null's rule at 0.001 ({ci(nul('split', 'combined', '0.001'), 'rate', 5)}). AUC,
ranking power and gene-level power do not separate it from unit weights or gibbs (sections 3.1 and 3.2). Its cost is in
the allelic channel: check (c) attributes about 5% attenuation to 1/Va weights ({f(pr['inv_va_pipeline']['mean'])}
against {f(pr['unit_pipeline']['mean'])} for unit weights), which the benchmark's own intervals do not resolve. Nothing
here contradicts the decision; on precision the evidence favours it.</p>
<p><b>The total channel's weights.</b> gibbs weights the half-read total by its Gibbs variance too. That costs: the total
channel's squared error is {E_('gibbs', 'total')} of unit weights', its stated standard error is too small, and its null
rejects too often ({ci(nul('gibbs', 'total', '0.05'), 'rate', 4)} at 0.05 over the stored permutations). The Gibbs draws
help in the allelic channel and hurt in the total channel, which is the split the default makes.</p>
<p><b>mixQTL as the baseline.</b> With the published cutoffs mixQTL has the lowest AUC of the hapmixQTL and mixQTL arms
({A_('mixqtl')}), a total slope at {B_('mixqtl', 'total', n=3)} of the truth and a combined squared error of
{E_('mixqtl', 'combined')} of unit weights'; with the permissive cutoffs {A_('mixqtl_permissive')},
{B_('mixqtl_permissive', 'total', n=3)} and {E_('mixqtl_permissive', 'combined')}. Split is ahead of both settings in
point estimate at every |beta| on AUC, ranking power and combined squared error. Most of mixQTL's total attenuation is its
two-step covariate adjustment and its choice of covariates on the outcome (section 3.8); its total slopes should not be
used as a reference for effect size.</p>
<p><b>TReCASE and a total-only scan as comparators.</b> TReCASE ranks below split in point estimate on AUC and power at 5%
realized FDP at every |beta| and rejects too often on null genes ({ci(S['null']['beta0.0']['trecase']['combined']['all']['0.05'], 'rate', 4)}
at 0.05 on the anchor); its slope recovers beta within its intervals ({B_('trecase', 'combined', n=3)}) but has
{Ex_('trecase')} of unit weights' squared error. Total-only tensorQTL is within its intervals of unit weights' combined
squared error at the causal variant ({E_(TQ, 'combined')}) but not on the anchor's null genes ({En_(TQ, 'combined')}).
TReCASE was run on Salmon point estimates rather than on reads (section 6), so this measures it as run here.</p>
<p><b>Where it narrows earlier results.</b> The run of earlier on 2026-10-01 on the same datasets' counts, the same
expression PCs, code and Meier's correction, but with the log2(CPM + 1) total ({EARLIER.parent.name}), is like for like
except the total's transform; its arms differed too (it also had plus_one, and its gibbs weighted the log2(CPM + 1)
total by that total's Gibbs variance). There unit weights'
total slope recovered {eb('unit')} of the count-scale truth, here {B_('unit', 'total', n=3)}: the half-read total removes
the attenuation that +1 CPM put on low-depth folds. split's combined squared error against unit weights was {em('split')},
here {E_('split', 'combined')}, and its power at 5% realized FDP {ep('split')}, here {P_('split')}. The stored null's
'before' side, the 8a06803 re-run on log2(CPM + 1), earlier expression PCs and no Meier's correction, read
{f(rb('split'), 5)} and {f(rb('unit'), 5)} for split and unit at 0.001 against {f(nul('split', 'combined', '0.001')['rate'], 5)}
and {f(nul('unit', 'combined', '0.001')['rate'], 5)} here; that difference is those three changes together. The 2026-09-19
finding that the Gibbs draws improve the allelic point estimate
(brainvar_hapmix_deploy/mixqtl_replication_20260919/REPORT.md: a median permutation variance under 1/v weights of 0.340 of
the unweighted one on 29 high-coverage genes) points the same way as the allelic ratio at 1,000 or more reads here,
{f(prec('beta0.4', 'split', 'allelic', 'nonnull', 'ratio_vs_unit', '>=1000')['value'], 2)} at |beta| 0.4; the statistics
differ, so only the direction is compared (section 6). The gene set is the 100 genes of the corrected null store, of
which {genes['100-999'] + genes['>=1000']} have 100 or more median haplotype-informative reads; section 6 lists what these
data cannot settle.</p>"""


def native_meaning():
    """The meaning section's sentence on the native-input arms; '' without them."""
    if not NATIVE:
        return ''
    return f''' TReCASE was also run on alignment counts from the same BAMs (section {native_sec()}):
there its power at 5% realized FDP is {P_('trecase_native')}, against {P_('split_native')} for split on the same counts,
and its anchor null-gene rate at 0.05 {f(S['null']['beta0.0']['trecase_native']['combined']['all']['0.05']['rate'], 4)};
its change from the Salmon-input arm mixes the input format, the zero-haplotype rule that arm carried and the
allele-specific depth, and an anticonservative null cannot raise power at realized FDP. RASQUAL was not run on native
counts.'''


def sec_limits():
    genes = {bn: S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z'][bn]['genes'] for bn in BANDS}
    tr = JF['trecase']
    pct = lambda x: f'{100 * x:.1f}%'   # noqa: E731
    if tr['df_not_1'] + tr['trec_na'] != tr['final_na']:
        raise SystemExit(f'TReCASE: final p missing in {tr["final_na"]} tests, not the {tr["df_not_1"]} with df != 1 plus '
                         f'{tr["trec_na"]} failed TReC fits; reword section 6')
    return f'''
<h2>6. Limits: what this analysis cannot establish</h2>
<p><b>No oracle variance.</b> Real data carry no true error variance per record, so the efficiency results are
relative to unit weights. They say which weighting is more precise than another on these data, not how far any
of them is from the best possible weights.</p>
<p><b>The null is the record permutation.</b> Both the generator and the map_cis null move donor records against
genotypes with the genotype principal components tied to the genotypes. The data therefore cannot say whether
that permutation rule, or the alternative in which the principal components move with the record, matches the
sampling distribution of an observed statistic; the open decision on that rule is untouched.</p>
<p><b>What the injected effects are.</b> Effects are made by thinning only, so expression can only go down, and
there is one causal variant per gene. The gene set is the 100 genes of the corrected null store, not a sample of
the transcriptome: {genes["100-999"] + genes[">=1000"]} of them have 100 or more median haplotype-informative reads
({genes["<100"]} below 100, {genes["100-999"]} at 100-999, {genes[">=1000"]} at 1,000 or more). There are 3 datasets
per effect size with 50 non-null genes each, so an effect size rests on
{S['lead']['beta0.4']['gibbs']['all']['units']} non-null gene units spread over the 100 genes (a gene is non-null in
about 1.5 of the 3 datasets), and a read band on far fewer; one gene can move a band's value.</p>
<p><b>What thinning cannot reproduce.</b> The point estimate is Salmon's variational-Bayes optimum, the read count
Salmon's optimizer assigns to each transcript, not an observed count. At lower depth Salmon puts one haplotype at
exactly zero more often, and binomial thinning cannot create those zeros, so the zero-haplotype admission rule is
exercised less here than it would be on truly shallower libraries. Thinning also adds binomial noise of the model's
own kind, which is expected to dilute the real data's coupling between weights and residuals on thinned records and
to pull the null-gene rates at |beta| &gt; 0 toward nominal; section 3.7 compares them with the stored null. Either way those rates are not calibration results. The precision of the 1/v arms on thinned records may
be more favourable than on real records at the same depth; this was not measured. The pipeline's transforms (the
0.5-read offsets of the allelic ratio and of the half-read total) attenuate a fold at low depth, so bias against beta mixes that
attenuation with estimator bias. The pipeline-scale truth separates the two only for an unweighted fit.</p>
<p><b>The allelic variance rule.</b> It overstates Va' by at most (1 - f) x 1.4% at 100-999 total reads and
(1 - f) x 9% at 10-99, because Salmon's Gibbs prior of one pseudo-read per transcript copy does not scale with depth
(README, Generator). No known-answer test of the rule against Salmon run at reduced depth exists; the
premise check is at native depth, on one donor, and its pass thresholds were set after its first result.</p>
<p><b>mixQTL is compared as run.</b> Its gene-level p comes from its own published permutation null (no
haplotype-label swap, no Beta approximation), not hapmixQTL's, and its efficiency against unit weights compares methods
as run, on a different admitted donor set.</p>
<p><b>TReCASE is run away from its design.</b> It sees Salmon's haplotype point estimates, rounded to integers, as
allele-specific counts and Salmon's fractional totals, not counts of reads at heterozygous SNPs; this benchmark says
nothing about it on its own read pipeline (the native-input arms of the delivered pages were the step toward that).</p>
<p>asSeq's joint TReCASE fit is missing in {pct(tr["joint_na_share"])} of the run's {tr["tests"]:,} tests and in
{pct(tr["causal_joint_na_share"])} of its {tr["causal_nonnull"]} causal-variant tests. asSeq's trace logs attribute
{tr["joint_fail_theta"]:,} of the missing fits ({pct(tr["joint_fail_theta"] / tr["tests"])} of all tests) to the
overdispersion step's search ending abnormally in its line search; that search is L-BFGS-B, an iterative optimizer
that approximates the curvature of the likelihood from its gradients, within bounds. The largest absolute gradient at
such a stop was {tr["theta_gradient_max"]:.1e} over the run; whether the abnormal stops are at the optimum was not
checked. Treating them as converged would require patching asSeq and was not done. Where the joint fit is
missing, asSeq's final p is its total-count test; at the causal variants the final p was the total-count test in
{tr["causal_final_trec"]} of {tr["causal_nonnull"]}, the joint test in {tr["causal_final_joint"]}. The trace logs also
count {tr["ase_fail"]:,} failed allele-specific fits and the {tr["linear_dosage"]:,} linear-dosage refits of section 3;
the output rows count {tr["few_het"]:,} tests ({100 * tr["few_het"] / tr["tests"]:.1f}%) with fewer than five
heterozygous donors, for which asSeq fits no allele-specific model (the classes can overlap). In {tr["df_not_1"]:,}
tests ({tr["df_not_1_anchor"]:,} on the anchor) the joint statistic has 0 degrees of freedom (the overdispersion at its
boundary under the alternative), so asSeq reports no p for them. They are absent from the null rates, the ranking and
detection, but their slope and derived standard error (&chi;<sup>2</sup> &gt; 0) enter bias and the precision
statistics, where the derived standard error is not a standard error.</p>
<p>TReCASE receives exactly the allele-specific records the hapmixQTL arms admit (the zero-haplotype rule of
section 2 removes {tr["zeroed"][0]:,} to {tr["zeroed"][1]:,} haplotype-informative donor-gene pairs per dataset, section 3);
asSeq's own floors then drop {tr["asseq_dropped"][0]} to {tr["asseq_dropped"][1]} records per dataset (fewer than five
allele-specific reads) and the allele-specific model at the {tr["few_het"]:,} tests above. Its likelihood is written for
read counts, and these are Salmon's fractional point estimates, rounded where the model needs integers. And the injected
total fold has, averaged over donors, the dosage form TReCASE assumes (section 2), while the linear total channels of
hapmixQTL and mixQTL approximate it by a straight line; the generator therefore suits TReCASE's total model.</p>
<p><b>Thresholds.</b> Nothing here tests nominal p below 1e-5, measures a null rate below 1e-3, or tests gene-level
thresholds at transcriptome scale. The combined p's Welch-Satterthwaite reference treats estimated channel weights as
fixed; Meier's correction (section 2) is applied to every hapmixQTL combined p on this page, and under the exact model
it leaves 1.004-1.017x nominal at 0.05 and 1.014-1.123x at 0.001 over 15 allelic donors to all
(docs/hapmixqtl_methods.md, Section 4.5), a residual this benchmark does not measure. The eigenMT gene-level p is a
Bonferroni bound over M<sub>eff</sub> tests on the arm's nominal p and inherits its miscalibration; only the permutation
p is referred to the arm's own null.
Of the generator checks, only check (c)'s pass rule was fixed before its first run;
the other thresholds of checks (a) to (c) were not pre-registered (01_check_inputs.py, its parameters' comments).</p>
{native_limits()}
{earlier_runs_limits()}'''


def earlier_runs_limits():
    """The comparisons with other runs that are not like for like, what separates each, and what would have to run to make
    it like for like; '' when there are none."""
    rows = []
    if INTERPRETED:
        rows.append(['RASQUAL, left out of the scored arms (sections 2 and 3)', C.COMMITTED.name,
                     'its staged results were made on the earlier datasets\' log2(CPM + 1) total, expression PCs and fingerprints',
                     '04_run_rasqual.py into this run\'s directory, then 06_score.py and 08_report.py, with RASQUAL back in '
                     'common.JOINT; on the committed run it took about 126 CPU-hours (93 ms per tested variant over 4.87 '
                     'million tests, on 64 jobs); the core budget is a user decision'])
        rows.append(['section 5\'s run of earlier on 2026-10-01', EARLIER.parent.name, 'the total\'s transform '
                     '(log2(CPM + 1) there) and the arms (plus_one there, gibbs\'s Vt on log2(CPM + 1)); counts, expression '
                     'PCs, code and Meier\'s correction the same', 'nothing: it is the comparison of one change'])
        rows.append(['section 5\'s earlier records', 'mixqtl_replication_20260919',
                     'other genes (29 high-coverage genes), the pre-correction pipeline and another statistic (a ratio of '
                     'median permutation variances)', 'that statistic recomputed on these datasets under this pipeline, on its '
                     '29 genes; not planned'])
    elif REF_COV != ARMS_COV:
        rows.append([f'this set against the {REF_SET} (the contrast section)', REF_RUN.parent.name,
                     f'the expression principal components ({REF_COV.name} against {ARMS_COV.name}), in the total and combined channels',
                     f'the {REF_SET} rerun on this code into common.GENE_SETS\' root for it, then this page'])
    if not rows:
        return ''
    return f'''
<p><b>Comparisons with other runs.</b> Every comparison within this run is like for like, and so is the one with the
stored null (section 2). The comparisons below involve other runs; the third column says what separates each, the last
what would have to run to make it like for like.</p>
{table(['comparison', 'other run', 'what differs', 'what would make it like for like'], rows)}'''


def sec_closing_limits():
    """A non-interpreted page's limits section: the native-input arms' limits and the comparisons with earlier runs."""
    body = native_limits() + earlier_runs_limits()
    return f'''
<h2>Limits: what this analysis cannot establish</h2>{body}''' if body else ''


def main():
    load()
    OUT.mkdir(parents=True, exist_ok=True)
    figs = dict(ranking=fig_ranking(), bias=fig_bias(), lead=fig_lead(), efficiency=fig_efficiency())
    contrast = () if SF is None else (sec_contrast(),)   # a results section: after section 3
    tail = (sec_critique(), sec_meaning(), sec_limits()) if INTERPRETED else (sec_closing_limits(),)
    body = '\n'.join((sec_head(), sec_why(), sec_run(), sec_results(figs)) + contrast + tail)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Simulated-effects eQTL benchmark</title><style>{CSS}</style>'
            f'</head><body><main>{body}</main></body></html>')
    C.write_atomic(PAGE, lambda fh: fh.write(page.encode()))
    print(f'wrote {PAGE} ({PAGE.stat().st_size:,} bytes) and {", ".join(p.name for p in figs.values())} in {OUT}')


if __name__ == '__main__':
    main()
