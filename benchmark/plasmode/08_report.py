"""HTML report of the benchmark (README: Report): one self-contained page from 06_score's summary,
the check files, the run facts, the joint models' summaries and the mixQTL ladder, with four
figures (embedded as base64 and written as PNG beside it). It computes nothing 06_score.py did
not, except the positions of points in the figures, small arithmetic on stored values (range
overlaps, differences and ratios) and the anchor's percentile among the per-permutation rates of
the stored null re-run under commit 8a06803 (DF_FIX, with 06_score.stored_rates). BEFORE is the
2026-09-26 run's summary of the arms before that commit (every hapmixQTL p referred to one shared
73-df t); the page compares the two where the commit changed a result. For a gene set other than
INTERPRETED_SET the interpretation paragraphs of section 3, section 3.8 and sections 4 to 6 are left
out, and a section after section 1 sets the set against the deep set's run of this code (REF_RUN) with
its selection (SELECT_LOG, POOL), the transcriptome-wide stratum rates (STRATA), the Salmon
half-depth test (HALF_DEPTH) and the committed run's dataset blocks (COMMITTED_RUN_LOG), in three
contrast figures and their tables; every input the set lacks is skipped with a printed line.
"""
import base64
import html
import json
import os
import re

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.ticker          # noqa: E402

import common as C                # noqa: E402
from tensorqtl.hapmixqtl import MIN_ALLELIC_DONORS   # noqa: E402

SC = C.module('06_score')
BEFORE = C.BEFORE_DF_FIX     # the arms scored before commit 8a06803 (not regenerable; a record)
DF_FIX = C.DF_FIX            # the stored null re-run under 8a06803 (scripts/allelic_df_null_check.py)
SMOKE = C.TRECASE_SMOKE      # the 2026-09-26 TReCASE smoke run: the largest theta gradient at an abnormal stop (section 6)
OUT, PAGE = C.REPORT, C.REPORT / 'plasmode_report.html'
INTERPRETED_SET = 'corrected_null_store_20260925'   # the gene set the interpretation prose (section 3 paragraphs, 3.8, 4-6, check_claims) was written for
INTERPRETED = C.GENE_SET == INTERPRETED_SET
REF_RUN = C.D / C.GENE_SETS[INTERPRETED_SET]['root'] / 'summary.json'   # the deep set's run of this code: the contrast on any other gene set's page (task 2026-09-27)
FIRST_RUN = C.D / C.GENE_SETS[INTERPRETED_SET]['committed']            # the first plasmode run, on the deep set (a contrast page's section 1)
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
NATIVE = SC.NATIVE_ARMS              # 05b_native_arms.py: split weighting and TReCASE on native alignment counts; () where C.NATIVE does not exist
SHOWN = ALL + NATIVE                 # every arm in the section 3 tables and Figures 1-4
LABEL = {'gibbs': 'gibbs (1/v both channels, shipped)', 'split': 'split (1/v allelic, unit total)',
         'unit': 'unit (weight 1 both channels)', 'plus_one': 'plus_one (1/(v+1) both channels)',
         'mixqtl': 'mixQTL, published cutoffs', 'mixqtl_permissive': 'mixQTL, permissive cutoffs',
         TQ: 'tensorQTL, total only, unweighted', 'rasqual': 'RASQUAL (joint model)', 'trecase': 'TReCASE, asSeq (joint model)',
         'split_native': 'split on native counts (control)', 'trecase_native': 'TReCASE, asSeq, on native counts'}
SHORT = {'gibbs': 'gibbs', 'split': 'split', 'unit': 'unit', 'plus_one': 'plus_one', 'mixqtl': 'mixQTL pub.',
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
S = SB = FX = FA = CG = CP = LF = JF = LD = SM = SF = NF = None   # the inputs, set once by load()


def skipped(what, key):
    print(f'{what} skipped: gene set {C.GENE_SET} has none (common.GENE_SETS[{C.GENE_SET!r}][{key!r}] is None)', flush=True)


def load():
    global S, SB, FX, FA, CG, CP, LF, JF, LD, SM, SF, NF
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
    SB = json.loads(BEFORE.read_text()) if BEFORE else skipped('the before/after comparison of commit 8a06803', 'before_df_fix')
    FX = json.loads(DF_FIX.read_text()) if DF_FIX else skipped('the stored null re-run under 8a06803 (anchor percentiles, tail rates)', 'df_fix')
    for path, X, arms in ((C.SUMMARY, S, ARMS), (BEFORE, SB, HM)):
        if X is not None and (tuple(X['arms']) != arms or tuple(X['joint_arms']) != JOINT or tuple(X['bands']) != BANDS):
            raise SystemExit(f'{path}: arms {X["arms"]} + {X["joint_arms"]} / bands {X["bands"]} differ from this script\'s')
    if INTERPRETED and (S['one_df_genes'] != SB['one_df_genes'] or len(S['one_df_genes']) != 1):
        raise SystemExit(f'one-df genes {S["one_df_genes"]} (this run) vs {SB["one_df_genes"]} (BEFORE); the text assumes one')
    if FX is not None and (FX['floor'] != MIN_ALLELIC_DONORS or set(HAPMIX) - set(FX['rates'])):
        raise SystemExit(f'{DF_FIX}: floor {FX["floor"]} or configurations {list(FX["rates"])} differ from this script\'s')
    CG, CP = (json.loads((C.CHECKS / f).read_text()) for f in ('check_generator.json', 'salmon_premise.json'))
    LD = json.loads((C.LADDER / 'ladder.json').read_text()) if C.LADDER else skipped('the mixQTL ladder (section 3.8)', 'ladder')
    LF = run_facts(json.loads((C.DATASETS / 'meta.json').read_text())['facts'],
                   json.loads((C.RESULTS / 'run_arms_facts.json').read_text()))
    JF = joint_facts(json.loads((C.JOINT['rasqual'] / 'summary.json').read_text()),
                     json.loads((C.JOINT['trecase'] / 'summary.json').read_text()))
    if FX is not None and LF['floor'][1] != sorted(FX['below_floor_genes']):
        raise SystemExit(f'below-floor genes {LF["floor"][1]} differ from {DF_FIX} {sorted(FX["below_floor_genes"])}')
    if SMOKE:
        SM = json.loads(SMOKE.read_text())
        if not SM['smoke']:
            raise SystemExit(f'{SMOKE} is not a smoke run')
        SM = max(d['joint_na_by_trace']['theta_fail_abs_gradient_max'] for d in SM['per_dataset'].values())
    else:
        skipped('the TReCASE smoke run (section 6)', 'trecase_smoke')
    FA = fixed_anchor() if FX is not None else None
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


def joint_facts(RS, TS):
    """Run counts of the joint arms, pooled over datasets, from their summary.json files."""
    n_ds = sum(S['n_datasets'].values())
    rq = RS['per_dataset']
    if len(rq) != n_ds or set(TS['per_dataset']) != set(rq):
        raise SystemExit(f'joint summaries cover {len(rq)} / {len(TS["per_dataset"])} datasets, want {n_ds}')
    rasqual = dict(RS['pooled'], nonconv_range=(min(v['nonconv'] for v in rq.values()), max(v['nonconv'] for v in rq.values())),
                   het=(min(v['het'] for v in rq.values()), max(v['het'] for v in rq.values())),
                   as00=(min(v['as00'] for v in rq.values()), max(v['as00'] for v in rq.values())),
                   chisq_le0_anchor=rq['beta0.0 rep 000']['chisq_le0'])
    per = TS['per_dataset'].values()
    drop = [rq[k]['het'] - c['as_records_admitted'] for k, c in TS['per_dataset'].items()]   # admitted records asSeq's min.AS.reads drops
    trace = lambda k: sum(c['joint_na_by_trace'][k] for c in per)   # noqa: E731
    trecase = dict(TS['pooled'], linear_dosage=trace('trec_linear_dosage'), ase_fail=trace('ase'),
                   few_het=sum(c['ase_na_few_het'] for c in per), df_not_1=sum(c['final_df_not_1'] for c in per),
                   constant=sorted({c['tested_constant_dosage'] for c in per}),
                   theta_gradient_max=max(c['joint_na_by_trace']['theta_fail_abs_gradient_max'] for c in per),
                   zeroed=(min(c['informative_zeroed_not_allelic_kept'] for c in per), max(c['informative_zeroed_not_allelic_kept'] for c in per)),
                   asseq_dropped=(min(drop), max(drop)), df_not_1_anchor=TS['per_dataset']['beta0.0 rep 000']['final_df_not_1'],
                   trec_na=sum(c['trec_na'] for c in per))
    return dict(rasqual=rasqual, trecase=trecase)


def fixed_anchor():
    """The anchor's rate at 0.05 against the per-permutation rates of DF_FIX's draws, per arm and channel."""
    q_lo, q_hi = (1 - SC.ANCHOR_CENTRAL) / 2, (1 + SC.ANCHOR_CENTRAL) / 2
    out = {}
    for a in HAPMIX:
        per_perm = SC.stored_rates(DF_FIX, a)
        for ch in CHANNELS:
            r, m = per_perm[ch], S['null']['beta0.0'][a][ch]['all'][str(SC.ANCHOR_ALPHA)]['rate']
            lo, hi = float(np.quantile(r, q_lo)), float(np.quantile(r, q_hi))
            out[a, ch] = dict(perm_lo=lo, perm_hi=hi, percentile=float(100 * np.mean(r <= m)), passed=lo <= m <= hi)
    return out


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


def fx(a, ch, when, al, subset='all'):
    """A rate of the stored null re-run (DF_FIX): when is 'before' (the shared 73-df reference), 'after' or
    'after_minus_before'; a dict with rate (diff) and its gene-clustered lo, hi."""
    return FX['rates'][a][ch][subset][when][al]


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


def all17():
    """Range over cutoffs and |beta| of the ladder's one-step all-17 total slope over the count-scale truth."""
    v = [LD['total_channel'][c][f'beta{b}']['one_step_all_trc']['all']['mean'] for c in ('published', 'permissive') for b in BETAS]
    return min(v), max(v)


# arm-level accessors used throughout the text: per-|beta| strings and anchor intervals
A_ = lambda a: per_beta(lambda b: auc(b, a)['mean'])                                              # noqa: E731
P_ = lambda a: per_beta(lambda b: fdp(b, a)['all']['power'])                                      # noqa: E731
R_ = lambda a: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['r2_high'], 2)                  # noqa: E731
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
    """The page's fixed comparative wording (which arm is higher, which intervals include 1, overlap or separate),
    checked against the values it is printed with; a failed claim stops the run so the sentence is reworded."""
    P = lambda b, a, ch, part, key, bn='all': prec(f'beta{b}', a, ch, part, key, bn)   # noqa: E731
    inc1 = lambda d: d['lo'] <= 1 <= d['hi']   # noqa: E731
    sep = lambda d, e: d['hi'] < e['lo'] or e['hi'] < d['lo']   # noqa: E731
    ovl = lambda d, e: not sep(d, e)   # noqa: E731
    pairs = lambda arms: [(x, y) for i, x in enumerate(arms) for y in arms[i + 1:]]   # noqa: E731
    bp = lambda a, ch, key='bias_pipeline', bn='all': [bias(b, a, ch, key, bn) for b in BETAS]   # noqa: E731
    nz = lambda a, ch: [P(b, a, ch, 'nonnull', 'sd_z') for b in BETAS]   # noqa: E731
    nr = lambda a, ch, key='ratio_vs_unit': [P(b, a, ch, 'nonnull', key) for b in BETAS]   # noqa: E731
    an = lambda a, ch, key='ratio_vs_unit', bn='all': prec('beta0.0', a, ch, 'null', key, bn)   # noqa: E731
    az = lambda a: prec('beta0.0', a, 'allelic', 'null', 'sd_z')   # noqa: E731
    r2 = lambda b, a: S['lead'][f'beta{b}'][a]['all']['r2_high']   # noqa: E731
    n05 = lambda sc, a, ch: S['null'][sc][a][ch]['all']['0.05']   # noqa: E731
    pw = lambda b, a: fdp(b, a)['all']['power']   # noqa: E731
    ex = lambda b, a: P(b, a, 'combined', 'nonnull', 'ratio_vs_unit_count')   # noqa: E731
    det = lambda b, a: S['detection'][f'beta{b}'][a]['combined']['all']['0.001']   # noqa: E731
    sq = lambda c, b, k: LD['total_channel'][c][f'beta{b}']['sq_error_vs_unit'][k]['all']   # noqa: E731
    rung = lambda c, b: LD['ladder'][f'beta{b}'][f'unit_{c}_cutoffs']['common_set']['total']['all']['value']   # noqa: E731
    scen = ('beta0.0',) + tuple(f'beta{b}' for b in BETAS)
    ga, ua = bp('gibbs', 'allelic'), bp('unit', 'allelic')
    zup = [P(b, 'unit', 'allelic', 'nonnull', 'sd_z', bn)['value'] for b in BETAS for bn in BANDS[2:]]
    claims = {
        '3.1 the four hapmixQTL arms\' AUC ranges overlap at every |beta|': all(len(overlap(x, y)) == 3 for x, y in pairs(HAPMIX)),
        '3.1 split\'s lowest AUC at 0.8 above mixQTL published\'s and RASQUAL\'s highest':
            auc('0.8', 'split')['lo'] > max(auc('0.8', 'mixqtl')['hi'], auc('0.8', 'rasqual')['hi']),
        '3.1 RASQUAL and TReCASE below split on AUC and FDP power, and below unit on AUC, at every |beta|':
            all(auc(b, j)['mean'] < min(auc(b, 'split')['mean'], auc(b, 'unit')['mean']) and pw(b, j) < pw(b, 'split') for j in JOINT for b in BETAS),
        '3.2 the four hapmixQTL arms\' BH power intervals overlap at every |beta|':
            all(ovl(bh(b, x)['power_bh']['all'], bh(b, y)['power_bh']['all']) for x, y in pairs(HAPMIX) for b in BETAS),
        '3.3 allelic bias, pipeline scale: gibbs/split and unit intervals overlap; 1/v higher at 0.2 and lower at 0.8':
            all(ovl(g, u) for g, u in zip(ga, ua)) and ga[0]['mean'] > ua[0]['mean'] and ga[2]['mean'] < ua[2]['mean'],
        '3.3 TReCASE\'s combined bias intervals alone include 1 at every |beta|; RASQUAL\'s exclude 1; its shortfall largest below 100 reads':
            [a for a in ALL if a != TQ and all(inc1(d) for d in bp(a, 'combined', 'bias_count'))] == ['trecase']
            and not any(inc1(d) for d in bp('rasqual', 'combined', 'bias_count'))
            and bias('0.8', 'rasqual', 'combined', 'bias_count', '<100')['mean'] == min(bias('0.8', 'rasqual', 'combined', 'bias_count', bn)['mean'] for bn in BANDS[1:]),
        '3.4 total sd(z): gibbs lower bounds at or above 1; split, unit and plus_one intervals include 1':
            all(d['lo'] >= 1 for d in nz('gibbs', 'total')) and all(inc1(d) for a in ('split', 'unit', 'plus_one') for d in nz(a, 'total')),
        '3.4 allelic sd(z) at the causal variant above 1 in point with intervals including 1, four arms; unit\'s excess below 100 reads':
            all(d['value'] > 1 and inc1(d) for a in HAPMIX for d in nz(a, 'allelic'))
            and min(P(b, 'unit', 'allelic', 'nonnull', 'sd_z', '<100')['value'] for b in BETAS) > max(zup),
        '3.4 anchor allelic sd(z): among the hapmixQTL arms only unit\'s interval excludes 1': [a for a in HAPMIX if not inc1(az(a))] == ['unit'],
        '3.4 allelic efficiency: plus_one and gibbs separated on the anchor, overlapping at the causal variant':
            sep(an('plus_one', 'allelic'), an('gibbs', 'allelic')) and all(ovl(p, g) for p, g in zip(nr('plus_one', 'allelic'), nr('gibbs', 'allelic'))),
        '3.4 gibbs total efficiency: the >=1000 intervals include 1, the lower bands\' values are above 1':
            all(inc1(d) and all(e['value'] > 1 for e in es) for d, es in (
                (P('0.4', 'gibbs', 'total', 'nonnull', 'ratio_vs_unit', '>=1000'), [P('0.4', 'gibbs', 'total', 'nonnull', 'ratio_vs_unit', bn) for bn in BANDS[1:3]]),
                (an('gibbs', 'total', bn='>=1000'), [an('gibbs', 'total', bn=bn) for bn in BANDS[1:3]]))),
        '3.4 plus_one total efficiency within 0.02 of 1': all(abs(d['value'] - 1) <= 0.02 for d in nr('plus_one', 'total')),
        '3.4 combined efficiency: split and plus_one below 1 in point; only split\'s interval wholly below 1, at 0.2 and 0.4 only':
            all(d['value'] < 1 for a in ('split', 'plus_one') for d in nr(a, 'combined'))
            and [b for b, d in zip(BETAS, nr('split', 'combined')) if d['hi'] < 1] == ['0.2', '0.4'] and not any(d['hi'] < 1 for d in nr('plus_one', 'combined')),
        '3.4 and 5 anchor combined efficiency: split and plus_one below 1, separated from each other, split separated from gibbs':
            an('split', 'combined')['hi'] < 1 and an('plus_one', 'combined')['hi'] < 1
            and sep(an('split', 'combined'), an('plus_one', 'combined')) and sep(an('split', 'combined'), an('gibbs', 'combined')),
        '3.4 gibbs combined efficiency: lower bounds above 1 at every |beta| and on the anchor':
            all(d['lo'] > 1 for d in nr('gibbs', 'combined')) and an('gibbs', 'combined')['lo'] > 1,
        '3.4 joint efficiency (count scale): lower bounds above 1 except TReCASE at 0.8, whose point is below split\'s with overlapping intervals':
            all(ex(b, 'rasqual')['lo'] > 1 for b in BETAS) and all(ex(b, 'trecase')['lo'] > 1 for b in BETAS[:2]) and ex('0.8', 'trecase')['lo'] <= 1
            and all(an(j, 'combined')['lo'] > 1 for j in JOINT)
            and ex('0.8', 'trecase')['value'] < ex('0.8', 'split')['value'] and ovl(ex('0.8', 'trecase'), ex('0.8', 'split'))
            and [(j, b) for j in JOINT for b in BETAS if ex(b, j)['value'] < ex(b, 'split')['value']] == [('trecase', '0.8')],
        '3.5 mixQTL published has the lowest r2 >= 0.8 share of the hapmixQTL and mixQTL arms at every |beta|':
            all(r2(b, 'mixqtl') < min(r2(b, a) for a in HM if a != 'mixqtl') for b in BETAS),
        '3.5 TReCASE\'s r2 >= 0.8 share within 0.02 of unit\'s; RASQUAL\'s the lowest of the nine arms at 0.2 and 0.4; neither above split\'s':
            all(abs(r2(b, 'trecase') - r2(b, 'unit')) <= 0.02 + 1e-12 for b in BETAS)   # shares of 150 units: 3/150 is 0.02 in float to 1e-17
            and all(r2(b, 'rasqual') < min(r2(b, a) for a in ALL if a != 'rasqual') for b in BETAS[:2])
            and all(r2(b, j) <= r2(b, 'split') for j in JOINT for b in BETAS),
        '3.6 RASQUAL\'s detection at 1e-3 below unit weights\' at every |beta|': all(det(b, 'rasqual') < det(b, 'unit') for b in BETAS),
        '3.7 gibbs combined and total null rates at 0.05: lower bounds above 0.05 in every scenario':
            all(n05(sc, 'gibbs', ch)['lo'] > 0.05 for sc in scen for ch in ('combined', 'total')),
        '3.7 TReCASE\'s 0.05 intervals above 0.05 in every scenario; RASQUAL\'s above on the anchor and including 0.05 at |beta| > 0':
            all(n05(sc, 'trecase', 'combined')['lo'] > 0.05 for sc in scen) and n05('beta0.0', 'rasqual', 'combined')['lo'] > 0.05
            and all(n05(sc, 'rasqual', 'combined')['lo'] <= 0.05 <= n05(sc, 'rasqual', 'combined')['hi'] for sc in scen[1:]),
        '3.8 at 0.8 the two-step fit separates from the one-step fit with the permissive cutoffs, overlaps with the published, and is the largest step '
        'in fold (permissive) and in increase (published, where donor admission is the larger fold); at 0.2 it is below the one-step fit, overlapping':
            sep(sq('permissive', '0.8', 'one_step_trc'), sq('permissive', '0.8', 'mixqtl_trc'))
            and ovl(sq('published', '0.8', 'one_step_trc'), sq('published', '0.8', 'mixqtl_trc'))
            and sq('permissive', '0.8', 'mixqtl_trc')['value'] / sq('permissive', '0.8', 'one_step_trc')['value']
            > max(sq('permissive', '0.8', 'one_step_trc')['value'] / sq('permissive', '0.8', 'one_step_all_trc')['value'], rung('permissive', '0.8'))
            and sq('published', '0.8', 'mixqtl_trc')['value'] - sq('published', '0.8', 'one_step_trc')['value']
            > sq('published', '0.8', 'one_step_trc')['value'] - sq('published', '0.8', 'one_step_all_trc')['value']
            and rung('published', '0.8') > sq('published', '0.8', 'mixqtl_trc')['value'] / sq('published', '0.8', 'one_step_trc')['value']
            and all(sq(c, '0.2', 'mixqtl_trc')['value'] < sq(c, '0.2', 'one_step_trc')['value'] and ovl(sq(c, '0.2', 'mixqtl_trc'), sq(c, '0.2', 'one_step_trc'))
                    for c in ('published', 'permissive')),
        '5 plus_one keeps less of the allelic gain than split on the anchor': an('plus_one', 'allelic')['value'] > an('split', 'allelic')['value'],
        '5 split ahead of both mixQTL settings in point at every |beta|: AUC, FDP power, r2 >= 0.8, total bias (count scale), count-scale efficiency':
            all(auc(b, 'split')['mean'] > auc(b, m)['mean'] and pw(b, 'split') > pw(b, m) and r2(b, 'split') > r2(b, m)
                and bias(b, 'split', 'total', 'bias_count')['mean'] > bias(b, m, 'total', 'bias_count')['mean'] and ex(b, 'split')['value'] < ex(b, m)['value']
                for m in C.MIXQTL_ARMS for b in BETAS),
        '5 mixQTL published has the lowest AUC of the hapmixQTL and mixQTL arms; permissive below split, ranges overlapping at 0.2 and 0.8 only':
            all(auc(b, 'mixqtl')['mean'] < min(auc(b, a)['mean'] for a in HM if a != 'mixqtl') for b in BETAS)
            and all(auc(b, 'mixqtl_permissive')['mean'] < auc(b, 'split')['mean'] for b in BETAS) and overlap('mixqtl_permissive', 'split') == ['0.2', '0.8'],
        '5 allelic bias (count scale): mixQTL permissive and split intervals overlap at every |beta|':
            all(ovl(bias(b, 'mixqtl_permissive', 'allelic', 'bias_count'), bias(b, 'split', 'allelic', 'bias_count')) for b in BETAS)}
    failed = [k for k, v in claims.items() if not v]
    if failed:
        raise SystemExit('the page\'s fixed wording no longer holds; reword: ' + '; '.join(failed))
    print(f'{len(claims)} fixed comparative claims of the text hold on this summary', flush=True)


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
    count-scale truth, the row where RASQUAL and TReCASE (one joint effect each) and tensorQTL appear."""
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


def cross_note():
    """gibbs's combined excess over unit weights on the count-scale truth, against the pipeline-scale one."""
    c = [prec(f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit_count') for b in BETAS]
    inc = [b for b, d in zip(BETAS, c) if d['lo'] <= 1 <= d['hi']]
    return f"""
<p>The choice of truth matters for gibbs. On this count-scale truth its combined squared error at the causal variant is
{' / '.join(ci(d, 'value', 2) for d in c)} of unit weights' at |beta| = 0.2 / 0.4 / 0.8, with intervals that include 1
at {at_betas(inc)}; on the pipeline-scale truth (the table above{' and section 4' if INTERPRETED else ''}) it is
{' / '.join(ci(prec(f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit'), 'value', 2) for b in BETAS)}.
On the count-scale truth unit weights' error contains the attenuation of log2(CPM + 1): their total slope recovers
{B_('unit', 'total', n=3)} of the count-scale total truth, gibbs's {B_('gibbs', 'total', n=3)} (section 3.3), which is consistent with the smaller
count-scale ratios but was not separated. The same denominator enters every row of this table. On the anchor's null
genes the truth is 0 for every arm, so the choice of truth drops out there, but each method's slope scale does not:
squared null error grows with the square of the slope scale, so the anchor ratio is exact among the hapmixQTL
weightings, which share one phenotype scale, and across methods it carries each method's slope scale. There gibbs's
combined squared error is {En_('gibbs', 'combined')} of unit weights'.</p>"""


def tab_conversion():
    """Each arm's published effect, its conversion to log2 aFC and where its standard error comes from."""
    truth = ('allelic: beta; total: per-gene total truth; combined: beta for bias, the inverse-variance combination '
             'of beta and the per-gene total truth for squared error')
    rows = [
        ['hapmixQTL, four weightings', 'allelic: slope of log2((L + 0.5)/(R + 0.5)) on xL &minus; xR; total: slope of '
         'log2(CPM + 1) on g/2; combined: their inverse-variance combination', 'none (log2 already)',
         'stated by the weighted least-squares fit', truth],
        ['mixQTL, two cutoff settings', 'the same three slopes on natural-log responses (asc log(L/R), trc '
         'log(total / 2 library size)); meta = inverse-variance combination', 'slope and se / ln 2',
         'stated by mixQTL\'s least-squares fits', truth],
        ['RASQUAL', '&pi;, the ALT allele\'s expected share of expression: RASQUAL scales expression by 2(1 &minus; &pi;), '
         '1, 2&pi; at ALT dosage 0, 1, 2 (nbem.c:1058)', 'log2(&pi; / (1 &minus; &pi;))',
         '<b>derived</b>: |slope| / &radic;&chi;<sup>2</sup>, &chi;<sup>2</sup> its likelihood-ratio statistic '
         '(RASQUAL reports none)', 'beta'],
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
            r, r3, n = S['anchor'][a][ch]['0.05'], S['anchor'][a][ch]['0.001'], FA[a, ch]
            rows.append([LABEL[a], ch, f(r['rate'], 6), f(r['stored'], 4), f'{f(r["perm_lo"], 6)} to {f(r["perm_hi"], 6)}',
                         f'{r["percentile"]:.1f}', 'yes' if r['passed'] else 'no',
                         f(fx(a, ch, 'after', '0.05')['rate'], 4), f'{f(n["perm_lo"], 6)} to {f(n["perm_hi"], 6)}',
                         f'{n["percentile"]:.1f}', 'yes' if n['passed'] else 'no',
                         f'{f(r3["rate"], 4)} ({f(r3["stored"], 4)} / {f(fx(a, ch, "after", "0.001")["rate"], 4)})'])
    return table(['arm', 'channel', 'this dataset, 0.05', 'stored mean, 200 permutations, before 8a06803',
                  'central 99% of stored permutations, before', 'percentile among stored, before', 'inside, before',
                  'stored mean, re-run under 8a06803', 'central 99%, re-run', 'percentile, re-run', 'inside, re-run',
                  '0.001: this dataset (stored mean before / re-run)'], rows)


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
    if not INTERPRETED:
        n_genes = S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z']['all']['genes']
        n_ds = S['n_datasets']
        ran = SF['lines'][0] * sum(n_ds.values()) // SF['lines'][1]   # datasets the arms ran on: blocks per scored dataset are the same for every dataset
        return (f'<h1>Plasmode eQTL benchmark: the {THIS_SET} ({SF["lo"]}-{SF["hi"]} reads)</h1>'
                f'<p class="sub">This page is the {THIS_SET}: {n_genes} genes drawn at random '
                f'({SF["seed"]}) from the {SF["cand"]} of {SF["pool"]} eQTL-filter genes whose median haplotype-informative '
                f'reads over admitted allelic donors lie in [{SF["lo"]}, {SF["hi"]}) and that have at least {SF["floor"]} '
                f'admitted allelic donors (median admitted reads per gene {SF["adm"][0]} to {SF["adm"][2]}, median '
                f'{SF["adm"][1]}; admitted allelic donors {SF["adm"][6]} to {SF["adm"][8]}, median {SF["adm"][7]}). It '
                f'holds {n_ds["0.0"]} beta = 0 anchor dataset and {n_ds["0.2"]} / {n_ds["0.4"]} / {n_ds["0.8"]} replicate '
                f'datasets at |beta| = 0.2 / 0.4 / 0.8, each with half the genes non-null, so every effect-size '
                f'comparison rests on {n_ds["0.4"]} replicates. hapmixQTL weightings, mixQTL mode, total-only tensorQTL, RASQUAL '
                f'and TReCASE on the BrainVar cohort\'s own Salmon output with injected effects, {n_genes} genes x 92 donors '
                f'({C.GENES}); hapmixQTL arms with commit 8a06803\'s per-channel t references and {MIN_ALLELIC_DONORS}-donor '
                f'allelic floor and with Meier\'s correction of the combined standard error for estimated channel weights '
                f'(commit a1b2ef4, section 2); this run\'s directory {C.ROOT}, with the hapmixQTL, mixQTL and tensorQTL arms '
                f'run there on 2026-09-27 and the RASQUAL and TReCASE results, which the correction does not touch, reused '
                f'from {C.COMMITTED}; units log2 aFC (beta = 1 is a twofold effect). The '
                f'section "The {THIS_SET} against the {REF_SET}", after section 1, sets it against the {REF_SET}, this '
                f'code\'s run on the {REF_SET} ({REF_RUN}), each set with its own intervals. The read bands of sections 2 and 3 use each gene\'s median '
                f'over all donors, on which {SF["below"]} of these genes fall below {SF["lo"]} reads; the set\'s own measure is '
                f'the median over admitted donors. In {C.COMMITTED.name}, the committed run whose RASQUAL and TReCASE results '
                f'are reused here, {COMMITTED_RUN_LOG.name} holds {SF["lines"][0]} dataset blocks from {ran} '
                f'datasets: its hapmixQTL and mixQTL arms first ran on {(ran - n_ds["0.0"]) // (len(n_ds) - 1)} replicates '
                f'per |beta|, the datasets were then regenerated at {n_ds["0.4"]} (user decision 2026-09-27; every generator '
                f'stream is keyed on the replicate index, so the kept replicates are unchanged), and its joint arms and '
                f'scoring used those {sum(n_ds.values())}. This run\'s hapmixQTL, mixQTL and tensorQTL arms ran once, on its own '
                f'{sum(n_ds.values())} datasets ({C.DATASETS}). Made by '
                f'benchmark/plasmode/08_report.py from {C.SUMMARY}, {REF_RUN}, {SELECT_LOG}, {POOL}, {STRATA}, {HALF_DEPTH}, '
                f'{COMMITTED_RUN_LOG}, the check files that 01_check_inputs.py wrote into {C.CHECKS} on this run '
                f'({C.ROOT / "01_check_inputs.log"}), the run facts of {C.DATASETS} and {C.RESULTS}, and the '
                f'joint models\' summaries in {C.JOINT["rasqual"]} and {C.JOINT["trecase"]}'
                + (f', and the native-input arms\' {C.NATIVE / "facts.json"} and {C.NATIVE_RESULTS["trecase_native"] / "summary.json"} '
                   f'(TReCASE and split weighting on alignment counts from the same BAMs, section {native_sec()})' if NATIVE else '')
                + '; figures also written as PNG in '
                f'{OUT}. The {REF_SET} page\'s interpretation paragraphs, its mixQTL ladder section '
                f'and its closing sections (critique, meaning, limits) are not made for this set; the contrast section '
                f'carries this set\'s comparisons, its limit and what it settles.</p>')
    return ('<h1>Plasmode eQTL benchmark: recovering known cis effects</h1>'
            '<p class="sub">hapmixQTL weightings, mixQTL mode, total-only tensorQTL, RASQUAL and TReCASE on the BrainVar '
            'cohort\'s own Salmon output with injected effects'
            + (', and TReCASE and split weighting also on alignment counts from '
               f'the same BAMs (section {native_sec()}, run 2026-09-28 into {C.NATIVE})' if NATIVE else '')
            + '; 100 genes x 92 donors; datasets of 2026-09-26, regenerated '
            'unchanged; hapmixQTL, mixQTL and tensorQTL arms and the mixQTL ladder run 2026-09-27 into '
            f'{C.ROOT}, the hapmixQTL arms on commit 8a06803 (per-channel t references and a 15-donor allelic '
            'admission floor) and with Meier\'s correction of the combined standard error for estimated channel weights '
            f'(commit a1b2ef4; section 2); RASQUAL and TReCASE of 2026-09-27, reused from {C.COMMITTED} because the '
            'correction does not touch them; units log2 aFC '
            f'(beta = 1 is a twofold effect). Made by benchmark/plasmode/08_report.py from {C.SUMMARY}, {BEFORE} (the arms '
            f'before commit 8a06803), {DF_FIX} and its draws (the stored null re-run under that commit), the check files that '
            f'01_check_inputs.py wrote into {C.CHECKS} on this run ({C.ROOT / "01_check_inputs.log"}), the run facts of {C.DATASETS} and {C.RESULTS}, the joint models\' summaries in {C.JOINT["rasqual"]} '
            f'and {C.JOINT["trecase"]}, '
            + (f'{C.LADDER}, and the native-input arms\' {C.NATIVE / "facts.json"} and '
               f'{C.NATIVE_RESULTS["trecase_native"] / "summary.json"}' if NATIVE else f'and {C.LADDER}')
            + f'; figures also written as PNG in {OUT}.</p>')


def sec_why():
    first = '''
<p>Until now the hapmixQTL weightings had been judged on null calibration only: whether the nominal p is
uniform when donor records are permuted against genotypes (the stored 100-gene, 200-permutation null runs).
A null says whether an arm's p values can be trusted. It cannot say how well an arm finds a real effect, how
close its slope comes to the true slope, or how much the Gibbs variance buys in precision, because real data
carry no known effect. No dataset with known cis effects existed. Simulating Salmon itself was rejected as
too slow, and datasets were built instead from the cohort's own Salmon output, keeping its depth, noise and
donor structure and adding a known effect.</p>''' if SF is None else f'''
<p>The first plasmode run ({FIRST_RUN.name}) built datasets with known cis effects from the cohort's own Salmon
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
reads against this set's {SF["adm"][1]}. The choice of weighting therefore rested on genes where that rate was lower.
This run repeats the benchmark, unchanged, on genes of that bin.</p>'''
    return '''
<h2>1. Why the analysis was needed</h2>''' + first + '''
<p>The question: on data with the real cohort's structure, how well do the four hapmixQTL weightings, and
mixQTL mode (the published estimator, which never sees the Gibbs draws), rank non-null genes above null ones,
discover them at a controlled false-discovery rate, estimate the injected slope without bias, state their
standard error correctly, and place the lead variant on the causal one? The answers bear on the open decision
of which weighting ships (docs/pipeline_rules.md, "Open decision: which weighting configuration ships"), which
so far rests on ''' + ('null calibration alone' if SF is None else f'null calibration and the {REF_SET}') + '''.</p>
<p>mixQTL is one published way to use allele-specific and total counts together. RASQUAL and TReCASE are two others,
which fit both kinds of count in one likelihood; they were run on the same datasets as further comparators, with every
method's effect put on one scale.''' + (''' And because mixQTL trails the unit-weight arm, a separate run took the two apart one
change at a time (section 3.8).</p>''' if INTERPRETED else '</p>')


def sec_run():
    one_df = one_df_gene() if INTERPRETED else None
    n_genes = S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z']['all']['genes']
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
            raise SystemExit(f'{C.CHECKS}: check (d) did not pass; reword section 2')
        gates = FX['gates']
        if not all(g['channel_slopes_vs_stored'] == 0 and g['channel_se_vs_stored'] == 0 and g['combined_vs_stored_admitted'] == 0
                   and g['combined_is_total_below_floor'] == 0 and g['floor_mismatches'] == 0 and g['dof_a_mismatches'] == 0
                   for g in gates.values()):
            raise SystemExit(f'{DF_FIX}: pairing gates {gates}; reword section 2')
        slope_dev = lambda v: max(v['pinned'][k]['max_slope_diff_se'] for k in ('allelic', 'total', 'combined_admitted'))   # noqa: E731
        differ = lambda d: sum(d['calls_differ'].values())   # noqa: E731
        repro = ('Check (d), exact reproduction of a stored null under commits 8a06803 and a1b2ef4: given the stored null '
                 'runs\' own permutation 0, the beta = 0 path reproduced that run\'s first permutation draw: ' + '; '.join(
                     f'{a}: {v["tests"]:,} tests; channel slopes and the admitted combined slope within '
                     f'{slope_dev(v):.1e} se of the stored ones; the admitted combined standard error within '
                     f'{v["pinned"]["combined_admitted"]["max_se_rel"]:.1e} relative of the stored one times sqrt(M), M '
                     f'Meier\'s factor recomputed from the stored channel standard errors and degrees of freedom (sqrt(M) '
                     f'{v["sqrt_meier"]["min"]:.3f} to {v["sqrt_meier"]["max"]:.3f}, median {v["sqrt_meier"]["median"]:.3f}, '
                     f'over the {v["sqrt_meier"]["tests"]:,} tests where both channels carry weight); pval_t calls at '
                     f'0.05 / 0.01 / 0.001 that differ '
                     f'{differ(v["pinned"]["pval_t"])}; pval_a and the admitted pval_nominal equal to the new references '
                     f'recomputed from the stored statistics (the combined t divided by sqrt(M)), with '
                     f'{differ(v["changed"]["pval_a"])} and '
                     f'{differ(v["changed"]["pval_nominal_admitted"])} calls differing; below the floor the combined '
                     f'statistic is the total channel\'s exactly ({v["below_floor"]["tests"]:,} tests); and against the '
                     f'stored draw the new references moved {v["calls_moved"]["combined"]["0.001"]:,} combined and '
                     f'{v["calls_moved"]["allelic"]["0.001"]:,} allelic calls at 0.001' for a, v in rp.items())
                 + '. The check covers the ' + ' and '.join(rp) + ' arms; the gates of the stored null\'s re-run '
                 f'(section 3.7) found the same exact pairing with the stored draw 0 for all of '
                 f'{", ".join(gates)} (channel slopes and standard errors, admitted combined statistic, total channel '
                 f'below the floor, the floor and dof_a all identical).')
    else:
        repro = (f'Check (d), exact reproduction of a stored null permutation, needs the stored 200-permutation null '
                 f'runs, which exist for the {INTERPRETED_SET} gene set only; it was skipped for this set '
                 f'(check_generator.json has no reproduction entry).')
    prem_by_s = '; '.join(f's in {k}: {f(v["ratio_median"], 2)} ({v["genes"]:,} genes)' for k, v in CP['by_ambiguous_share'].items())
    n_ds, mix = S['n_datasets'], LF['mix']
    rng = lambda arm, i: '-'.join(dict.fromkeys(str(g(x[i] for x in mix[arm])) for g in (min, max)))   # noqa: E731
    rq, tr, mp, em, td = JF['rasqual'], JF['trecase'], S['mixqtl_permutation'], S['eigenmt'], LF['tdiff']
    if [td['finite_one']] != tr['constant']:
        raise SystemExit(f'tensorQTL and unit weights disagree on {td["finite_one"]} pairs, not the {tr["constant"]} of constant '
                         f'dosage; reword section 2')
    miss = lambda a: ' / '.join(str(S['missing_causal'][f'beta{b}'][a]) for b in BETAS)   # noqa: E731
    set_desc = ('the 100 genes of the corrected null store' if INTERPRETED
                else f'the {n_genes} genes of the {C.GENE_SET} gene set ({C.GENES})')
    band_desc = 'fewer than 100, 100-999, at least 1,000' if INTERPRETED else BAND_HTML.replace(' / ', ', ')
    if LF['floor'] and LF['floor'][0] == 0:
        floor_txt = 'No gene falls below it in any dataset or arm (run_arms_facts.json).'
    elif LF['floor']:
        floor_txt = (f'In every\ndataset and arm the same {LF["floor"][0]} genes fall below it '
                     f'({", ".join(LF["floor"][1])}; run_arms_facts.json).')
    else:
        ns = [n for n, _ in LF['floor_sets']]
        floor_txt = (f'Between {min(ns)} and {max(ns)} genes fall below it, the set differing between datasets and arms '
                     f'as thinning moves genes across the floor (run_arms_facts.json).')
    vp = em['vs_permutation']
    calibrated = [a for a in (*HAPMIX, TQ) if a != 'gibbs']
    shape = [vp[a]['m_eff_over_shape2'] for a in HAPMIX]
    ratio = [vp[a]['eigenmt_over_pval_beta'] for a in calibrated]
    null05 = lambda a: S['null']['beta0.0'][a]['combined']['all']['0.05']   # noqa: E731
    if not (null05('gibbs')['lo'] > 0.05 and all(null05(a)['lo'] <= 0.05 for a in calibrated) and min(ratio) > 1):
        raise SystemExit('eigenMT sentence: gibbs is not the one arm above 0.05 on the beta = 0 null, or an eigenMT / pval_beta '
                         'median is at most 1; reword section 2')
    em_txt = f'''On this design shrinkage, not linkage disequilibrium, sets M<sub>eff</sub>: with {em["donors"]} donors,
fewer than the {em["window"]} variants of a window, the window's sample correlation has rank at most {em["donors"] - 1},
and unshrunk it reaches 99% of its variance with {f(em["unshrunk_share_min"], 2)} to {f(em["unshrunk_share_max"], 2)} of
the tested variants; the Ledoit-Wolf weight, a median {f(em["lw_weight_min"], 2)} to {f(em["lw_weight_max"], 2)} per gene,
puts every eigenvalue of the shrunk matrix at about that weight or more, so 99% of the variance needs most of them
(the {f(em["share_min"], 2)} to {f(em["share_max"], 2)} above). M<sub>eff</sub> is therefore a
median {f(min(shape), 1)} to {f(max(shape), 1)} times the permutation's own count of independent tests (the shape2 of the
Beta distribution map_cis fits to the hapmixQTL arms' permuted minimum p, Beta(1, M) for M independent tests), and where
pval_beta lies between {vp["unit"]["band"][0]:g} and {vp["unit"]["band"][1]:g} (the range of a dataset's Benjamini-Hochberg
thresholds) the eigenMT p is a median {f(min(ratio), 1)} to {f(max(ratio), 1)} times pval_beta for
{", ".join(SHORT[a] for a in calibrated)} ({f(vp["gibbs"]["eigenmt_over_pval_beta"], 2)} for gibbs, whose nominal p is
anticonservative, section 3.7): for an arm whose nominal p is not anticonservative the eigenMT column is conservative
relative to the permutation p (06_score.py, eigenmt_structure and eigenmt_vs_permutation).'''
    mix_perm_txt = f'''mixQTL's own permutation scan ran on every dataset at both cutoff settings, {mp["nperm"]:,}
permutations (median {' and '.join(f'{v:.0f}' for v in mp["seconds_per_dataset"].values())} s per dataset for the published
and permissive cutoffs, CPU), under mixQTL's published null: the phenotype bundle (the two haplotype counts, the total
and the library size) and the RNA-tied covariates move with the donor record, the genotype principal components stay,
the covariate offset is refitted on each permuted dataset, and no haplotype labels are swapped. Its gene-level p is the
empirical permutation p of the gene's largest |meta statistic|, (1 + the number of permuted maxima at least as
large) / (1 + the number of finite permuted maxima), without a Beta approximation, which mixQTL's port does not have.
<b>eigenMT</b> (Davis et al. 2016) gives every arm, the joint models included, a second gene-level p that needs no
permutation: a gene's effective number of independent tests, M<sub>eff</sub>, is the number of eigenvalues of its
tested variants' genotype correlation matrix (Ledoit-Wolf shrunk: the sample correlation pulled toward the identity by
a weight estimated from the data; in windows of 200 consecutive variants) needed to
explain 99% of their variance, and its gene-level p is min(1, M<sub>eff</sub> x the gene's smallest nominal p), a
Bonferroni correction over M<sub>eff</sub> tests. M<sub>eff</sub> depends on the genotypes alone, so it is the same
for every arm and dataset: {em["m_eff_min"]:,} to {em["m_eff_max"]:,} per gene (median {em["m_eff_median"]:,.0f}),
{f(em["share_min"], 2)} to {f(em["share_max"], 2)} of the tested variants. {em_txt}'''
    if INTERPRETED:
        no_fsnp_txt = f'''In one gene, {one_df}, in every dataset
({rq["no_fsnp"]} gene-datasets), RASQUAL did not admit the pseudo feature SNP and fitted the total counts alone; it is
the gene with two allelic donors discussed in section 3.7.'''
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
channel of gibbs, split and plus_one below. In the 10-99 read band both weightings fall short of the
pipeline-scale truth: {f(b10["inv_va_pipeline"]["mean"])} (gene-clustered se
{f(b10["inv_va_pipeline"]["gene_clustered_se"])}) for 1/Va' and {f(b10["unit_pipeline"]["mean"])}
({f(b10["unit_pipeline"]["gene_clustered_se"])}) for unit weights, over {b10["unit_pipeline"]["genes"]} genes.
That shortfall is not decomposed; 01_check_inputs.py names one untested candidate, the zero-haplotype admission
rule, which conditions on the thinned outcome.'''
    else:
        no_fsnp_txt = (f'RASQUAL did not admit the pseudo feature SNP, and fitted the total counts alone, in '
                       f'{rq["no_fsnp"]} gene-datasets.')
        constant_txt = (f'TReCASE has no row for the {"/".join(f"{x:,}" for x in tr["constant"])} tested pairs per dataset '
                        f'whose ALT dosage is the same in every donor.')
        nd = pr['unit_nodrop_beta']
        rec_txt = (f"Write Va for the allelic Gibbs variance of an unthinned record and Va' for that of a thinned record. "
                   f"The 1/Va' weights the arms use recover {f(pr['inv_va_beta']['mean'])} of beta (gene-clustered se "
                   f"{f(pr['inv_va_beta']['gene_clustered_se'])}), against {f(pr['inv_va_real_beta']['mean'])} when the "
                   f"weights come from the unthinned record's Va and {f(pr['inv_va_exp_beta']['mean'])} at the expected "
                   f"thinned counts; on the pipeline scale 1/Va' weights recover {f(pr['inv_va_pipeline']['mean'])} "
                   f"({f(pr['inv_va_pipeline']['gene_clustered_se'])}) against {f(pr['unit_pipeline']['mean'])} "
                   f"({f(pr['unit_pipeline']['gene_clustered_se'])}) for unit weights. Unit weights over every record with "
                   f"allelic information, the zero-haplotype drop not applied, recover {f(nd['mean'])} "
                   f"({f(nd['gene_clustered_se'])}) of beta.")
    return f'''
<h2>2. What was run</h2>
<p><b>Generator.</b> Each dataset starts from the real cohort's Salmon output for the {LF["pairs"][0]} donor-gene
pairs of {set_desc} (92 donors; {LF["pairs"][1]} pairs have
haplotype-informative reads). Three steps turn it into a dataset with a known answer. First, the real
associations are broken: donor records are permuted against fixed genotypes. A record's point estimates,
Gibbs draws, library size and RNA-tied covariates move together; the genotype principal components stay with
the genotypes; each moved record's L and R labels are swapped with probability one half. Second, an effect is
injected. For each non-null gene one causal variant is drawn among its tested variants
({LF["tested"][0]} to {LF["tested"][2]} per gene, median {LF["tested"][1]}), with |beta| = 0.2, 0.4 or 0.8 log2
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
haplotype-specific reads, and thinning scales those reads by f.</p>
<p><b>Datasets.</b> {n_ds["0.0"]} beta = 0 anchor dataset (every gene null, no thinning) and
{n_ds["0.2"]} / {n_ds["0.4"]} / {n_ds["0.8"]} datasets at |beta| = 0.2 / 0.4 / 0.8, each with {n_genes // 2} of the {n_genes} genes
non-null. Dataset r uses the same permutation, causal variants, signs and null genes at every |beta|, so the
effect sizes are paired, not independent replicates. A gene is non-null in about {n_ds["0.4"] / 2:g} of the {n_ds["0.4"]} datasets of a scenario, so every statistic below is pooled
over gene-dataset units (a <i>causal unit</i> is one non-null gene in one dataset, at its causal variant).
Genes are grouped into three <i>read bands</i> by their real median haplotype-informative reads over donors
({band_desc}). Among heterozygous donor-gene pairs of non-null genes, a share of
{LF["expr"][0]} had both haplotypes at 0.5 reads or more before thinning. Reads removed per donor, as a fraction
of the cohort's median effective library size, were at most {LF["expr"][2]:.1e} (median
{LF["expr"][1]:.1e}), so library sizes were left unchanged.</p>
<p><b>Arms.</b> Four hapmixQTL weightings, all in default mode, Var(eps) = sigma<sup>2</sup> v: eps is a record's
residual, v its Gibbs variance, and sigma<sup>2</sup> the residual scale fitted per variant. All run after the
zero-haplotype admission rule (an allelic record with exactly one
haplotype below 0.5 reads is excluded; {LF["zeroed"][0]}-{LF["zeroed"][1]} donor-gene pairs per dataset):
<b>gibbs</b>, weights 1/v in both channels (the shipped default); <b>split</b>, 1/v in the allelic channel and
weight 1 in the total channel; <b>unit</b>, weight 1 in both channels; <b>plus_one</b>, 1/(v + 1) in both
channels. All four run on commit 8a06803 and with Meier's correction (below). Each channel's p is referred to t with that channel's own residual degrees of
freedom (informative donors minus the fitted columns), and the combined p to the <i>Welch-Satterthwaite</i> degrees of
freedom of the inverse-variance combination: the degrees of freedom of the scaled &chi;<sup>2</sup> whose first two
moments match those of a fixed weighted sum of independent variance estimates, here
(w<sub>a</sub> + w<sub>t</sub>)<sup>2</sup> / (w<sub>a</sub><sup>2</sup>/&nu;<sub>a</sub> +
w<sub>t</sub><sup>2</sup>/&nu;<sub>t</sub>) with w = 1/se<sup>2</sup> and &nu; each channel's degrees of freedom. The
allelic channel enters the combined statistic only for a gene with at least {MIN_ALLELIC_DONORS} informative allelic donors, mixQTL's
own cutoff for combining its two channels; below that the combined slope, se and p are the total channel's. {floor_txt}
Since commit a1b2ef4 the combined standard error also carries <i>Meier's correction</i> (Meier 1953) for channel
weights estimated from the residuals they combine: the plug-in variance 1/(w<sub>a</sub> + w<sub>t</sub>) is
multiplied by M = 1 + 4 f<sub>a</sub> f<sub>t</sub> (1/&nu;<sub>a</sub> + 1/&nu;<sub>t</sub>), f being each
channel's share of the weight, so the combined t falls by &radic;M on the same Welch-Satterthwaite degrees of
freedom; M is 1 wherever one channel carries all the weight, and slopes and per-channel statistics are unchanged. It
applies in map_nominal and in map_cis's scan and every permutation alike. Before commit
8a06803 every hapmixQTL p was referred to t with 73 degrees of freedom (N &minus; 2 &minus; 17 covariates){
'; section 3.7 compares the two' if BEFORE else ''} (docs/hapmixqtl_methods.md, Section 4.5, has the derivation and the
reference's measured cost).
Two mixQTL-mode arms run on the thinned point estimates, never
on the draws: <b>published cutoffs</b> (total reads 100, allelic reads 50 to 1,000, weight cap 10) and
<b>permissive cutoffs</b> (20, 5 to 5,000, cap 100). The realized fold cap is min(weight cap, floor(n/10)) for n
admitted donors, at most 9 with 92 donors, so the two weight-cap settings act identically and only the count
cutoffs differ between the mixQTL arms. mixQTL's combined estimate, its <i>meta statistic</i>, is the
inverse-variance combination of its allelic (asc) and total (trc) estimates. Its natural-log slopes and standard
errors are divided by ln 2. Under the published cutoffs {rng("mixqtl", 0)} of {n_genes} genes had at least 15 allelic
donors per dataset (median allelic donors per gene {rng("mixqtl", 2)}); under the permissive cutoffs
{rng("mixqtl_permissive", 0)} (median {rng("mixqtl_permissive", 2)}). The hapmixQTL arms were also run through
map_cis for gene-level p (1,000 permutations of donor records with haplotype-label swaps, GPU), with the Beta
approximation: a Beta distribution fitted to the permuted minimum p values, used to smooth the gene-level p
(section 3.2). {mix_perm_txt}</p>
<p><b>Total-only tensorQTL.</b> tensorQTL's own cis scan (tensorqtl.cis map_nominal and map_cis) on the total phenotype T
alone, log2(CPM + 1), unweighted and with no allelic channel, with the same 17 covariates (the genotype principal
components among them as ordinary covariates) and the same tested variants per gene; map_cis permutes the
covariate-residualized phenotype 1,000 times and fits the Beta approximation. It regresses on ALT dosage g, so its
slope and standard error are doubled to put them on g/2, the scale of the hapmixQTL total channel and of the truth.
It is the standard total-expression eQTL scan, and its least-squares fit is unit weights' total channel: on dataset
{td["dataset"]} its t differs from unit weights' total-channel t by at most {td["max_abs_diff"]:.1e} over
{td["finite_both"]:,} pairs (largest |t| {td["max_abs_t"]:.1f}; tensorQTL computes in single precision), and the
{td["finite_one"]:,} pairs whose ALT dosage is the same in every donor have no tensorQTL statistic, where unit weights'
total channel returns p = 1. It is scored as a one-test arm: its one slope is the combined row, held to the total
truth (pipeline scale at the causal variant, as for the hapmixQTL arms; count scale across methods), and its squared
error is compared with unit weights' combined slope, so that ratio measures what the allelic channel adds to a
total-only scan.</p>
<p><b>Joint models.</b> Two published methods that fit the total and allele-specific counts in one likelihood were
run on every dataset, nominal only (no permutation p; eigenMT gives them a gene-level p). Each gives one test per variant, scored here as its combined
channel; their allelic and total rows read n/a. <b>RASQUAL</b> models total counts as negative binomial and
allele-specific counts as beta-binomial (the two count models that allow <i>overdispersion</i>, variance of the
counts beyond that of a Poisson or binomial count), sharing one allelic parameter, and adds a reference-mapping bias,
a sequencing error rate and genotype uncertainty; it reports a <i>likelihood-ratio statistic</i> &chi;<sup>2</sup>,
twice the gain in log-likelihood when the variant's effect is added to the model. With no
reads to give it, each gene gets one pseudo feature SNP in its gene body, at which every donor-gene pair the hapmixQTL
arms admit to the allelic channel is heterozygous with allele counts equal to its thinned haplotype point estimates
rounded to integers ({rq["het"][0]:,} to {rq["het"][1]:,} pairs per dataset); the tested variants carry the real phased
genotypes, the total counts are the thinned Salmon totals as they are (fractional), the size factor is the effective
library size, and the 17 covariates are those of the other arms. RASQUAL's defaults are kept except its
Hardy-Weinberg filter on tested variants (a test that a variant's genotype counts match those expected from its allele
frequency), turned off (-h 0) because these genotypes are the truth and no other arm filters on it (04_run_rasqual.py). <b>TReCASE</b> (asSeq 0.99.501) models total counts as negative binomial (TReC) and
allele-specific counts as beta-binomial (ASE), fits both jointly, and runs a cis/trans test of whether the total and
allelic effects agree; asSeq's final p is the joint p when that test does not reject at 0.05 and the total-count p
otherwise, which is also what it reports when the joint fit fails. Its inputs are the same donor-gene pairs as allele-specific
records (counts rounded per haplotype, because its beta-binomial needs integers), fractional totals, the log effective
library size as offset, and the same 17 covariates; asSeq's defaults are kept except the p cutoff for writing a row
(05_run_trecase.py).</p>
<p><b>One scale for every method.</b> Every slope on this page is a log2 allelic fold change (aFC), ALT over REF,
where beta = 1 is a twofold effect. The table gives each arm's published effect, its conversion, and where its
standard error comes from. RASQUAL and TReCASE report no standard error: it is derived as |slope| / &radic;&chi;<sup>2</sup>
(the Wald inversion of &chi;<sup>2</sup>: the standard error at which (slope / se)<sup>2</sup> equals
&chi;<sup>2</sup>), so under truth 0 (null genes) z = slope / se is &plusmn;&radic;&chi;<sup>2</sup>
by construction, and their realized-over-stated standard error on null genes (section 3.4) measures the calibration of
their likelihood-ratio test, not a reported standard error. At the causal variant z = (slope &minus; beta) / se instead
measures how well the derived se describes the slope's spread around beta, and absorbs bias. For
comparisons across methods every arm is held to the count-scale truth (defined in section 3.3); the pipeline-scale
truth is a hapmixQTL-only diagnostic and never ranks methods. In both joint models the total mean at ALT dosage
0, 1, 2 is proportional to 1, (1 + &kappa;)/2, &kappa; for an ALT over REF ratio &kappa;. That is the expected form,
averaged over donors, of the total fold the generator injects: thinning one haplotype's reads and the shared reads by
different factors gives exactly this form only for a donor whose haplotype-specific reads are balanced, and on average
over the random direction of real imbalance. Their estimand is therefore log2 &kappa; = beta, also when asSeq falls back
to its total-count test, whose model has the same dosage form (glm.c:1577); the per-gene total truth, a straight-line fit
of that fold on g/2, is the estimand of the linear total channels of hapmixQTL and mixQTL. One asSeq fallback cannot be
identified per test: where its total-count dosage model fails it refits the dosage as a linear covariate, whose slope
is a log fold per ALT allele, about half of ln &kappa;, so about half of beta after the conversion
({tr["linear_dosage"]:,} of {tr["tests"]:,} tests, {100 * tr["linear_dosage"] / tr["tests"]:.1f}%). Such a row at a
causal variant lowers TReCASE's bias ratio and raises its squared error; the rows are not flagged, so how many fall at
causal variants is not known.</p>
{tab_conversion()}
<p><b>Missing rows.</b> RASQUAL's rows where its fit did not converge are left out: {rq["nonconv"]:,} of {rq["tests"]:,}
tests ({rq["nonconv_range"][0]:,} to {rq["nonconv_range"][1]:,} per dataset). Another {rq["chisq_le0"]:,} rows have
&chi;<sup>2</sup> &le; 0 (p = 1), whose derived standard error is undefined: they are left out of the standard-error
statistics only, and stay in the ranking, the null rates and squared error. {no_fsnp_txt} {constant_txt} Causal
units without a row, at |beta| = 0.2 / 0.4 / 0.8 (of {S['lead']['beta0.4']['gibbs']['all']['units']} each): RASQUAL
{miss('rasqual')}, TReCASE {miss('trecase')}. They are left out of that arm's causal-variant detection shares, and in
bias and precision they are non-finite and so excluded and counted like any other non-finite unit.</p>
<p><b>Checks before the run.</b> 01_check_inputs.py was not rerun for this run: its check files are those of the same
script on the same datasets before Meier's correction, which changes the combined standard error and p (and so check
(d)'s pinned combined values) and none of checks (p) and (a) to (c). The premise of the allelic rule was tested on donor {CP["sample"]}'s dumped
equivalence classes: the observed allelic Gibbs variance beyond counting noise, over the variance predicted
from the shared-read share s, has median {f(CP["ratio_median"])} (interquartile range {f(CP["ratio_iqr"][0])} to
{f(CP["ratio_iqr"][1])}) over {CP["genes_retained"]:,} of the {CP["genes_min_u"]:,} genes with at least
{CP["min_u"]} haplotype-specific reads on each haplotype ({CP["genes_excess_le_0"]} dropped for non-positive
excess, {CP["genes_s_eq_0"]} for s = 0). The ratio is not flat in s ({prem_by_s}), so the median is set
mainly by the largest group. The rule under-predicts the excess where informative reads are a larger share and over-predicts it
where almost all reads are shared; the generator's rule does not depend on s, because thinning leaves s
unchanged. The prediction ranks genes' excess with Spearman correlation (the Pearson correlation of the ranks)
{f(CP["spearman_excess_s2H"])} against {f(CP["spearman_excess_counting"])} for the counting term alone. Its pass
thresholds were set after that first result, so it guards the derivation against regression rather than testing
it independently. Check (a), identity: with every thinning factor 1,
the generator reproduces the pipeline's inputs exactly: A, the allelic log2 ratio log2((pL + 0.5)/(pR + 0.5)); T,
the total log2(CPM + 1); and Va and Vt, their Gibbs variances (largest difference in A after a permutation and swap
{idn["max_abs_dA"]:.1e}). Check (b), thinning: the Fano factor (across-draw variance over mean) of the
total Gibbs draws stays at its real value after thinning by f = {th["f"]}, and the allelic rule's arithmetic
holds to {th["allelic_rule"]["max_rel_dev"]:.1e} relative over {th["allelic_rule"]["records_checked"]:,}
records:</p>
{fano}
<p>Check (c), recovery of an injected |beta| = {rc["beta"]} over {rc["n_datasets"]} all-non-null datasets
({pr["unit_pipeline"]["genes"]} genes{f' with at least {mr} reads' if mr else ''}, {pr["unit_pipeline"]["units"]:,} units). Unit weights
recover the pipeline-scale truth ({f(pr["unit_pipeline"]["mean"])}, gene-clustered se
{f(pr["unit_pipeline"]["gene_clustered_se"])}), which is the pass rule. Here the gene-clustered se is the standard
deviation of the per-gene means over the square root of the number of genes, as 01_check_inputs.py computes it; it is
not the resampling interval used in section 3. {rec_txt}</p>
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
    Pb = lambda a: per_beta(lambda b: SB['ranking'][f'beta{b}'][a]['fdp_matched']['all']['power'])   # noqa: E731
    dP = max(abs(fdp(b, a)['all']['power'] - SB['ranking'][f'beta{b}'][a]['fdp_matched']['all']['power']) for b in BETAS for a in HAPMIX)
    dA = max(abs(auc(b, a)['mean'] - SB['ranking'][f'beta{b}'][a]['auc']['all']['mean']) for b in BETAS for a in HAPMIX)
    g1 = one_df_gene()
    return f"""
<p>At |beta| = 0.2 / 0.4 / 0.8 the AUC is {A_('split')} for split and {A_('plus_one')} for plus_one,
{A_('unit')} for unit and {A_('gibbs')} for gibbs; mixQTL reaches {A_('mixqtl')} with the published cutoffs and
{A_('mixqtl_permissive')} with the permissive ones. The four hapmixQTL arms' ranges overlap at every |beta|. At
|beta| 0.8 split's lowest dataset AUC ({f(auc('0.8', 'split')['lo'])}) exceeds mixQTL published's highest
({f(auc('0.8', 'mixqtl')['hi'])}), so split ranks higher in each of the three datasets. A narrow range such as
unit's {ci(auc('0.4', 'unit'), 'mean')} at |beta| 0.4 means three datasets happened to agree, not that the estimate
is precise, so arms should not be ordered by the width of these ranges.</p>
<p>Power at 5% realized FDP spreads the arms further: split {P_('split')}, plus_one {P_('plus_one')}, unit
{P_('unit')}, gibbs {P_('gibbs')}; mixQTL {P_('mixqtl')} (published) and {P_('mixqtl_permissive')} (permissive).
{gibbs_low()} At 0.4 and 0.8 its cut fell at a lead p of {thr('0.4', 'gibbs')}
and {thr('0.8', 'gibbs')}, where split's fell at {thr('0.4', 'split')} and {thr('0.8', 'split')}: null genes' lead p
reached below split's cut, so to keep them out gibbs had to stop at smaller p. Section 4 takes up whether that
reflects null p values that are too small.</p>
<p><b>The allelic admission floor and the ranking.</b> Before commit 8a06803, {g1}'s allelic p was too small in the
four hapmixQTL arms (section 3.7), while mixQTL, RASQUAL and TReCASE all leave that gene's allelic channel out, so the
earlier version of this page called the hapmixQTL ranking power a lower bound in the comparison with the other
methods. Since the commit hapmixQTL leaves it out as well (below the floor its combined statistic is the total
channel's), and that exposure is gone. Rescored, power at 5% realized FDP went from {Pb('split')} to {P_('split')} for
split, {Pb('unit')} to {P_('unit')} for unit, {Pb('plus_one')} to {P_('plus_one')} for plus_one and {Pb('gibbs')} to
{P_('gibbs')} for gibbs at |beta| = 0.2 / 0.4 / 0.8: at most {f(dP)} in either direction, and the AUC by at most
{f(dA)}. The change mixes the floor with the per-pair reference of every other gene and with Meier's correction (the
ranking is by lead p), so it cannot be assigned to {g1} alone. Every ranking value on this page is the rescored one.</p>"""


def interp_gene_level():
    P = lambda a: per_beta(lambda b: bh(b, a)['power_bh']['all']['rate'])   # noqa: E731
    N = lambda a: ' / '.join(f'{bh(b, a)["false_discoveries"]} of {bh(b, a)["discoveries"]}' for b in BETAS)   # noqa: E731
    Pe = lambda a: per_beta(lambda b: bhe(b, a)['power_bh']['all']['rate'])   # noqa: E731
    Pm = lambda a: per_beta(lambda b: bhe(b, a)['fdp_matched']['all']['power'])   # noqa: E731
    nre = lambda a: S['gene_level_eigenmt']['beta0.0'][a]['null_rate']['all']   # noqa: E731
    nr = lambda sc, a: S['gene_level'][sc][a]['null_rate']['all']   # noqa: E731
    worst = max(((nr(f'beta{b}', a)['rejections'], b, a) for b in BETAS for a in HAPMIX))
    wr = nr(f'beta{worst[1]}', worst[2])
    anc = ' / '.join(f'{nr("beta0.0", a)["rejections"]}' for a in HAPMIX)
    return f"""
<p>With each arm referred to its own permutation null, Benjamini-Hochberg power at |beta| = 0.2 / 0.4 / 0.8 is
{P('split')} for split, {P('gibbs')} for gibbs, {P('plus_one')} for plus_one and {P('unit')} for unit. The
gene-clustered intervals of the four arms overlap at every |beta| (table), so at {S['lead']['beta0.4']['gibbs']['all']['units']} non-null gene units per
effect size the arms are not distinguishable on gene-level power. Null genes among the calls: gibbs {N('gibbs')},
split {N('split')}, unit {N('unit')}, plus_one {N('plus_one')}; these are small counts without an interval.
Here the generator's null and map_cis's null are the same record permutation with label swaps (RNA-tied
covariates moving, genotype PCs fixed, one thinning factor per null gene), so the permutation p of a null gene
is valid by construction. The share of null gene units with pval_beta below 0.05 (at |beta| &gt; 0 at most
{worst[0]} of {wr["tests"]} in any arm and scenario, gene-clustered interval {f(wr["lo"])} to {f(wr["hi"])}; on the
anchor {anc} of {nr("beta0.0", "gibbs")["tests"]} for gibbs / split / unit / plus_one) therefore checks the
plumbing and the Beta approximation. It says nothing about calibration on real data, where the record
permutation may not match the sampling distribution of an observed statistic (section 6). tensorQTL's and mixQTL's
permutation nulls are their own (section 2), not the generator's, so that argument does not carry over to them.</p>
<p><b>The other arms, and eigenMT.</b> On its own permutation p total-only tensorQTL reaches Benjamini-Hochberg power
{P(TQ)} and mixQTL {P('mixqtl')} (published) and {P('mixqtl_permissive')} (permissive) at |beta| = 0.2 / 0.4 / 0.8, with
{N(TQ)}, {N('mixqtl')} and {N('mixqtl_permissive')} null genes among the calls. On the eigenMT p the power is
{'; '.join(f'{SHORT[a]} {Pe(a)}' for a in ALL)}. The eigenMT p multiplies an arm's smallest nominal p by
M<sub>eff</sub>, so it inherits whatever miscalibration that nominal p has (section 3.7), where the permutation p is
referred to the arm's own null; null gene units with eigenMT p below 0.05 on the anchor:
{'; '.join(f'{SHORT[a]} {nre(a)["rejections"]} of {nre(a)["tests"]}' for a in ALL)}. So the eigenMT power above
rewards an anticonservative nominal p: an arm that calls more null genes also calls more non-null ones. Held to the same
5% realized false-discovery proportion on the same eigenMT p (Figure 1 D; the gene units of the scenario's datasets
ranked by it and cut, using the truth, where at most 5% of the units called are null; no interval) the power is
{'; '.join(f'{SHORT[a]} {Pm(a)}' for a in ALL)}, and Figure 1 E gives each arm's realized false-discovery proportion of
its Benjamini-Hochberg calls.</p>"""


def interp_bias():
    ladder_all17 = '{:.2f} to {:.2f}'.format(*all17())
    B = lambda a, ch, key, bn='all': per_beta(lambda b: bias(b, a, ch, key, bn)['mean'])   # noqa: E731
    lo100 = lambda a: per_beta(lambda b: bias(b, a, 'allelic', 'bias_pipeline', '<100')['mean'], 2)   # noqa: E731
    pc = lambda b: bias(b, 'gibbs', 'allelic', 'bias_pipeline')   # noqa: E731
    cc = lambda b: bias(b, 'gibbs', 'allelic', 'bias_count')   # noqa: E731
    mq, mp = (bias('0.4', a, 'allelic', 'bias_count') for a in ('mixqtl', 'mixqtl_permissive'))
    return f"""
<p><b>Total channel.</b> Every hapmixQTL arm recovers the pipeline-scale truth: {B('gibbs', 'total', 'bias_pipeline')}
for gibbs, {B('split', 'total', 'bias_pipeline')} for split and unit (which share one total-channel fit) and
{B('plus_one', 'total', 'bias_pipeline')} for plus_one, with intervals that include 1. Against the count-scale truth
split and unit's slopes read {B('split', 'total', 'bias_count')} (gibbs {B('gibbs', 'total', 'bias_count')}, plus_one
{B('plus_one', 'total', 'bias_count')}): the +1 of log2(CPM + 1) compresses a fold at low depth, and the
pipeline-scale truth removes that part.</p>
<p><b>Allelic channel.</b> gibbs and split fit the allelic channel identically (every allelic row of the two arms is
the same fit, not two arms agreeing). Their allelic slope recovers {B('gibbs', 'allelic', 'bias_pipeline')} of the
pipeline-scale truth, against {B('unit', 'allelic', 'bias_pipeline')} for unit weights and
{B('plus_one', 'allelic', 'bias_pipeline')} for plus_one. The gibbs/split and unit intervals overlap at every |beta|,
and the order is not constant: 1/v is higher at 0.2, about equal at 0.4 and lower at 0.8. The benchmark therefore
does not resolve a bias difference between 1/v and unit weights. The roughly 5% attenuation from 1/v weights rests
on check (c) (section 2): {f(CG['recovery']['primary']['inv_va_pipeline']['mean'])} against
{f(CG['recovery']['primary']['unit_pipeline']['mean'])} for unit weights, on the pipeline-scale truth, over
{CG['recovery']['n_datasets']} all-non-null datasets. Unit weights also fall short of the
pipeline-scale truth below 100 reads ({lo100('unit')}; gibbs and split {lo100('gibbs')}), with wide intervals (gibbs
and split at |beta| 0.8, {ci(bias('0.8', 'gibbs', 'allelic', 'bias_pipeline', '<100'), 'mean')}). That shortfall is
neither the transform, which the pipeline-scale truth removes, nor weighting, since the weights are unit; it is not
attributed here. One untested candidate, named in 01_check_inputs.py: the zero-haplotype admission rule selects on
the thinned outcome.</p>
<p><b>Two denominators in the allelic rows.</b> In {pc('0.4')['excluded_nonfinite']} of the {cc('0.4')['units']}
causal units per |beta| the allelic channel has no admitted heterozygous donor: map_nominal returns slope 0 with an
infinite standard error and p = 1 (a NaN p since commit 8a06803 where the gene's allelic channel is switched off
altogether). Their pipeline-scale truth is undefined, so the pipeline line drops them
({pc('0.4')['units']} units); the count-scale truth is beta, so the count line keeps them as exact zeros
({cc('0.4')['units']} units), which lowers the count-scale mean by the factor {pc('0.4')['units']}/{cc('0.4')['units']}
against a mean over the units with data. Most of the gap between the two lines of a hapmixQTL allelic row is these
zeros, not the +0.5 pseudocount; the table prints each line's units and exclusions. The allelic sd(z) and efficiency ratios at
the causal variant (section 3.4) use the {pc('0.4')['units']} units with data. mixQTL's count-scale allelic rows drop
their units without an estimate, so they are on a different filter from the hapmixQTL count lines.</p>
<p><b>Combined slope.</b> Against beta it reads {B('gibbs', 'combined', 'bias_count')} for gibbs and
{B('split', 'combined', 'bias_count')} for split. It mixes the two channels' estimands and both transforms'
attenuation, so it is not a measure of estimator bias on its own.</p>
<p><b>mixQTL.</b> Its total slope recovers {B('mixqtl', 'total', 'bias_count')} (published) and
{B('mixqtl_permissive', 'total', 'bias_count')} (permissive) of the count-scale truth; at &ge;1000 reads and
|beta| 0.8 it is still {ci(bias('0.8', 'mixqtl', 'total', 'bias_count', '>=1000'), 'mean', 2)} (published) and
{ci(bias('0.8', 'mixqtl_permissive', 'total', 'bias_count', '>=1000'), 'mean', 2)} (permissive). mixQTL's response has
no +1 and no pseudocount, so this is not the low-depth transform. Section 3.8 explains most of it: mixQTL fits its
covariate offset from the covariates alone, before the variant enters, and then regresses on the genotype without
adjusting it for those covariates, which multiplies the slope by one minus the share of the genotype's variance they
explain; and it chooses those covariates on the outcome, which carries the effect. A one-step fit with all 17
covariates, which has neither feature, recovers {ladder_all17} of the truth in that section's units, and
that remainder is not decomposed. On a common
count-scale truth the total slope at |beta| 0.8 is {ci(bias('0.8', 'split', 'total', 'bias_count'), 'mean', 2)} for
split and unit against {ci(bias('0.8', 'mixqtl', 'total', 'bias_count'), 'mean', 2)} (published) and
{ci(bias('0.8', 'mixqtl_permissive', 'total', 'bias_count'), 'mean', 2)} (permissive) for mixQTL, the comparison
Figure 2 draws. mixQTL's allelic slope recovers
{B('mixqtl', 'allelic', 'bias_count')} (published) and {B('mixqtl_permissive', 'allelic', 'bias_count')} (permissive).
At |beta| 0.4 the published cutoffs leave {mq['excluded_nonfinite']} of {mq['units'] + mq['excluded_nonfinite']}
allelic units without a finite estimate and the permissive cutoffs {mp['excluded_nonfinite']}; each count includes
the {pc('0.4')['excluded_nonfinite']} units with no allelic data in any arm (above).</p>"""


def interp_precision():
    moved = [a for a in HAPMIX if prec('beta0.0', a, 'allelic', 'null', 'sd_z')['value']
             != SB['precision']['beta0.0'][a]['allelic']['null']['sd_z']['all']['value']]
    if moved:
        raise SystemExit(f'{C.SUMMARY} vs {BEFORE}: anchor allelic sd(z) changed for {moved}; reword section 3.4')
    Ec = lambda a, ch: ' / '.join(ci(prec(f'beta{b}', a, ch, 'nonnull', rkey(a)), 'value', 2) for b in BETAS)   # noqa: E731
    zx = lambda a: prec('beta0.0', a, 'allelic', 'null', 'sd_z', NO_ONE_DF)   # noqa: E731
    Zx = lambda a: ci(zx(a), 'value', 3)   # noqa: E731
    side = lambda a: 'above 1' if zx(a)['lo'] > 1 else 'below 1' if zx(a)['hi'] < 1 else 'including 1'   # noqa: E731
    lo = lambda a, ch, key: ' / '.join(f(prec(f'beta{b}', a, ch, 'nonnull', key)['lo']) for b in BETAS)   # noqa: E731
    hi = lambda a, ch: ' / '.join(f(prec(f'beta{b}', a, ch, 'nonnull', 'ratio_vs_unit')['hi']) for b in BETAS)   # noqa: E731
    bands = lambda a, ch: ' / '.join(f(prec('beta0.4', a, ch, 'nonnull', 'ratio_vs_unit', bn)['value'], 2) for bn in BANDS[1:])   # noqa: E731
    bands_ci = lambda sc, part, a, ch: ' / '.join(ci(prec(sc, a, ch, part, 'ratio_vs_unit', bn), 'value', 2) for bn in BANDS[1:])   # noqa: E731
    zlow = lambda a: per_beta(lambda b: prec(f'beta{b}', a, 'allelic', 'nonnull', 'sd_z', '<100')['value'], 2)   # noqa: E731
    zup = [prec(f'beta{b}', 'unit', 'allelic', 'nonnull', 'sd_z', bn)['value'] for b in BETAS for bn in BANDS[2:]]
    mix_anchor_low = f(prec('beta0.0', 'mixqtl', 'combined', 'null', 'ratio_vs_unit', '<100')['value'], 2)
    mix_anchor_up = ' / '.join(f(prec('beta0.0', 'mixqtl', 'combined', 'null', 'ratio_vs_unit', bn)['value'], 2) for bn in BANDS[2:])
    return f"""
<p><b>Stated standard error.</b> The clean comparison is the total channel, where the hapmixQTL arms share the truth
and the donors. At the causal variant sd(z) is {Z_('gibbs', 'total')} for gibbs against {Z_('split', 'total')} for
split and unit and {Z_('plus_one', 'total')} for plus_one; gibbs's intervals start at or above 1 (lower bounds
{lo('gibbs', 'total', 'sd_z')}), the others' include 1. On the anchor's null genes gibbs reads {Zn_('gibbs', 'total')}
and unit {Zn_('unit', 'total')}. So the Gibbs-weighted total channel's slope varies more than its stated se says, by
these factors, on genes with an effect as on genes without one, while unit weights state it correctly.</p>
<p>In the allelic channel the point value of sd(z) at the causal variant is above 1 for the four hapmixQTL arms
(gibbs and split {Z_('gibbs', 'allelic')}, unit {Z_('unit', 'allelic')}, plus_one {Z_('plus_one', 'allelic')}), but every
interval includes 1 (lower bounds gibbs and split {lo('gibbs', 'allelic', 'sd_z')}, unit
{lo('unit', 'allelic', 'sd_z')}, plus_one {lo('plus_one', 'allelic', 'sd_z')}). The point excess sits below 100 reads
(unit {zlow('unit')} there, against {f(min(zup), 2)} to {f(max(zup), 2)} in the two higher bands), which is also where
the allelic bias is largest (section 3.3). At the causal variant sd(z) absorbs bias whose sign follows the random
sign of beta, so this excess may be bias rather than a stated standard error that is too small; the two were not
separated. mixQTL's allelic channel reads {Z_('mixqtl_permissive', 'allelic')} with the permissive cutoffs and
{Z_('mixqtl', 'allelic')} with the published ones. On the anchor's null genes, among the hapmixQTL arms only unit's
interval excludes 1 ({Zn_('unit', 'allelic')}); gibbs and split read {Zn_('gibbs', 'allelic')} and plus_one
{Zn_('plus_one', 'allelic')}; mixQTL reads {Zn_('mixqtl', 'allelic')} (published) and
{Zn_('mixqtl_permissive', 'allelic')} (permissive). Much of the hapmixQTL arms' anchor excess is one null gene,
{one_df_gene()}, whose allelic scale is fitted on two donors, one residual degree of freedom, so its stated se is
itself a one-degree-of-freedom estimate and its z is heavy-tailed; mixQTL fits no allelic channel on two donors. sd(z)
does not depend on the t reference, so commit 8a06803 (section 3.7) leaves these values as they were: it changes
which p the gene's z is referred to, not z. Without that gene the hapmixQTL arms read {Zx('gibbs')}
(gibbs and split, interval {side('gibbs')}), {Zx('unit')} (unit, {side('unit')}) and {Zx('plus_one')} (plus_one,
{side('plus_one')}). Two cautions. sd(z) responds to a few extreme values while a
rejection rate at 0.05 does not: unit's allelic null rate on the anchor is
{ci(S['null']['beta0.0']['unit']['allelic']['all']['0.05'], 'rate', 4)}. And the bias contribution at the causal
variant is consistent with mixQTL's total sd(z) rising with |beta| ({Z_('mixqtl', 'total')} published) alongside its
attenuated slope; that was not tested either.</p>
<p><b>Efficiency.</b> In the allelic channel the gibbs and split squared error is {E_('gibbs', 'allelic')} of unit
weights' at the causal variant and {En_('gibbs', 'allelic')} on the anchor's null genes; by read band
(&lt;100 / 100-999 / &ge;1000, |beta| 0.4) it is {bands('gibbs', 'allelic')}, the gain growing with depth. plus_one keeps
less of that gain. On the anchor's null genes its allelic ratio is {En_('plus_one', 'allelic')} against gibbs and
split's {En_('gibbs', 'allelic')}, with separated intervals; at the causal variant ({Ec('plus_one', 'allelic')} against
{Ec('gibbs', 'allelic')}) the intervals overlap.</p>
<p>In the total channel gibbs's squared error is {E_('gibbs', 'total')} of unit weights' at the causal variant and
{En_('gibbs', 'total')} on the anchor's null genes. That cost comes from genes below 1,000 reads. By band
(&lt;100 / 100-999 / &ge;1000) it is {bands_ci('beta0.4', 'nonnull', 'gibbs', 'total')} at the causal variant at
|beta| 0.4, and {bands_ci('beta0.0', 'null', 'gibbs', 'total')} on the anchor's null genes; at 1,000 reads or more
these data show neither a cost nor a gain. plus_one is unit weighting in practice in this channel
({E_('plus_one', 'total')}).</p>
<p>For the combined slope, both split ({E_('split', 'combined')}) and plus_one ({E_('plus_one', 'combined')}) are below
1 in point estimate at every |beta|. Only split's interval lies wholly below 1, and only at |beta| 0.2 and 0.4 (upper
bounds {hi('split', 'combined')}; plus_one's {hi('plus_one', 'combined')}). On the anchor's null genes both lie below 1
with separated intervals (split {En_('split', 'combined')}, plus_one {En_('plus_one', 'combined')}). At the causal
variant split's gain is therefore about 10% of unit weights' squared error and plus_one's about 2%. gibbs is above 1 at
the causal variant ({E_('gibbs', 'combined')}, lower bounds {lo('gibbs', 'combined', 'ratio_vs_unit')}) and on the
anchor ({En_('gibbs', 'combined')}).</p>
<p>For mixQTL the ratio is a comparison of methods as run. The published arm's allelic ratio is
{En_('mixqtl', 'allelic')} on the anchor's null genes and {Ec('mixqtl', 'allelic')} at the causal variant; the reversal
is resolved only at |beta| 0.8. That is what 03_run_arms.py's caveat predicts (its allelic cap of 1,000 reads makes
admission depend on the injected effect), but it was not tested separately. The published arm's combined ratio is
{E_('mixqtl', 'combined')} at the causal variant, highest at |beta| 0.8. It is already {En_('mixqtl', 'combined')} on the
anchor's null genes, where there is no effect to attenuate, so most of it is not attenuation. On the anchor it sits
in the genes below 100 reads ({mix_anchor_low} there, against {mix_anchor_up} in the two higher bands). That points
to which donors the published count cutoffs admit, but it was not tested separately. The permissive arm's ratio is
{En_('mixqtl_permissive', 'combined')} on the anchor and {E_('mixqtl_permissive', 'combined')} at the causal variant,
growing with |beta|, as expected for a combined slope that recovers {B_('mixqtl_permissive', 'combined')} of beta: its squared
error includes the missing share of the effect, which grows with |beta|. Section 3.8 decomposes the total channel's
part of this gap.</p>"""


def interp_lead():
    Cc = lambda a: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['lead_is_causal'], 2)   # noqa: E731
    hm = lambda i: [S['lead'][f'beta{BETAS[i]}'][a]['all']['r2_high'] for a in HAPMIX]   # noqa: E731
    rng = lambda i: f'{f(min(hm(i)), 2)}-{f(max(hm(i)), 2)}'   # noqa: E731
    und = {S['lead'][f'beta{b}'][a]['r2_undefined'] for b in BETAS for a in ARMS}
    if len(und) != 1:
        raise SystemExit(f'{C.SUMMARY}: r2_undefined differs between arms or |beta| ({und}); reword section 3.5')
    und = und.pop()
    return f"""
<p>The share of non-null gene units whose lead is within r<sup>2</sup> &ge; 0.8 of the causal variant is
{R_('split')} for split, {R_('gibbs')} for gibbs, {R_('unit')} for unit and {R_('plus_one')} for plus_one; mixQTL reaches
{R_('mixqtl')} (published) and {R_('mixqtl_permissive')} (permissive). The lead is the causal variant itself in
{Cc('split')} of units for split and {Cc('mixqtl')} for mixQTL published. The summary carries no interval for these
shares, so the differences among the four hapmixQTL arms are not interpreted. Among the hapmixQTL and mixQTL arms,
mixQTL with the published cutoffs has the lowest point value at every |beta| ({R_('mixqtl')} against {rng(0)} / {rng(1)} / {rng(2)} for the hapmixQTL arms);
with no interval, that ordering is descriptive. It has no finite p at all in
{' / '.join(str(S['lead'][f'beta{b}']['mixqtl']['no_finite_p']) for b in BETAS)} of its non-null gene units. In every
arm and |beta|, {und} unit has a causal or lead variant with one dosage in every donor, so its r<sup>2</sup> is
undefined; it counts as not recovered in the shares. The median r<sup>2</sup> is over units where r<sup>2</sup> is
defined: {S['lead']['beta0.4']['gibbs']['all']['r2_defined']} for the hapmixQTL arms and
{' / '.join(str(S['lead'][f'beta{b}']['mixqtl']['all']['r2_defined']) for b in BETAS)} for mixQTL published, whose units
without a finite p drop out of its median and lift it.</p>"""


def interp_detection():
    st = lambda a: ci(fx(a, 'combined', 'after', '0.001'), 'rate', 4)   # noqa: E731
    r3 = [fx(a, 'combined', 'after', '0.001')['rate'] / 0.001 for a in HAPMIX[1:]]
    nod = S['recovery']['beta0.4']['gibbs']['allelic']['bias_pipeline']['all']['excluded_nonfinite']
    return f"""
<p>At p &lt; 1e-3 the combined statistic detects the causal variant in {D_('gibbs')} of non-null gene units
for gibbs, {D_('split')} for split, {D_('unit')} for unit and {D_('plus_one')} for
plus_one; mixQTL {D_('mixqtl')} (published) and {D_('mixqtl_permissive')} (permissive). The
allelic channel shows the weights most directly: gibbs and split {D_('gibbs', 'allelic')} against unit
{D_('unit', 'allelic')} and plus_one {D_('plus_one', 'allelic')}. In the total channel detection is close across the
hapmixQTL arms (gibbs {D_('gibbs', 'total')}, unit {D_('unit', 'total')}), but gibbs's total channel rejects too often
on null genes (section 3.7), so its detections there are not comparable at face value. The same holds for gibbs's
combined statistic: at 1e-3 its 200-permutation null rate on the stored null re-run under commit 8a06803 (made before
Meier's correction) is
{st('gibbs')}, against {st('split')} / {st('unit')} / {st('plus_one')} for split / unit / plus_one (section 3.7);
so its combined detection ({D_('gibbs')}) is not comparable at face value either. The other three arms'
rates there are {f(min(r3), 2)} to {f(max(r3), 2)} times nominal. No null rate at 1e-5 was
measured. In the allelic channel
{nod} of the {S['detection']['beta0.4']['gibbs']['allelic']['all']['units']} units per |beta| have no allelic data
(section 3.3) and cannot be detected there in any arm; this is the same for every arm, so it does not change their
order.</p>"""


def interp_null():
    Rb = lambda a, ch, al: per_beta(lambda b: S['null'][f'beta{b}'][a][ch]['all'][al]['rate'], 4)   # noqa: E731
    sv = lambda a, ch, al: fx(a, ch, 'after', al)['rate']   # noqa: E731  the stored null re-run under 8a06803
    svf = lambda a, ch, al: f(sv(a, ch, al), 4)   # noqa: E731
    bench = lambda a, ch, al: [S['null'][f'beta{b}'][a][ch]['all'][al] for b in BETAS]   # noqa: E731
    covers = lambda a, ch, al: all(d['lo'] <= sv(a, ch, al) <= d['hi'] for d in bench(a, ch, al))   # noqa: E731
    lower = lambda a, ch, al: all(d['rate'] < sv(a, ch, al) for d in bench(a, ch, al))   # noqa: E731
    dmax = max(abs(d['rate'] - sv('gibbs', ch, '0.05')) for ch in ('allelic', 'total') for d in bench('gibbs', ch, '0.05'))
    AL = ('gibbs', 'unit', 'plus_one')   # split's allelic channel is gibbs's fit
    name = dict(gibbs='gibbs and split', unit='unit', plus_one='plus_one')
    if not (covers('gibbs', 'total', '0.05') and covers('gibbs', 'allelic', '0.05')
            and all(covers(a, 'allelic', '0.001') for a in AL) and not lower('gibbs', 'total', '0.001')):
        raise SystemExit(f'{C.SUMMARY} vs {DF_FIX}: |beta| > 0 null rates no longer as section 3.7 describes; reword it')
    low = [a for a in AL if lower(a, 'allelic', '0.001')]
    notlow = [a for a in AL if a not in low]
    vs = lambda a: f'{Rb(a, "allelic", "0.001")} against {svf(a, "allelic", "0.001")}'   # noqa: E731
    low_txt = ('the allelic point rates are lower than that run\'s for ' + ' and for '.join(f'{name[a]} ({vs(a)})' for a in low)
               + ('' if not notlow else ', and not for ' + ' or for '.join(f'{name[a]} ({vs(a)})' for a in notlow))
               if low else 'no arm\'s allelic point rates are lower than that run\'s (' + '; '.join(f'{name[a]} {vs(a)}' for a in AL) + ')')
    g1 = one_df_gene()
    before = lambda a, ch='combined': fx(a, ch, 'before', '0.001')   # noqa: E731
    after = lambda a, ch='combined': fx(a, ch, 'after', '0.001')   # noqa: E731
    dif = lambda a: fx(a, 'combined', 'after_minus_before', '0.001', 'n_a >= 40')   # noqa: E731
    tb = [before(a)['rate'] for a in HAPMIX]
    ta = [after(a)['rate'] for a in HAPMIX[1:]]
    g1r = lambda a, when: FX['below_floor_genes'][g1][a]['combined'][when]['0.001']   # noqa: E731
    n3 = lambda X, a, g: X['null']['beta0.0'][a]['combined'][g]['0.001']['rejections']   # noqa: E731
    held = lambda X, a: f'{n3(X, a, "all") - n3(X, a, NO_ONE_DF):,} of {a}\'s {n3(X, a, "all"):,}'   # noqa: E731
    dof40 = [FX['dof'][a]['dof_nominal_draw0']['n_a >= 40']['median'] for a in HAPMIX[1:]]
    n40 = FX['genes_per_subset']['n_a >= 40']
    if any(FX['verdict'][a]['contains_0_001'] for a in HAPMIX[1:]) or FX['passed']:
        raise SystemExit(f'{DF_FIX}: verdict {FX["verdict"]}; reword section 3.7')
    old = [(a, ch, S['anchor'][a][ch]['0.05']) for a in HAPMIX for ch in CHANNELS if not S['anchor'][a][ch]['0.05']['passed']]
    new = [(a, ch, FA[a, ch]) for a in HAPMIX for ch in CHANNELS if not FA[a, ch]['passed']]
    rate05 = lambda a, ch: S['null']['beta0.0'][a][ch]['all']['0.05']   # noqa: E731
    outside = lambda rows: '; '.join(   # noqa: E731
        f'{a} {ch}, {f(rate05(a, ch)["rate"], 6)} ({rate05(a, ch)["rejections"]:,} of {rate05(a, ch)["tests"]:,} tests) '
        f'at the {r["percentile"]:.1f}th percentile, against a central 99% of {f(r["perm_lo"], 6)} to {f(r["perm_hi"], 6)}'
        for a, ch, r in rows) or 'none'
    pct_old = lambda ch: [S['anchor'][a][ch]['0.05']['percentile'] for a in HAPMIX]   # noqa: E731
    pct_new = lambda ch: [FA[a, ch]['percentile'] for a in HAPMIX]   # noqa: E731
    rng = lambda v: f'{min(v):.1f}-{max(v):.1f}th'   # noqa: E731
    return f"""
<p>At 0.05, over the anchor and |beta| = 0.2 / 0.4 / 0.8: gibbs combined {N_('gibbs')} and total
{N_('gibbs', 'total')}, with gene-clustered intervals above 0.05 throughout; split combined {N_('split')};
unit combined {N_('unit')} and allelic {N_('unit', 'allelic')}; plus_one combined {N_('plus_one')};
mixQTL combined {N_('mixqtl')} (published) and {N_('mixqtl_permissive')} (permissive). For the
four hapmixQTL arms the pattern is that of the stored null runs, whose 200-permutation rates are in the anchor table;
no stored null run exists for mixQTL here.</p>
<p><b>Rates at |beta| &gt; 0.</b> Thinning is expected to dilute the real data's coupling between weights and
residuals and so to pull these rates toward nominal. They are compared here with the stored null re-run under commit
8a06803, whose references are the ones these arms use. At 0.05 the dilution is not seen for gibbs: its total-channel
rates ({Rb('gibbs', 'total', '0.05')}) and allelic rates ({Rb('gibbs', 'allelic', '0.05')}) lie within {f(dmax, 4)} of
that run's means ({svf('gibbs', 'total', '0.05')} and {svf('gibbs', 'allelic', '0.05')}), inside their intervals. At
0.001 {low_txt}, and every interval includes that run's value (gibbs at |beta| 0.4,
{ci(S['null']['beta0.4']['gibbs']['allelic']['all']['0.001'], 'rate', 4)}); gibbs's total channel is not lower
({Rb('gibbs', 'total', '0.001')} against {svf('gibbs', 'total', '0.001')}). The dilution is therefore not resolved
here, and the |beta| &gt; 0 rates are not used as calibration results.</p>
<p><b>The tail, before and after commit 8a06803.</b> Before the commit every hapmixQTL p was referred to t with
N &minus; 2 &minus; 17 = 73 degrees of freedom, and the stored 200-permutation combined rates at 0.001 were
{f(tb[0], 4)} for gibbs and {' / '.join(f(x, 4) for x in tb[1:])} for split / unit / plus_one, {f(min(tb) / 0.001, 1)}
to {f(max(tb) / 0.001, 1)} times nominal. Most of the excess of split, unit and plus_one came from one null gene,
{g1}: its allelic channel has two donors, so its through-origin allelic fit has one residual degree of freedom, and its
allelic p was computed as if it had 73. On this benchmark's anchor, scored before the commit, the gene held
{held(SB, 'split')} combined rejections at 0.001, {held(SB, 'unit')}, {held(SB, 'plus_one')} and {held(SB, 'gibbs')}.
The commit refers each channel's p to its own degrees of freedom and the combined p to the Welch-Satterthwaite degrees
of freedom (section 2), and {g1}, below the {MIN_ALLELIC_DONORS}-donor floor, is now tested on its total channel alone. On the anchor
it now holds {held(S, 'split')}, {held(S, 'unit')}, {held(S, 'plus_one')} and {held(S, 'gibbs')}, the last from
gibbs's own total channel. On the stored null re-run under the commit ({DF_FIX.parent.name}: the same
{FX['n_draw']} permutations and genes, with slopes and standard errors unchanged), {g1}'s split combined rate at 0.001
went from {f(g1r('split', 'before'), 4)} to {f(g1r('split', 'after'), 4)}, and the pooled combined rates at 0.001 are
{ci(after('gibbs'), 'rate', 4)} for gibbs and {ci(after('split'), 'rate', 5)} / {ci(after('unit'), 'rate', 5)} /
{ci(after('plus_one'), 'rate', 5)} for split / unit / plus_one.</p>
<p>Those three intervals exclude 0.001, so the rule stated before the re-run (split, unit and plus_one within their
gene-clustered intervals of 0.001, {g1} included) failed, for two reasons the floor does not touch. First, the
Welch-Satterthwaite reference is more liberal than 73 degrees of freedom in admitted genes: in the {n40} genes with at
least 40 allelic donors its median over their tested pairs is {f(min(dof40), 0)} to {f(max(dof40), 0)} by arm (the
re-run's first permutation), so the same statistic gets a
smaller p, and their combined rate at 0.001 rose, paired on the same resampled genes, by {ci(dif('split'), 'diff', 5)}
(split), {ci(dif('unit'), 'diff', 5)} (unit) and {ci(dif('plus_one'), 'diff', 5)} (plus_one). That reference treats the
channel weights as fixed although they are estimated from the same residuals, which makes it anticonservative
(docs/hapmixqtl_methods.md, Section 4.5, measures this under the exact model); Meier's correction, which this run's
combined statistic carries (section 2), addresses that part, and the stored re-run predates it. Second, the unit-weighted total channel, which the commit does not change, is itself at
{ci(after('split', 'total'), 'rate', 5)} at 0.001. The three arms' combined rates lie within {f(max(ta) - min(ta), 5)}
of each other, so the tail does not separate them. gibbs's combined rate is {f(after('gibbs')['rate'] / 0.001, 1)}
times nominal, with its total channel at {f(after('gibbs', 'total')['rate'], 4)} (unchanged by the commit). The
mixQTL arms, RASQUAL and TReCASE leave {g1}'s allelic channel out by their own rules (mixQTL fits an allelic channel
only with more than two donors and combines its channels only with at least 15 in each; RASQUAL did not admit its
pseudo feature SNP, section 2; asSeq requires five
allele-specific reads per record and five heterozygous donors), and since the commit so does hapmixQTL's combined
statistic, so the gene no longer has to be left out of a tail comparison across methods: on the anchor at 0.001 split
reads {ci(S['null']['beta0.0']['split']['combined']['all']['0.001'], 'rate', 5)}, RASQUAL
{ci(S['null']['beta0.0']['rasqual']['combined']['all']['0.001'], 'rate', 5)} and TReCASE
{ci(S['null']['beta0.0']['trecase']['combined']['all']['0.001'], 'rate', 5)}.</p>
<p><b>The anchor.</b> Its one permutation sits at the {rng(pct_old('total'))} percentile of the stored per-permutation
total-channel rates in the four arms, like for like, since the commit does not change the total channel. For the
combined and allelic channels 06_score.py's reference is the stored runs made before the commit, whose p used the 73-df
reference; against it this dataset sits at the {rng(pct_old('allelic'))} percentile of the allelic rates and outside
the central 99% for: {outside(old)}. Like for like, against the per-permutation rates of the stored null re-run under
the commit, it sits at the {rng(pct_new('allelic'))} percentile of the allelic rates and the
{rng(pct_new('combined'))} of the combined ones (that re-run predates Meier's correction, which this run's combined
statistic carries, so in the combined channel the comparison is no longer like for like), and outside the central 99%
for: {outside(new)}. This says where one
permutation fell. The plumbing was checked exactly by check (d) described in section 2; that check is committed in
01_check_inputs.py and is in the check file used here.</p>"""


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
    rows = [(a, ch) for a in ('gibbs', 'split', 'plus_one') for ch in CHANNELS if (a, ch) != ('split', 'total')]
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
    high = ('gibbs', 'mixqtl', 'mixqtl_permissive', 'rasqual', 'trecase')   # the arms whose eigenMT calls carry a high null share (review 2026-09-27)
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
{REF_SET} is this code's run ({REF_RUN}) on the datasets of the first plasmode run ({FIRST_RUN.name}), with Meier's
correction as here: the same benchmark on the {INTERPRETED_SET} genes, {SF["ref_above"]} of
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
reference for mixQTL's meta statistic, a chi-squared likelihood-ratio reference for RASQUAL and TReCASE). The
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
                 f"RASQUAL and TReCASE paragraphs). mixQTL with published cutoffs is left out of the figure: its anchor ratio "
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
permutation scan, which has no Beta approximation; RASQUAL and TReCASE have none here. The second, for every arm, is
the <b>eigenMT</b> p (Davis et al. 2016), which needs no permutation: a gene's effective number of independent tests,
M<sub>eff</sub>, is the number of eigenvalues of its tested variants' genotype correlation matrix (Ledoit-Wolf shrunk:
the sample correlation pulled toward the identity by a weight estimated from the data; in windows of 200 consecutive
variants) needed to explain 99% of their variance, and the gene-level p is min(1,
M<sub>eff</sub> x the gene's smallest nominal p), a Bonferroni correction over M<sub>eff</sub> tests that inherits any
miscalibration of the arm's nominal p. With 92 donors in 200-variant windows the shrinkage, not linkage disequilibrium,
sets M<sub>eff</sub>, so for an arm whose nominal p is not anticonservative the eigenMT p is conservative relative to
the permutation p (section 2 measures by how much on this set). <b>Benjamini-Hochberg at 5%</b> keeps the k smallest gene-level p values of a
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
                 f'permutation p (every arm but RASQUAL and TReCASE; gene-clustered 95% intervals). Fourth row: the eigenMT p '
                 f'of every arm held to a common error rate, power at 5% realized false-discovery proportion as in the second '
                 f"row (no interval). Bottom: the realized false-discovery proportion of each arm's Benjamini-Hochberg calls at "
                 f'5% on the eigenMT p (dashed line: 0.05); an arm above the line calls null genes beyond the 5% the procedure '
                 f"promises, so its Benjamini-Hochberg power on this p (table) is not comparable with the others'. The "
                 f'statistics are defined in the paragraph above the tables.')}
{joint}
{limit}
{settled}'''


def interp_joint(part):
    """The RASQUAL and TReCASE paragraph of one results section."""
    Bb = lambda a, bn='all': per_beta(lambda b: bias(b, a, 'combined', 'bias_count', bn)['mean'], 2)   # noqa: E731
    Zj = lambda a: per_beta(lambda b: prec(f'beta{b}', a, 'combined', 'nonnull', 'sd_z')['value'], 2)   # noqa: E731
    n0 = lambda a, al, g='all': S['null']['beta0.0'][a]['combined'][g][al]   # noqa: E731
    tr = [S['null'][sc]['trecase']['combined']['all']['0.05']['rate'] / 0.05 for sc in S['null']]
    zx = lambda a: S['precision']['beta0.0'][a]['combined']['null_excluded']['nonfinite_z']   # noqa: E731
    zn = lambda a: prec('beta0.0', a, 'combined', 'null', 'sd_z')['units']   # noqa: E731
    raise_pct = lambda a: f'{100 * ((zn(a) + zx(a) - 1) / (zn(a) - 1)) ** 0.5 - 100:.1f}%'   # noqa: E731
    hm = [bias(b, a, 'combined', 'bias_count')['mean'] for b in BETAS for a in HAPMIX]
    exc = lambda b, a: ci(prec(f'beta{b}', a, 'combined', 'nonnull', 'ratio_vs_unit_count'), 'value', 2)   # noqa: E731
    dA = lambda a, b0: per_beta(lambda b: auc(b, b0)['mean'] - auc(b, a)['mean'])   # noqa: E731
    dD = max(abs(S['detection'][f'beta{b}']['trecase']['combined']['all']['0.001']
                 - S['detection'][f'beta{b}']['split']['combined']['all']['0.001']) for b in BETAS)
    comp = {k: v['all']['0.05'] for k, v in S['trecase_components'].items()}
    name = dict(trec='total-count (TReC)', joint='joint', ase='allele-specific (ASE)')
    fin = n0('trecase', '0.05')['rate']
    if any(v['lo'] <= 0.05 for v in comp.values()) or fin <= max(v['rate'] for v in comp.values()):
        raise SystemExit(f'{C.SUMMARY}: TReCASE components {comp} against final {fin}; reword section 3.7')
    text = dict(
        ranking=f"""
<p><b>The joint models.</b> RASQUAL's AUC is {A_('rasqual')} and TReCASE's {A_('trecase')} at |beta| = 0.2 / 0.4 /
0.8, against {A_('split')} for split and {A_('unit')} for unit; their power at 5% realized FDP is {P_('rasqual')} and
{P_('trecase')}, against {P_('split')} and {P_('unit')}. Both are below split in point estimate on both measures at
every |beta|. TReCASE's AUC is below unit's by {dA('trecase', 'unit')} and RASQUAL's by {dA('rasqual', 'unit')}. The
ranges of per-dataset AUCs overlap for TReCASE and unit at {at_betas(overlap('trecase', 'unit'))}, for TReCASE and
split at {at_betas(overlap('trecase', 'split'))}, and for RASQUAL and split at
{at_betas(overlap('rasqual', 'split'))}; at |beta| 0.8 split's lowest dataset AUC ({f(auc('0.8', 'split')['lo'])})
exceeds RASQUAL's highest ({f(auc('0.8', 'rasqual')['hi'])}). With three datasets per effect size, the three
agreeing in direction is the most the data can show (a sign test on three datasets cannot fall below p = 0.25,
two-sided), and power at realized FDP has no interval. At |beta| 0.2 their pooled cuts fell at lead p of
{thr('0.2', 'rasqual')} (RASQUAL) and {thr('0.2', 'trecase')} (TReCASE), against {thr('0.2', 'split')} for split: null
genes' lead p reached below split's cut, as for gibbs. For TReCASE that agrees with its null-gene rates; RASQUAL's
anchor rate is {f(n0('rasqual', '0.05')['rate'], 4)} at 0.05 but {f(n0('rasqual', '0.001')['rate'], 4)} at 0.001
(section 3.7).</p>""",
        bias=f"""
<p><b>The joint models.</b> Their one slope has estimand beta (section 2), so unlike the hapmixQTL combined slope its
ratio to beta measures estimator bias. RASQUAL recovers {Bb('rasqual')} of beta and TReCASE {Bb('trecase')} at
|beta| = 0.2 / 0.4 / 0.8; by read band at |beta| 0.8 (&lt;100 / 100-999 / &ge;1000) that is
{' / '.join(f(bias('0.8', 'rasqual', 'combined', 'bias_count', bn)['mean'], 2) for bn in BANDS[1:])} and
{' / '.join(f(bias('0.8', 'trecase', 'combined', 'bias_count', bn)['mean'], 2) for bn in BANDS[1:])}. At |beta| 0.8
RASQUAL's interval is {ci(bias('0.8', 'rasqual', 'combined', 'bias_count'), 'mean')} and TReCASE's
{ci(bias('0.8', 'trecase', 'combined', 'bias_count'), 'mean')}. TReCASE's intervals include 1 at every |beta|,
the only combined slope here of which that is true (the hapmixQTL combined slopes, which mix two estimands and the
transforms' attenuation, read {f(min(hm), 2)} to {f(max(hm), 2)} against beta); RASQUAL's exclude 1 at every |beta|, the shortfall largest
below 100 reads. What causes RASQUAL's shortfall is not decomposed here. Two candidates are named from its source,
neither tested: (i) RASQUAL fits the covariates once, in a negative-binomial model under the null without the
genotype, and passes the fitted covariate effect as a fixed per-sample offset into every variant's fit
(rasqual_src/src/main.c:631 and :641-643; nbem.c:297 and :334). That is the same two-step structure whose
attenuation section 3.8 measures exactly for mixQTL, here inside a count likelihood and mixed with a covariate-free
allelic part. (ii) RASQUAL fits a reference-mapping bias phi below 0.5 at the pseudo feature SNP, where no mapping
bias can exist; a phi below 0.5 absorbs part of the allelic imbalance.</p>""",
        precision=f"""
<p><b>The joint models.</b> For RASQUAL and TReCASE the standard error is derived by the Wald inversion of
&chi;<sup>2</sup> (section 2). On null genes z = &plusmn;&radic;&chi;<sup>2</sup>, so sd(z)<sup>2</sup> is, up to the
mean of z (near 0), the mean of &chi;<sup>2</sup>: 1 when the statistic has the mean of a &chi;<sup>2</sup> with one
degree of freedom. It checks the statistic's mean, not its tail, which section 3.7 gives. On the anchor's null genes it
is {Zn_('rasqual', 'combined', 3)} for RASQUAL and {Zn_('trecase', 'combined', 3)} for TReCASE. So on the anchor TReCASE's statistic is larger on average
than a &chi;<sup>2</sup> with one degree of freedom, which agrees with its null-gene rate (section 3.7), while RASQUAL's
interval includes 1. Tests whose derived z is undefined are left out ({zx('rasqual'):,} and {zx('trecase'):,} on the
anchor): &chi;<sup>2</sup> &le; 0 ({JF['rasqual']['chisq_le0_anchor']:,} of RASQUAL's; for TReCASE, &chi;<sup>2</sup>
printed as 0.000 or missing) and, for RASQUAL only, &pi; printed as exactly 0.5, where slope and derived se are both 0.
These would enter as z = 0, so leaving them out raises sd(z) by about {raise_pct('rasqual')} and
{raise_pct('trecase')}. At the causal variant sd(z) is {Zj('rasqual')} and {Zj('trecase')}; there
z = (slope &minus; beta) / se describes how well the derived se matches the slope's spread around beta and absorbs
bias, so it is not a calibration of the test.</p>
<p>On squared error, with the arm and unit weights both held to the count-scale truth, RASQUAL's slope has
{Ex_('rasqual')} of unit weights' at |beta| = 0.2 / 0.4 / 0.8 and TReCASE's {Ex_('trecase')}, against {Ex_('split')} for
split; on the anchor's null genes, where the truth is 0 for every arm, the ratios are {En_('rasqual', 'combined')}, {En_('trecase', 'combined')}
and {En_('split', 'combined')}. Both joint slopes have more squared error than unit weights, with intervals above 1, at every |beta|
and on the anchor, with one exception: TReCASE at |beta| 0.8 ({exc('0.8', 'trecase')}), whose point estimate is also
below split's ({exc('0.8', 'split')}), with overlapping intervals. That is a comparison of squared error, not of
precision: on the count-scale truth unit weights' squared error contains their own bias (the next paragraph), which
grows with beta<sup>2</sup>, while on the anchor's null genes, which carry no bias, TReCASE's slope has {En_('trecase', 'combined')}
of unit weights' squared error. How much of the 0.8 value that bias accounts for was not separated.</p>""",
        lead=f"""
<p>RASQUAL's share of leads within r<sup>2</sup> &ge; 0.8 of the causal variant is {R_('rasqual')} and TReCASE's
{R_('trecase')}, against {R_('split')} for split and {R_('unit')} for unit, with no interval. TReCASE's shares are within
0.02 of unit's; RASQUAL's are the lowest of the nine arms at |beta| 0.2 and 0.4.</p>""",
        detection=f"""
<p>RASQUAL detects the causal variant at p &lt; 1e-3 in {D_('rasqual')} of non-null gene units and TReCASE in
{D_('trecase')}, against {D_('split')} for split and {D_('unit')} for unit; a causal unit without a row is left out of a
joint arm's share (section 2). TReCASE's shares are within {f(dD, 2)} of split's. On the anchor, though, TReCASE's test
rejects at 1e-3 in {ci(n0('trecase', '0.001'), 'rate', 4)} of null-gene tests and RASQUAL's in
{ci(n0('rasqual', '0.001'), 'rate', 4)} (section 3.7), so neither is comparable at face value with the hapmixQTL arms
above; RASQUAL's shares are below unit weights' at every |beta|.</p>""",
        null=f"""
<p><b>The joint models.</b> Their likelihood-ratio p, referred to &chi;<sup>2</sup> with one degree of freedom,
rejects at 0.05 on null genes in {N_('rasqual', 'combined', 4)} of tests for RASQUAL (over its converged rows) and {N_('trecase', 'combined', 4)} for
TReCASE (anchor, then |beta| = 0.2 / 0.4 / 0.8). On the anchor the gene-clustered intervals are
{ci(n0('rasqual', '0.05'), 'rate', 4)} and {ci(n0('trecase', '0.05'), 'rate', 4)} at 0.05,
{ci(n0('rasqual', '0.01'), 'rate', 4)} and {ci(n0('trecase', '0.01'), 'rate', 4)} at 0.01, and
{ci(n0('rasqual', '0.001'), 'rate', 5)} and {ci(n0('trecase', '0.001'), 'rate', 5)} at 0.001, that is
{f(n0('rasqual', '0.001')['rate'] / 0.001, 1)} and {f(n0('trecase', '0.001')['rate'] / 0.001, 1)} times nominal there.
TReCASE's intervals at 0.05 lie above 0.05 in every scenario, at {f(min(tr), 2)} to {f(max(tr), 2)} times nominal; RASQUAL's lies just
above 0.05 on the anchor and includes it at |beta| &gt; 0. Neither model fits {one_df_gene()}'s allelic channel
(above), and without that gene their anchor rates at 0.001 are {f(n0('rasqual', '0.001', NO_ONE_DF)['rate'], 5)}
and {f(n0('trecase', '0.001', NO_ONE_DF)['rate'], 5)}. No stored null run exists for them, so where the anchor's one
permutation falls among their permutations is not known; the three datasets at each |beta| &gt; 0 are three further
permutations, on half the genes and thinned.</p>
<p><b>Where TReCASE's excess comes from.</b> asSeq's final p is one of its component tests, chosen per test. Scored
alone on the anchor's null genes, over the tests where its p is finite, each component rejects at 0.05 in
{'; '.join(f'{name[k]} {ci(v, "rate", 4)} ({v["tests"]:,} tests)' for k, v in comp.items())}, against {f(fin, 4)} for
the final p. So each component is above nominal on its own. The final p rejects more often than any component does over
all its tests, so the choice between them, which falls back to the total-count test wherever the joint fit is missing
and wherever asSeq's cis/trans test rejects, adds to an excess the components already have; each component's rate on the
subset where asSeq uses it was not scored. One candidate for the components' own excess, untested: a likelihood-ratio
test that fits an intercept, 17 covariates and a dispersion to 92 donors can reject too often at this sample size, and
RASQUAL has the same design.</p>""")
    return text[part]


def sec_results(figs):
    interp = lambda fn: fn() if INTERPRETED else ''   # noqa: E731
    joint = lambda part: interp_joint(part) if INTERPRETED else ''   # noqa: E731
    genes = {bn: S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z'][bn]['genes'] for bn in BANDS}
    n_rep = S['n_datasets']['0.4']
    if INTERPRETED:
        off = [g for g, v in FX['below_floor_genes'].items() if v['gibbs']['allelic']['after']['n_tests'] == 0]
        n_old, n_new = (X['null']['beta0.0']['gibbs']['allelic']['all']['0.05']['tests'] for X in (SB, S))
        if len(off) != 1 or n_old - n_new != FX['below_floor_genes'][off[0]]['gibbs']['allelic']['before']['n_tests'] // FX['n_draw']:
            raise SystemExit(f'{DF_FIX}: allelic channel off in {off}, anchor allelic tests {n_old} -> {n_new}; reword section 3.4')
        exc_txt = f'''One exception since commit 8a06803: a gene whose allelic channel
is switched off altogether ({", ".join(off)}, one informative allelic donor) has a NaN allelic p and leaves the allelic
null rate ({n_old - n_new:,} tests on the anchor), where before it counted at p = 1; that alone raises an allelic null
rate by the factor {n_old:,} / {n_new:,} = {f(n_old / n_new, 4)}. The stored null re-run under that commit, against
which section 3.7 compares, uses the same convention.'''
        auc_txt = '''It is computed
per dataset and averaged over the 3 datasets. Its interval is the range of the three per-dataset AUCs: with
three datasets, the 2.5% and 97.5% quantiles of the mean over datasets resampled with replacement are the smallest
and largest dataset values. It is not a 95% interval, and it carries no gene-to-gene variation, because the three
datasets hold the same 100 genes.'''
        fig1_bar = 'range of the three per-dataset AUCs, not a 95% interval'
        anchor_txt = '''For the
four hapmixQTL arms its rate is compared with the stored 100-gene, 200-permutation null runs of the same arms, as
made before commit 8a06803 (06_score.py's reference) and as re-run under it (like for like except in the combined
channel, whose stored runs predate Meier's correction; below): the
percentile of this dataset's rate among the 200 stored per-permutation rates, and whether it lies inside their
central 99%. This is descriptive: one permutation cannot test the plumbing.'''
        anchor_tab = f'''<p>The anchor against the stored null runs (hapmixQTL arms only):</p>
{tab_anchor()}'''
    else:
        exc_txt = ('Since commit 8a06803 a gene whose allelic channel is switched off altogether (fewer than two '
                   'informative allelic donors) has a NaN allelic p and leaves the allelic null rate.')
        auc_txt = (f'It is computed per dataset and averaged over the {n_rep} datasets. Its interval is the 2.5% and 97.5% '
                   f'quantiles of the mean over datasets resampled with replacement ({n_rep} datasets); it carries no '
                   f'gene-to-gene variation, because every dataset holds the same {genes["all"]} genes.')
        fig1_bar = 'interval of the mean over datasets resampled with replacement'
        anchor_txt = (f'No stored 200-permutation null run exists for this gene set, so where its one permutation falls '
                      f'among permutations is not known here; it is the same permutation as the {REF_SET}\'s anchor, and '
                      f'the contrast section places it against that set\'s stored null.')
        anchor_tab = ''
    ladder = sec_ladder() if LD is not None else ''   # the ladder was run for the interpreted set only
    return f'''
<h2>3. Results</h2>
<p><b>How to read the intervals.</b> Unless stated, an interval is a <i>gene-clustered 95% interval</i>: the
genes are resampled with replacement {S["n_boot"]:,} times, each gene carrying all its units from the
scenario's datasets, the statistic is recomputed each time, and the 2.5% and 97.5% quantiles are reported. It
carries gene-to-gene spread, the main source of uncertainty when the same genes recur across datasets. The
read bands hold {" / ".join(str(genes[b]) for b in BANDS[1:])} of the {genes["all"]} genes. All arms were
run on the same datasets, so their errors are correlated, and the summary carries no interval for the
difference between two arms: a gap between arms is read against each arm's own interval. Separated intervals
are then good evidence of a difference; overlapping ones do not show that two arms are equal.</p>

<h3>3.1 Gene ranking: AUC and power at a realized false-discovery proportion</h3>
<p>Within each dataset the {genes["all"]} genes are ordered by the nominal p of their <i>lead variant</i> (the tested
variant with the smallest p; the combined statistic for hapmixQTL, the meta statistic for mixQTL, the one joint test
for RASQUAL and TReCASE). The
<b>AUC</b> (area under the receiver operating characteristic curve) is the probability that a randomly chosen
non-null gene ranks above a randomly chosen null gene: 0.5 is chance, 1 is perfect separation. {auc_txt} <b>Power at 5% realized
false-discovery proportion</b>: the gene units of the {n_rep} datasets are pooled and walked down the ranking; the
realized false-discovery proportion (FDP) at depth k is the share of the top k that are truly null, known here
from the truth; the walk stops at the deepest k with FDP &le; 0.05, and power is the share of non-null gene
units above that point. The summary gives it no interval. A within-dataset ranking uses no threshold, so a
miscalibration shared by every gene does not move it, but one specific to a gene does. Since commit 8a06803 each
hapmixQTL pair has its own t reference (section 2), so ranking hapmixQTL genes by lead p is no longer the same as
ranking them by |t|. The ranking is also confounded by the number of tested variants per gene (a null gene with many
variants has a smaller lead p by chance), a confounding every arm shares.</p>
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
     + " and tensorQTL, mixQTL's own pval_perm; RASQUAL and "
     + ('the two TReCASE arms' if NATIVE else 'TReCASE')
     + " have none; gene-clustered intervals). D: the eigenMT p of every "
     'arm held to a common error rate, the share of non-null gene units called at the deepest point of the pooled '
     'ranking by that p where at most 5% of calls are null genes (as B; no interval). E: the realized '
     "false-discovery proportion of each arm's Benjamini-Hochberg calls at 5% on the eigenMT p, null gene units "
     'among the units called (dashed line: 0.05); an arm above the line calls null genes beyond the 5% the procedure '
     'promises, which is why its Benjamini-Hochberg power on this p (table below) is not comparable with the '
     "others'. Points are offset sideways within each |beta| so that intervals do not overlap.")}

<h3>3.2 Gene-level power</h3>
<p>map_cis gives each gene a gene-level p, <b>pval_beta</b>: the lead variant's nominal p is compared with the
smallest p of each of 1,000 permutations of donor records (with L/R swaps) under the arm's own weights, and
the comparison is smoothed by fitting a Beta distribution to those permuted minima. Because each arm is
referred to its own permutation null, pval_beta absorbs whatever miscalibration of that arm's nominal p the
permutation reproduces. Total-only tensorQTL's pval_beta is the same construction on its own null (its
covariate-residualized phenotype permuted), and mixQTL's gene-level p is the empirical p of its own permutation
scan, without the Beta smoothing (section 2). The eigenMT p (section 2) is a second gene-level p for every arm, the
joint models included; it needs no permutation. The <b>Benjamini-Hochberg</b> procedure at 5% then calls genes within each dataset:
the {genes["all"]} gene-level p are sorted and the k smallest are called, k being the largest rank with
p<sub>(k)</sub> &le; 0.05 k / {genes["all"]}; with valid p values the expected share of null genes among the calls is at
most 5%. Power is the share of non-null gene units called. The last four columns count null gene units with
gene-level p below 0.05, permutation p / eigenMT p (a gene-level false-positive rate before any multiple-testing
correction).</p>
{interp(interp_gene_level)}
{tab_gene_level()}

<h3>3.3 Bias at the causal variant</h3>
<p>The <b>bias ratio</b> is the mean, over causal units, of the estimated slope divided by the true slope at
the causal variant: 1 means the injected effect is recovered on average, 0.9 means 90% of it. Two truths are
used. On the <i>count scale</i> the allelic truth is beta itself, and the total truth is the least-squares
slope, with intercept, of the exact log2 total fold on half the ALT dosage (g/2) over the dataset's 92 donors
(near beta but not equal to it, because genotype counts are asymmetric). On the <i>pipeline scale</i> (hapmixQTL
arms only) the truth is the
slope the pipeline's own transformed phenotypes would show with no noise, after the +0.5 pseudocount of
log2((L + 0.5)/(R + 0.5)) and the +1 of log2(CPM + 1), which compress a fold at low depth. It is an unweighted
slope, so a weighted arm's bias against it still contains how the weights re-target a shift that varies with
depth. mixQTL's response has neither pseudocount nor +1, so its count-scale truth is already its own scale.
The combined slope mixes the two channels' estimands and is reported against beta only. Units whose ratio is
not finite are excluded and counted. A unit whose allelic channel has no admitted heterozygous donor returns
slope 0 with an infinite standard error; its count-scale ratio 0 / beta is finite, so it enters the count-scale
mean as 0, while its pipeline-scale truth is undefined and it is excluded there. In the table the first line is
the count scale, the second (grey) the pipeline scale, each followed by (units / excluded). The figure draws every
arm against the count-scale truth; the table's grey line carries the pipeline-scale diagnostic for hapmixQTL.</p>
{interp(interp_bias)}
{joint('bias')}
{tab_bias()}
{img(figs['bias'], 'Figure 2. Bias ratio (mean slope / truth at the causal variant, gene-clustered interval) by '
     'arm and read band, every arm against the count-scale truth. Top two rows: the allelic and total channels of the '
     "arms that have them (hapmixQTL, mixQTL; tensorQTL's one slope is drawn in the total row). Bottom row: every "
     "arm's one combined slope, the row where RASQUAL and TReCASE appear, each fitting one joint effect for both kinds "
     "of count, beside hapmixQTL's combined slope, which on this truth carries the log2(CPM + 1) attenuation of its "
     'total channel (section 3.3). The hapmixQTL allelic points keep as zeros the units with no allelic data, which '
     'the mixQTL points drop (section 3.3). mixQTL channels: allelic = asc, total = trc. Colour shade = |beta|. The '
     'y axes differ between panels.')}

<h3>3.4 Precision: stated standard error and efficiency against unit weights</h3>
<p><b>Realized over stated standard error</b>, written sd(z): z = (slope &minus; truth) / stated se, and sd(z) is
its standard deviation over units. It is 1 when the stated se equals the realized spread of the slope, 1.2
when the realized spread is 20% larger than the se says (se too small, p too small), and below 1 when the se is
too large. This is the reciprocal of a stated-over-true ratio; it is reported as 06_score.py computes it.
Non-null units use the causal variant and the pipeline-scale truth (mixQTL: count scale; RASQUAL and TReCASE: beta;
combined: the same inverse-variance combination of the two channel truths); in the allelic channel that is the
{S['precision']['beta0.4']['gibbs']['allelic']['nonnull']['sd_z']['all']['units']} of
{S['recovery']['beta0.4']['gibbs']['allelic']['bias_count']['all']['units']} causal units per |beta| that have
allelic data (section 3.3). Null units use every tested variant of the null genes, with truth 0. In the allelic
channel these include variants of null genes with no admitted heterozygous donor, whose output is p = 1 and
z = 0: they cannot reject and enter sd(z) as zeros, which lowers both the allelic null rate (section 3.7) and the
allelic null sd(z). The summary does not count them. {exc_txt} The <b>mean squared error ratio against unit weights</b>
(efficiency) is the sum of squared
errors under the arm divided by the same sum under unit weights, over the same units; below 1 means more
precise than unit weights. split and unit share the total channel's weights, so split's total-channel ratio is
1 by construction. Real data have no oracle variance, so this is efficiency relative to unit weights, never
against the best possible weights. For mixQTL the ratio is a comparison of methods as run: mixQTL admits a
different donor set (its count cutoffs; under the published allelic cap of 1,000 reads admission even depends
on the injected effect, because thinning pulls records down into the band), so its ratio mixes weighting with
admission. For mixQTL, RASQUAL and TReCASE the ratio at the causal variant holds the arm and unit weights both to the
count-scale truth, so no method is ranked on the pipeline scale.</p>
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
and the per-gene total truth; for tensorQTL, the per-gene total truth; for RASQUAL and TReCASE, beta).</p>
{tab_cross()}
{cross_note()}
{img(figs['efficiency'], 'Figure 3. Mean squared error ratio against unit weights (log scale; below 1 = more '
     'precise than unit weights), gene-clustered interval. Top: causal variant of non-null genes. Bottom: every '
     'tested variant of null genes ("anchor" is the beta = 0 dataset). Allelic and total panels: hapmixQTL arms '
     '(top, pipeline-scale truth for arm and unit alike) and mixQTL (bottom only). Combined panels: every arm, '
     'RASQUAL and TReCASE included, and at the causal variant the count-scale truth for arm and unit alike, so the '
     'top combined panel is the cross-method comparison and its hapmixQTL points differ from the pipeline-scale '
     'values in the text. unit is 1 by definition and not drawn; the total channel of split is 1 by construction. '
     'mixQTL with published cutoffs is left out of the figure (its ratios are in the tables above; on the anchor '
     f"{prec('beta0.0', 'mixqtl', 'combined', 'null', 'ratio_vs_unit')['value']:.0f}x unit weights', which would "
     'compress every other arm onto one line). The y range of each panel covers every plotted interval.')}

<h3>3.5 Lead-variant recovery</h3>
<p>For each non-null gene unit the lead variant is compared with the causal one. <b>LD r<sup>2</sup></b> is the
squared Pearson correlation (the ordinary correlation coefficient) of ALT allele dosages over the 92 donors between the two variants (1 when they are
the same variant). Reported: the share of units whose lead is the causal variant, the share with
r<sup>2</sup> &ge; 0.8, and the median r<sup>2</sup>. A unit without a finite p counts as not recovered. No
interval is given in the summary; each share is over {S['lead']['beta0.4']['gibbs']['all']['units']} gene units per |beta|.</p>
{interp(interp_lead)}
{joint('lead')}
{tab_lead()}
<p>Share with r<sup>2</sup> &ge; 0.8 by read band:</p>
{tab_bands(lambda b, a: S['lead'][f'beta{b}'][a], 'r2_high')}
{img(figs['lead'], 'Figure 4. Share of non-null gene units whose lead variant is in LD r^2 >= 0.8 with the '
     'causal variant, by arm, |beta| and read band. No interval.')}

<h3>3.6 Causal-variant detection</h3>
<p>The share of non-null gene units whose nominal p at the causal variant falls below 0.05, 1e-3 and 1e-5, per
channel. This is power at a fixed nominal threshold, so it rewards an arm whose p values are too small; read it
together with the null rates in 3.7.</p>
{interp(interp_detection)}
{joint('detection')}
{tab_detection()}

<h3>3.7 Null genes and the beta = 0 anchor</h3>
<p>The <b>null-gene rate</b> is the share of tested variants of null genes whose nominal p is below a threshold
(0.05 in the first table). In the allelic channel it includes the tests with no allelic data (p = 1; section 3.4),
which cannot reject. At |beta| &gt; 0 the null genes are thinned too, which adds binomial noise of the model's own
kind and is expected to dilute the real data's coupling between weights and residuals, so those rates are not
calibration results. The beta = 0 anchor is one dataset, that is ONE record permutation, with no thinning. {anchor_txt}</p>
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


def sec_native():
    """The native-input arms of 05b_native_arms.py beside their Salmon-input counterparts: the inputs, the donors each
    admits, the ranking, the calibration and the limits, the main table, then TReCASE's component statistics in one table
    after it. Empty where 06 scored no native arms (no C.NATIVE)."""
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
<p>RASQUAL and TReCASE are written for integer alignment counts, and every arm above reads Salmon's point estimates. Two
further arms read counts from the same STAR alignments instead ({C.NATIVE_COUNTS}, scripts/native_counts.py): the total
is featureCounts' count of fragments on the gene's exons (primary, uniquely mapped, reverse-stranded read pairs; a fragment
on exons of two genes is not counted), and the allele-specific counts a and b are phASER's counts of fragments over
heterozygous SNPs on the haplotypes carrying the analysis VCF's first and second allele (counted per transcript strand
on strand-split BAMs, at SNVs in exon stretches that one gene owns on its strand, the GTEx-style collapsed gene model;
HLA genes and CHM13 short-read-inaccessible regions excluded; reads WASP-filtered for allele-dependent mapping, scripts/phaser_wasp.py; scripts/phaser_stranded.py), set to a = b = 0 where a + b
exceeds the total ({gs['negative']:,} of this gene set's {gs['pairs_with_reads']:,} donor-gene pairs with phASER reads,
holding {gs['allelic_fragments_in_negative']:,} of its {gs['allelic_fragments']:,} allele-specific fragments). Each
dataset's record permutation, label swaps and thinning factors are applied to them (05b_native_arms.py; exact binomial
thinning of integers in 02's order), so the null genes, causal variants, injected effects and truth are the Salmon-input
arms'. The count-scale total truth depends only on the causal genotypes and beta, so it is the truth of both inputs; no
pipeline-scale truth is computed for the native arms, whose standard-error and squared-error figures in sections 3.3 and
3.4 use the count-scale truth. The effective library size is recomputed from the native totals by the Salmon run's edgeR
rule (filterByExpr, edgeR's filter keeping genes with enough counts in enough samples; protein-coding autosomal genes;
TMM, the trimmed mean of log expression ratios against a reference sample, as the scale factor; {F['library']['genes_kept']:,}
genes kept) and is {f(lib_lo, 2)} to {f(lib_hi, 2)} times the Salmon one across donors (median {f(lib_med, 2)}).
<b>{SHORT['trecase_native']}</b> is the same TReCASE runner on the native total and on a and b as they are (no
zero-haplotype rule; asSeq's own floor of five allele-specific reads applies), with the log native effective library size
as offset and the same 17 covariates. <b>{SHORT['split_native']}</b> is split weighting on the same counts, the control
that separates the model from the quantifier: the allelic log2((a + 0.5)/(b + 0.5)) weighted by one over its counting
variance (1/(a + 0.5) + 1/(b + 0.5))/ln2<sup>2</sup>, the total log2(CPM + 1) unweighted.</p>
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
<p><b>What the native arms cannot show.</b> The 14 RNA-tied covariates are unchanged, and their expression
principal components were computed from Salmon's log2 CPM. The two inputs admit different donors to the allelic channel
(above: phASER needs a fragment over a heterozygous SNP, and only the Salmon-input arms apply the zero-haplotype rule);
featureCounts drops multimapping fragments and fragments on two genes' exons{f", which leaves {', '.join(notrun)} with a total of 0 in every donor and no allele-specific counts: asSeq stops on a constant total and tensorQTL's input generator drops a constant phenotype, so neither native arm tests it and both rank it last in every dataset" if notrun else ''}.
The inputs therefore differ in which reads are counted, not only in integer against fractional values, and a difference
between a native and a Salmon-input arm combines those differences. The permutation, thinning, truth and design are
shared with the Salmon-input arms, so this compares inputs to the same models, not TReCASE's or RASQUAL's own read
pipelines, and RASQUAL was not run on native counts; {SHORT['split_native']} against {SHORT['trecase_native']} is the
comparison in which both methods see the same counts. Each |beta| rests on {S['n_datasets']['0.4']} datasets, and power at
realized FDP has no interval.</p>
{t1}
<p><b>TReCASE's component tests</b> (second table): the share of tests whose joint fit is missing (asSeq's final p is
then its total-count test), the same at each gene unit's reported lead, the power at 5% realized FDP when genes are
ranked by the lead p of one component test alone (06_score.py, trecase_parts), and each component's null-gene rate on
the anchor over the tests where its p is finite. TReCASE's joint fit is missing at the reported lead in
{f(both('trecase'), 2)} of the gene units at |beta| &gt; 0 on Salmon's inputs and {f(both('trecase_native'), 2)} on
native counts, and in {f(tests['trecase']['joint_na_share'], 2)} and {f(tests['trecase_native']['joint_na_share'], 2)}
of all tests. Its total-count test alone reaches {per_beta(lambda b: tp(b, 'trecase')['fdp_power']['trec'])} power at
5% realized FDP on Salmon's inputs and {per_beta(lambda b: tp(b, 'trecase_native')['fdp_power']['trec'])} on native
counts, against tensorQTL's {P_(TQ)} on Salmon's totals.</p>
{t2}'''


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
weights. A separate run (benchmark/plasmode/07_mixqtl_ladder.py, output {C.LADDER / "ladder.json"}) went from the unit-weight arm to mixQTL one
change at a time, in the total channel, on the causal units where every step has a finite error: {per_beta(lambda b:
cu[f"beta{b}"]["total"], 0)} of {cu["beta0.4"]["non_null"]} non-null units at |beta| = 0.2 / 0.4 / 0.8 (the <i>common
set</i>). Every number in this section is read from that file.</p>
<p><b>mixQTL fits its total channel in two steps.</b> First the <i>covariate offset</i>: the natural-log total,
log(total reads / 2 / library size), is regressed on the 17 covariates without the genotype; the covariates whose t
statistic exceeds 2 in absolute value are kept (the <i>selected covariates</i>, median {"-".join(map(str, nsel))} of 17
per gene and dataset) and the regression is refitted on them. Second, the offset is subtracted from the response and
the result is regressed on x = (h1 + h2)/2, half the ALT dosage, without first removing from x the part that the
selected covariates explain (mixQTL's own R code: the offset in rlib_covariate.R:27-40, the genotype regression on
the residual in rlib_matrix_ls.R:26-46; ported as covariate_offset and trc_channel). The <b>Frisch-Waugh-Lovell identity</b> states that in a least-squares fit of y on an
intercept, x and covariates C, the slope on x equals the slope of y on x after x is replaced by its residual from a
regression on the intercept and C. It follows that when both steps use the same donors, mixQTL's slope is
(1 &minus; R<sup>2</sup>) times the <i>one-step</i> slope, the slope on x when y is regressed on the intercept, x and
the selected covariates together, where R<sup>2</sup> is the share of the variance of x over donors that the selected
covariates explain. The ladder checked this where the offset's donors (total reads above 0) and the total channel's
donors (total reads at the cutoff) are the same: the largest relative difference between mixQTL's slope and
(1 &minus; R<sup>2</sup>) times the one-step slope was {idn["max_rel_dev"]:.1e} over {idn["units"]} non-null units. The
attenuation is therefore arithmetic, not noise.</p>
<p>Mean slope over the count-scale total truth at the causal variant (gene-clustered interval), common set. The
one-step slope with all 17 covariates is fitted without any selection on the outcome.</p>
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
permissive ones. The chance level is the same quantity at the null genes' causal variants, where selection sees no
genotype effect but the genotype principal components among the covariates can still correlate with x:
{N2("published")} (published) and {N2("permissive")} (permissive). The non-null mean lies {above("published")} the null
interval (published) and {above("permissive")} it (permissive) at the three effect sizes.</p>
<p><b>Squared error, one change at a time.</b> Each column is the summed squared error at the causal variant over that
of the unit-weight arm, on the common set (gene-clustered interval). The <i>cutoff rung</i> is the unit-weight arm
(unit weights, log2(CPM + 1), all 17 covariates in one fit) with only the donors mixQTL's count cutoffs admit. The next
step changes the response to mixQTL's natural-log total over 2 x library size and the truth to the count scale,
together; then the covariates become the selected ones; then the fit becomes mixQTL's two steps. As the ladder defines
it, the unit-weight arm's squared error in the denominator is against its pipeline-scale truth, so these ratios are not
the count-scale cross-method ratios of section 3.4. All columns share that one denominator, but the step to mixQTL's
response also moves the numerator from the pipeline-scale to the count-scale truth: it mixes a change of response
with a change of the truth it is measured against, so the ladder as built cannot assign it a cost. The other steps
each hold the truth fixed and compare with each other.</p>
{t2}
<p>Donor admission alone gives {per_beta(lambda b: rung("published", b)["value"], 2)} at the published cutoffs and
{per_beta(lambda b: rung("permissive", b)["value"], 2)} at the permissive ones, which admit nearly every donor. The step to
mixQTL's response, which also changes the truth, gives {Q("published", "one_step_all_trc")} published and
{Q("permissive", "one_step_all_trc")} permissive; for the reason just given it is not read as a cost. On bias the change
of response runs the other way: the one-step fit with all 17 covariates on mixQTL's response recovers {f(lo17, 2)} to
{f(hi17, 2)} of the count-scale total truth (first table), where unit weights' total slope on log2(CPM + 1) recovers
{ub} (section 3.3). Selecting the covariates on the outcome gives {Q("published", "one_step_trc")} and
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
    gl = lambda a: per_beta(lambda b: bh(b, a)['power_bh']['all']['rate'])   # noqa: E731
    an = S['anchor']['gibbs']['total']['0.05']
    tg, ts = (fdp('0.4', a)['p_threshold'] for a in ('gibbs', 'split'))
    n3 = lambda a: S['null']['beta0.0'][a]['combined']['all']['0.001']['rate']   # noqa: E731
    cc = [prec(f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit_count') for b in BETAS]
    inc = [b for b, d in zip(BETAS, cc) if d['lo'] <= 1 <= d['hi']]
    En_band = lambda a, ch: ' / '.join(f(prec('beta0.0', a, ch, 'null', 'ratio_vs_unit', bn)['value'], 2) for bn in BANDS[1:])   # noqa: E731
    g02 = fdp('0.2', 'gibbs')
    low02 = ('made no call at all at |beta| 0.2' if g02['p_threshold'] is None else
             f'called {f(g02["all"]["power"])} of non-null units at |beta| 0.2, its cut at lead p {thr("0.2", "gibbs")}')
    return f"""
<h2>4. The strongest critique, and what it changed</h2>
<p><b>The ranking is not free of calibration.</b> A within-dataset ranking uses no threshold, but the cut at 5%
realized FDP is set by where the null genes land, and an arm whose null genes get too-small p pushes them up its
ranking. gibbs's total channel does this: its null-gene rate at 0.05 is {f(an['rate'], 4)} on the anchor and
{f(an['stored'], 4)} over the stored 200 permutations. In the ranking of section 3.1 gibbs {low02}, and at |beta| 0.4
its cut fell at lead p {tg:.1e} against split's {ts:.1e}, {ts / tg:.0f}-fold smaller.
If those null p values are too small, part of gibbs's ranking deficit is a calibration effect and not a lack of
signal. The same objection applies to
the joint models: at |beta| 0.2 their cuts fell at lead p {thr('0.2', 'rasqual')} (RASQUAL) and
{thr('0.2', 'trecase')} (TReCASE) against split's {thr('0.2', 'split')}, and on the anchor their nominal p rejects at
0.001 in {f(n3('rasqual'), 4)} and {f(n3('trecase'), 4)} of null-gene tests (section 3.7), so part of their ranking
deficit may also be calibration. Before commit 8a06803 the hapmixQTL arms carried a null outlier of their own,
{one_df_gene()}, whose allelic p was referred to 73 degrees of freedom on a one-degree-of-freedom fit; the allelic
admission floor took it out of their combined statistic, and section 3.1 gives the ranking before and after. In gibbs
the gene still rejects too often through its Gibbs-weighted total channel (section 3.7).</p>
<p><b>What addressing it changed.</b> Gene-level Benjamini-Hochberg on pval_beta changes three things at once: each arm
is referred to its own permutation null; pval_beta also corrects for the number of tested variants per gene (the
confounder named in section 3.1); and calls are made per dataset, not at a pooled realized-FDP cut. On it gibbs reads
{gl('gibbs')} against split's {gl('split')} at |beta| = 0.2 / 0.4 / 0.8, with overlapping intervals, so the ranking gap
(power at 5% realized FDP {P_('gibbs')} against {P_('split')}, no interval) is not resolved at gene level, and on
gene-level power the four hapmixQTL arms cannot be told apart at this size. Which of the three changes removes the
gap is not identified here; gibbs's anticonservative total channel ({f(an['stored'], 4)} stored at 0.05) is a
candidate, not a measured share. What survives the critique is the signal-side cost, which needs no reference
distribution: gibbs's combined slope has {En_('gibbs', 'combined')} of unit weights' squared error on the anchor's null
genes, where the truth is 0 for every arm, and {E_('gibbs', 'combined')} at the causal variant on the pipeline-scale
truth (every interval above 1); on the count-scale truth the causal-variant ratios are {' / '.join(ci(d, 'value', 2) for d in cc)}, with intervals
that include 1 at {at_betas(inc)}, consistent with unit weights' count-scale error containing the attenuation of
log2(CPM + 1) (section 3.4). Its total-channel standard error is too small (section 3.4).</p>
<p><b>A second objection: the allelic gain could be made by the generator.</b> The allelic Gibbs variance of a thinned
record follows its thinned counts by the generator's own rule, so 1/v weights might track the true error on thinned
records by construction. The anchor answers this: nothing is thinned there and v is Salmon's own, and the gibbs and
split allelic squared error on the anchor's null genes is {En_('gibbs', 'allelic')} of unit weights', against
{E_('gibbs', 'allelic')} at the causal variants. The total channel's loss is present on the anchor too
({En_('gibbs', 'total')}; by band &lt;100 / 100-999 / &ge;1000, {En_band('gibbs', 'total')}). Neither result comes from
the generator's rule.</p>
<p><b>What is not established.</b> At the causal variant the allelic sd(z) is above 1 in point estimate for all four
hapmixQTL arms, unit weights included, but every interval includes 1, and the point excess sits in genes below 100
reads, where the allelic bias is also largest (section 3.4). Whether the stated allelic standard error is too small at
low depth, or sd(z) there absorbs bias, is not established. Because unit weights show the same point excess, it does
not by itself bear on the choice among weightings.</p>"""


def sec_meaning():
    pr = CG['recovery']['primary']
    genes = {bn: S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z'][bn]['genes'] for bn in BANDS}
    st = lambda a, ch: f(S['anchor'][a][ch]['0.05']['stored'], 4)   # noqa: E731
    st3 = lambda a: f(S['anchor'][a]['combined']['0.001']['stored'], 4)   # noqa: E731
    t3 = [fx(a, 'combined', 'after', '0.001')['rate'] for a in HAPMIX[1:]]
    tr = JF['trecase']
    hi = lambda a, ch: ' / '.join(f(prec(f'beta{b}', a, ch, 'nonnull', 'ratio_vs_unit')['hi']) for b in BETAS)   # noqa: E731
    tband = lambda sc, part: ' / '.join(ci(prec(sc, 'gibbs', 'total', part, 'ratio_vs_unit', bn), 'value', 2) for bn in BANDS[1:])   # noqa: E731
    return f"""
<h2>5. What it means for the open decisions</h2>
<p><b>Which weighting ships.</b> Until now the decision rested on the stored 200-permutation null runs, whose
combined rates at 0.05 were {st('gibbs', 'combined')} for gibbs, {st('split', 'combined')} for split,
{st('unit', 'combined')} for unit and {st('plus_one', 'combined')} for plus_one (re-run under commit 8a06803:
{' / '.join(f(fx(a, 'combined', 'after', '0.05')['rate'], 4) for a in HAPMIX)}), with gibbs's total channel at
{st('gibbs', 'total')} and the other three at {st('split', 'total')}. This benchmark adds the signal side. The
separating evidence is the anchor, on unthinned records: there split's combined squared error is
{En_('split', 'combined')} of unit weights', with an interval separated from plus_one's {En_('plus_one', 'combined')} and
gibbs's {En_('gibbs', 'combined')}. At the causal variant it is {E_('split', 'combined')} (upper bounds
{hi('split', 'combined')}). AUC, ranking power and gene-level power do not separate split, unit and plus_one
(sections 3.1 and 3.2). split's total slope is unbiased on the pipeline scale and its total-channel standard error
matches the slope's spread. Its cost is in the allelic channel: split's allelic slope falls short of the
pipeline-scale truth at |beta| 0.8 ({ci(bias('0.8', 'split', 'allelic', 'bias_pipeline'), 'mean')}); check (c)
attributes about 5% to 1/v weights ({f(pr['inv_va_pipeline']['mean'])} against {f(pr['unit_pipeline']['mean'])} for
unit weights), while the benchmark itself does not separate 1/v from unit weights on bias (section 3.3). gibbs has the same
allelic fit, but below 1,000 reads its total-channel weights make the slope less precise than unit weights (by band
&lt;100 / 100-999 / &ge;1000 at |beta| 0.4, {tband('beta0.4', 'nonnull')}; anchor null genes
{tband('beta0.0', 'null')}; at 1,000 reads or more neither a cost nor a gain is shown), and its total channel
understates its standard error, so its combined slope is less precise than unit weights' (anchor null genes
{En_('gibbs', 'combined')}; causal variant {E_('gibbs', 'combined')} on the pipeline-scale truth, a gap the count-scale
truth narrows, section 3.4). plus_one's combined rate at 0.05 is {st('plus_one', 'combined')} in the stored runs and
{ci(fx('plus_one', 'combined', 'after', '0.05'), 'rate', 4)} re-run under commit 8a06803. At 0.001 the stored
combined rates of plus_one, split and unit ({st3('plus_one')}, {st3('split')}, {st3('unit')}) were mostly one gene,
{one_df_gene()}, whose allelic p was referred to 73 degrees of freedom on a one-degree-of-freedom fit; re-run under the
commit they are {ci(fx('plus_one', 'combined', 'after', '0.001'), 'rate', 5)},
{ci(fx('split', 'combined', 'after', '0.001'), 'rate', 5)} and
{ci(fx('unit', 'combined', 'after', '0.001'), 'rate', 5)}, above 0.001 for the two reasons of section 3.7 (the
Welch-Satterthwaite reference and the unit-weighted total channel) and within
{f(max(t3) - min(t3), 5)} of one another, so the tail does not separate them, while gibbs reads {ci(fx('gibbs', 'combined', 'after', '0.001'), 'rate', 4)}. plus_one's combined
squared error is {E_('plus_one', 'combined')} of unit weights', and it keeps less of
the allelic gain than split (anchor {En_('plus_one', 'allelic')} against {En_('split', 'allelic')}). unit weights give that gain up. Nothing here contradicts the null-based record; on
precision the evidence favours split weighting, at the allelic cost just stated. The choice, including leaving the
shipped default, remains a user decision, and section 6 lists what these data cannot settle (100 genes, of which
{genes['100-999'] + genes['>=1000']} have 100 or more median haplotype-informative reads; one permutation rule).</p>
<p><b>Where it narrows earlier results.</b> The 2026-09-19 finding that the Gibbs draws improve the point estimate
(brainvar_hapmix_deploy/mixqtl_replication_20260919/REPORT.md) was measured on the allelic channel of 29
high-coverage genes: the median permutation variance of the slope under 1/v weights was 0.340 of its unweighted
value. It points the same way here and is of similar size in the comparable stratum: at 1,000 or more reads the gibbs
and split allelic squared error is {f(prec('beta0.4', 'gibbs', 'allelic', 'nonnull', 'ratio_vs_unit', '>=1000')['value'], 2)}
of unit weights' at |beta| 0.4 and {f(prec('beta0.0', 'gibbs', 'allelic', 'null', 'ratio_vs_unit', '>=1000')['value'], 2)}
on the anchor's null genes. The statistics differ (there a ratio of median variances over 40 null permutations, here
a ratio of summed squared errors), so only the order of magnitude is compared. Pooled over all 100 genes the ratio
here is {E_('gibbs', 'allelic')}. The total channel was not part of that record. Here its Gibbs weights cost precision
below 1,000 reads, on known effects as on the corrected pipeline's nulls (docs/pipeline_rules.md, "What made the total
channel worse"). An earlier count-scale measurement on the pre-correction pipeline
(brainvar_hapmix_deploy/count_scale_weights_20260925/) had found the total channel's Gibbs weights to buy nothing; on
the corrected pipeline they cost precision below 1,000 reads. Those records differ from this one in gene set, pipeline
and statistic, so only the direction is compared, not the magnitude.</p>
<p><b>mixQTL as the baseline.</b> As run with the published cutoffs, mixQTL has the lowest AUC of the hapmixQTL and
mixQTL arms at every |beta| ({A_('mixqtl')}) and the lowest point share of leads within r<sup>2</sup> &ge; 0.8 of the causal variant
({R_('mixqtl')}, no interval), leaves some gene units without any finite p, and attenuates its slopes in both channels.
With the permissive cutoffs it is closer to the hapmixQTL arms (AUC {A_('mixqtl_permissive')}) but below split's point
estimates ({A_('split')}) at every |beta|; their ranges overlap at the smallest and the largest effect size and separate at
the middle one. Its null-gene combined rates at 0.05 are {per_beta(lambda b: S['null'][f'beta{b}']['mixqtl']['combined']['all']['0.05']['rate'])} (published) and
{per_beta(lambda b: S['null'][f'beta{b}']['mixqtl_permissive']['combined']['all']['0.05']['rate'])} (permissive) at |beta| = 0.2 / 0.4 / 0.8, and
{f(S['null']['beta0.0']['mixqtl']['combined']['all']['0.05']['rate'])} and
{f(S['null']['beta0.0']['mixqtl_permissive']['combined']['all']['0.05']['rate'])} on the anchor. On its own permutation
null its gene-level Benjamini-Hochberg power is {per_beta(lambda b: bh(b, 'mixqtl')['power_bh']['all']['rate'])}
(published) and {per_beta(lambda b: bh(b, 'mixqtl_permissive')['power_bh']['all']['rate'])} (permissive), against
split's {per_beta(lambda b: bh(b, 'split')['power_bh']['all']['rate'])} (section 3.2). Against both mixQTL settings split is ahead in point estimate at every
|beta| on AUC, on ranking power at 5% realized FDP, on the share of leads with r<sup>2</sup> &ge; 0.8, on total-channel
bias on a common count-scale truth ({B_('split', 'total')} against {B_('mixqtl', 'total')} published and
{B_('mixqtl_permissive', 'total')} permissive), and on combined squared error against unit weights with both on the
count-scale truth ({Ex_('split')} against {Ex_('mixqtl')} and {Ex_('mixqtl_permissive')}). At |beta| 0.8
split's lowest dataset AUC ({f(auc('0.8', 'split')['lo'])}) exceeds mixQTL published's highest
({f(auc('0.8', 'mixqtl')['hi'])}). The allelic slope does not separate: on the count scale at |beta| 0.8 mixQTL
permissive recovers {ci(bias('0.8', 'mixqtl_permissive', 'allelic', 'bias_count'), 'mean')} of beta and split
{ci(bias('0.8', 'split', 'allelic', 'bias_count'), 'mean')}, with overlapping intervals at every |beta|; split's
count-scale line keeps as zeros the units with no allelic data, which mixQTL's drops (section 3.3). Most of the
attenuation of mixQTL's total slope comes from its two-step covariate adjustment and its choice of covariates on the
outcome (section 3.8); a one-step fit with all 17 covariates still recovers {'{:.2f} to {:.2f}'.format(*all17())} of
the truth, a remainder not decomposed. Its total slopes should not be used as a reference for effect size.</p>
<p><b>The joint models as comparators.</b> RASQUAL and TReCASE fit a total-count model whose dosage form is the
generator's expected total fold, averaged over donors (section 2). They receive exactly the allele-specific records the
hapmixQTL arms admit; asSeq's own floors then drop {tr['asseq_dropped'][0]} to {tr['asseq_dropped'][1]} of them per
dataset (fewer than five allele-specific reads) and skip the allele-specific model in {tr['few_het']:,} tests
({100 * tr['few_het'] / tr['tests']:.1f}%), which have fewer than five heterozygous donors. On these data neither
outranks split in point estimate: AUC {A_('rasqual')} (RASQUAL) and {A_('trecase')} (TReCASE) against {A_('split')}; power
at 5% realized FDP {P_('rasqual')} and {P_('trecase')} against {P_('split')}; leads within r<sup>2</sup> &ge; 0.8
{R_('rasqual')} and {R_('trecase')} against {R_('split')}. TReCASE's range of per-dataset AUCs overlaps split's at
{at_betas(overlap('trecase', 'split'))}; power at realized FDP and lead recovery have no interval; each effect size
rests on 3 datasets. TReCASE's slope recovers beta within its intervals at every |beta|
({B_('trecase', 'combined')}), which no other combined slope here
does, but its p rejects on null genes at {N_('trecase')}
at 0.05 (anchor, then |beta| = 0.2 / 0.4 / 0.8), and its slope has more squared error than unit weights' at |beta| 0.2
and 0.4 ({Ex_('trecase')} at the three |beta|, on the count-scale truth) and on the anchor's null genes
({En_('trecase', 'combined')}). RASQUAL's rate at 0.05 is nearer
nominal ({N_('rasqual')}), though on the anchor at 0.001
it is {f(S['null']['beta0.0']['rasqual']['combined']['all']['0.001']['rate'] / 0.001, 1)} times nominal; its slope falls
short of beta ({B_('rasqual', 'combined')}) and has {Ex_('rasqual')}
times unit weights' squared error. So on this benchmark split weighting is not outperformed in point estimate by either
joint model on ranking, on ranking power at 5% realized FDP or on lead placement. On squared error only TReCASE's point
estimate at |beta| 0.8 is lower, with overlapping intervals and a denominator that contains unit weights' own
count-scale bias (section 3.4). TReCASE's advantage is in bias, and it comes with an anticonservative nominal p. Both joint models were run on
Salmon point estimates at a pseudo feature SNP rather than on reads (section 6), so this measures them as run here, not
joint likelihood modelling as such.{native_meaning()}</p>"""


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
    rq, tr = JF['rasqual'], JF['trecase']
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
to pull the null-gene rates at |beta| &gt; 0 toward nominal. Section 3.7, comparing them with the stored null re-run
under commit 8a06803, shows that this is not resolved here. Either way those rates are not calibration results. The precision of the 1/v arms on thinned records may
be more favourable than on real records at the same depth; this was not measured. The pipeline's transforms (the
+0.5 of the allelic ratio, the +1 of log2(CPM + 1)) attenuate a fold at low depth, so bias against beta mixes that
attenuation with estimator bias. The pipeline-scale truth separates the two only for an unweighted fit.</p>
<p><b>The allelic variance rule.</b> It overstates Va' by at most (1 - f) x 1.4% at 100-999 total reads and
(1 - f) x 9% at 10-99, because Salmon's Gibbs prior of one pseudo-read per transcript copy does not scale with depth
(README, Generator). No known-answer test of the rule against Salmon run at reduced depth exists; the
premise check is at native depth, on one donor, and its pass thresholds were set after its first result.</p>
<p><b>mixQTL is compared as run.</b> Its gene-level p comes from its own published permutation null (no
haplotype-label swap, no Beta approximation), not hapmixQTL's, and its efficiency against unit weights compares methods
as run, on a different admitted donor set.</p>
<p><b>The joint models are run away from their design.</b> RASQUAL sees Salmon's haplotype point estimates, rounded,
at one pseudo feature SNP per gene, not reads at each heterozygous exonic SNP. The processes its read-level parameters
model, reference-mapping bias, sequencing errors and uncertain genotypes, are absent from these data, yet the
parameters are still fitted and do not sit at their no-effect values (the fitted sequencing error rate is not near
zero), so they may absorb other variation; whether they contribute to RASQUAL's shortfall against beta (section 3.3)
is not tested, and this benchmark says nothing about what they buy on real reads. Its defaults are kept, except that
its Hardy-Weinberg filter on tested variants is off (-h 0).</p>
<p>asSeq's joint TReCASE fit is missing in {pct(tr["joint_na_share"])} of the run's {tr["tests"]:,} tests and in
{pct(tr["causal_joint_na_share"])} of its {tr["causal_nonnull"]} causal-variant tests. asSeq's trace logs attribute
{tr["joint_fail_theta"]:,} of the missing fits ({pct(tr["joint_fail_theta"] / tr["tests"])} of all tests) to the
overdispersion step's search ending abnormally in its line search; that search is L-BFGS-B, an iterative optimizer
that approximates the curvature of the likelihood from its gradients, within bounds. The largest absolute gradient at
such a stop was {tr["theta_gradient_max"]:.1e} over the run, against at most {sci(SM)} in the smoke run ({SMOKE.parent.name}), which
the earlier run_trecase_asseq.py read as a stop at essentially the optimum; whether the full run's abnormal stops are at the optimum
was not checked. Treating them as converged would require patching asSeq and was not done. Where the joint fit is
missing, asSeq's final p is its total-count test; at the causal variants the final p was the total-count test in
{tr["causal_final_trec"]} of {tr["causal_nonnull"]}, the joint test in {tr["causal_final_joint"]}. The trace logs also
count {tr["ase_fail"]:,} failed allele-specific fits and the {tr["linear_dosage"]:,} linear-dosage refits of section 2;
the output rows count {tr["few_het"]:,} tests ({100 * tr["few_het"] / tr["tests"]:.1f}%) with fewer than five
heterozygous donors, for which asSeq fits no allele-specific model (the classes can overlap). In {tr["df_not_1"]:,}
tests ({tr["df_not_1_anchor"]:,} on the anchor) the joint statistic has 0 degrees of freedom (the overdispersion at its
boundary under the alternative), so asSeq reports no p for them. They are absent from the null rates, the ranking and
detection, but their slope and derived standard error (&chi;<sup>2</sup> &gt; 0) enter bias and the precision
statistics, where the derived standard error is not a standard error.</p>
<p>Both joint arms receive exactly the allele-specific records the hapmixQTL arms admit (the zero-haplotype rule of
section 2 removes {tr["zeroed"][0]:,} to {tr["zeroed"][1]:,} haplotype-informative donor-gene pairs per dataset).
Rounding sets {rq["as00"][0]} to {rq["as00"][1]} of them per dataset to zero reads on both haplotypes for RASQUAL, and
asSeq's own floors then drop {tr["asseq_dropped"][0]} to {tr["asseq_dropped"][1]} records per dataset (fewer than five
allele-specific reads) and the allele-specific model at the {tr["few_het"]:,} tests above. Both likelihoods are
written for read counts, and these are Salmon's fractional point estimates, rounded where a model needs
integers{f"; section {native_sec()} gives TReCASE (not RASQUAL) the featureCounts totals and phASER allele counts of the same BAMs and states what that comparison cannot show" if NATIVE else ''}. And
the injected total fold has, averaged over donors, the dosage form both joint models assume (section 2), while the
linear total channels of hapmixQTL and mixQTL approximate it by a straight line; the generator therefore suits the joint
models' total model.</p>
<p><b>Thresholds.</b> Nothing here tests nominal p below 1e-5, measures a null rate below 1e-3, or tests gene-level
thresholds at transcriptome scale. The combined p's Welch-Satterthwaite reference treats estimated channel weights as
fixed; Meier's correction (section 2) is applied to every hapmixQTL combined p on this page, and under the exact model
it leaves 1.004-1.017x nominal at 0.05 and 1.014-1.123x at 0.001 over 15 allelic donors to all
(docs/hapmixqtl_methods.md, Section 4.5), a residual this benchmark does not measure. The eigenMT gene-level p is a
Bonferroni bound over M<sub>eff</sub> tests on the arm's nominal p and inherits its miscalibration; only the permutation
p is referred to the arm's own null.
Of the generator checks, only check (c)'s pass rule was fixed before its first run;
the other thresholds of checks (a) to (c) were not pre-registered (01_check_inputs.py, its parameters' comments).</p>'''


def main():
    load()
    OUT.mkdir(parents=True, exist_ok=True)
    figs = dict(ranking=fig_ranking(), bias=fig_bias(), lead=fig_lead(), efficiency=fig_efficiency())
    head = (sec_head(), sec_why()) + (() if SF is None else (sec_contrast(),))
    tail = (sec_critique(), sec_meaning(), sec_limits()) if INTERPRETED else ()
    body = '\n'.join(head + (sec_run(), sec_results(figs)) + tail)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Plasmode eQTL benchmark</title><style>{CSS}</style>'
            f'</head><body><main>{body}</main></body></html>')
    C.write_atomic(PAGE, lambda fh: fh.write(page.encode()))
    print(f'wrote {PAGE} ({PAGE.stat().st_size:,} bytes) and {", ".join(p.name for p in figs.values())} in {OUT}')


if __name__ == '__main__':
    main()
