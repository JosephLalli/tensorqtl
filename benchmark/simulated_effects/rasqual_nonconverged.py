"""RASQUAL's non-converged rows scored instead of dropped (task 2026-10-02).

04_run_rasqual.py keeps only the RASQUAL rows whose convergence flag (field 23) is 0. This script rebuilds the gene
set's RASQUAL results from the raw rows 04 checkpoints (raw_repNNN/<gene>.txt) with 04's own assemble(), three ways:
  converged  04's construction; must equal the delivered nominal_repNNN.parquet in every column (the known answer)
  reported   non-converged rows kept at the chi-square RASQUAL reports (chisq <= 0 gives p = 1, as for a converged row)
  p_one      non-converged rows kept with chisq set to 0: p = 1 and no derived standard error
and scores 'reported' and 'p_one' with 06_score.py unchanged, each in a root whose every other input is a link to the
delivered run, so every number of every other arm must equal the delivered summary's.

Output OUT/<variant>/ (results_rasqual/, the links, 06's summary.json and 06_score.log) and OUT/compare.json: per
variant, RASQUAL's headline statistics beside the delivered ones, the summary values outside RASQUAL that differ
(path strings aside), and the gene-dataset units whose lead variant changes.
Usage: rasqual_nonconverged.py   (SIMULATED_EFFECTS_GENE_SET picks the gene set)
"""
import json
import math
import os
import subprocess
import sys

import pandas as pd

import common as C

R4 = C.module('04_run_rasqual')
OUT = C.D / 'rasqual_nonconverged_20261002' / C.GENE_SET
VARIANTS = ('reported', 'p_one')
LINKED = ('datasets', 'results', 'results_trecase', 'native', 'eigenmt_m_eff.tsv')   # all else 06_score.py reads from ROOT
BETAS, ALPHAS = ('0.2', '0.4', '0.8'), ('0.05', '0.01', '0.001')


def raw_rows(path, g):
    rows = [ln.split('\t') for ln in path.read_text().splitlines()]
    if not rows or any(len(r) != len(C.RASQUAL_FIELDS) or r[0] != g or r[1] == 'SKIPPED' for r in rows):
        raise SystemExit(f'{path}: malformed or SKIPPED RASQUAL rows')
    return pd.DataFrame(rows, columns=C.RASQUAL_FIELDS)


def leads(tab, genes):
    """Each gene's lead variant: the largest chi-square, so the smallest p (RASQUAL's p is chi2.sf of it); '' if none."""
    lead = tab.loc[tab.groupby('phenotype_id', sort=False).chisq.idxmax()].set_index('phenotype_id').variant_id
    return lead.reindex(genes).fillna('')


def rebuild(S, meta):
    """Write each variant's nominal files after checking the converged rebuild; count rows and changed leads."""
    added = dict(nonconverged=0, chisq_positive=0)
    changed = {v: dict(units=0, null=0, nonnull=0, to_causal=0, from_causal=0) for v in VARIANTS}
    for sc, r in C.runs(meta):
        ds = C.load_dataset(C.DATASETS, sc, r)
        d = C.JOINT['rasqual'] / sc / 'rasqual'
        parts = {v: [] for v in ('converged',) + VARIANTS}
        for k, g in enumerate(S['genes']):
            raw = raw_rows(d / f'raw_rep{r:03d}' / f'{g}.txt', g)
            nc = (raw.convergence.astype(float) != 0) & (raw.rs_id != f'{g}_pseudo_fsnp')
            added['nonconverged'] += int(nc.sum())
            added['chisq_positive'] += int((nc & (raw.chisq.astype(float) > 0)).sum())
            reported = raw.assign(convergence=raw.convergence.mask(nc, '0'))
            p_one = reported.assign(chisq=reported.chisq.mask(nc, '0'))
            causal_g = None if ds['is_null'][k] else str(ds['causal_variant'][k])
            for v, frame in (('converged', raw), ('reported', reported), ('p_one', p_one)):
                parts[v].append(R4.assemble(g, frame, S['tested'][g], causal_g)[0])
        tab = {v: pd.concat(p, ignore_index=True) for v, p in parts.items()}
        pd.testing.assert_frame_equal(tab['converged'], pd.read_parquet(d / f'nominal_rep{r:03d}.parquet'))
        base = leads(tab['converged'], S['genes'])
        causal = pd.Series(ds['causal_variant'].astype(str), index=S['genes'])
        null = pd.Series(ds['is_null'], index=S['genes'])
        for v in VARIANTS:
            C.write_parquet(tab[v], OUT / v / 'results_rasqual' / sc / 'rasqual' / f'nominal_rep{r:03d}.parquet',
                            C.fingerprint(ds, 'rasqual'), 'log2')
            new = leads(tab[v], S['genes'])
            moved = new.ne(base)
            c = changed[v]
            c['units'] += int(moved.sum())
            c['null'] += int((moved & null).sum())
            c['nonnull'] += int((moved & ~null).sum())
            c['to_causal'] += int((moved & ~null & new.eq(causal)).sum())
            c['from_causal'] += int((moved & ~null & base.eq(causal)).sum())
        print(f'{sc} rep {r:03d}: rebuilt converged rows equal the delivered file ({len(tab["converged"]):,} rows); '
              f'with non-converged rows {len(tab["reported"]):,}; cumulative lead changes: reported '
              f'{changed["reported"]["units"]}, p_one {changed["p_one"]["units"]}', flush=True)
    print(f'non-converged rows added over the datasets: {added}', flush=True)
    return added, changed


def score(v):
    """06_score.py on the variant's root: its RASQUAL files, links to every other delivered input."""
    root = OUT / v
    for name in LINKED:
        if not (C.ROOT / name).exists():
            raise SystemExit(f'{C.ROOT / name}: missing in the delivered run')
        if not (root / name).is_symlink():
            (root / name).symlink_to(C.ROOT / name)
    with open(root / '06_score.log', 'w') as fh:
        subprocess.run([sys.executable, str(C.HERE / '06_score.py')], stdout=fh, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'SIMULATED_EFFECTS_ROOT': str(root)})
    return json.loads((root / 'summary.json').read_text())


def flat(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, path + (str(k),))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, path + (str(i),))
    else:
        yield path, x


def other_differences(new, old):
    """Summary values outside RASQUAL that differ, path strings (which name the root) aside."""
    a, b = dict(flat(new)), dict(flat(old))
    out = []
    for p in sorted(a.keys() | b.keys()):
        if 'rasqual' in p:
            continue
        if p not in a or p not in b:
            out.append('/'.join(p))
        elif isinstance(a[p], str) and isinstance(b[p], str) and a[p].startswith('/') and b[p].startswith('/'):
            continue
        elif not (a[p] == b[p] or (isinstance(a[p], float) and isinstance(b[p], float) and math.isnan(a[p]) and math.isnan(b[p]))):
            out.append('/'.join(p))
    return out


def headline(S):
    rk = lambda b: S['ranking'][f'beta{b}']['rasqual']   # noqa: E731
    return dict(power=[rk(b)['fdp_matched']['all']['power'] for b in BETAS],
                called=[rk(b)['fdp_matched']['discoveries'] for b in BETAS],
                false=[rk(b)['fdp_matched']['false'] for b in BETAS],
                auc=[rk(b)['auc']['all']['mean'] for b in BETAS],
                anchor_null={al: S['null']['beta0.0']['rasqual']['combined']['all'][al] for al in ALPHAS},
                bias=[S['recovery'][f'beta{b}']['rasqual']['combined']['bias_count']['all'] for b in BETAS],
                causal_detection_005=[S['detection'][f'beta{b}']['rasqual']['combined']['all']['0.05'] for b in BETAS],
                missing_causal=[S['missing_causal'][f'beta{b}']['rasqual'] for b in BETAS])


def main():
    if os.environ.get('SIMULATED_EFFECTS_ROOT'):
        raise SystemExit('unset SIMULATED_EFFECTS_ROOT: this script reads the delivered run and sets the root for 06 itself')
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    added, changed = rebuild(S, meta)
    delivered = json.loads(C.SUMMARY.read_text())
    res = dict(gene_set=C.GENE_SET, delivered_root=str(C.ROOT), nonconverged_added=added, lead_changes=changed,
               delivered=headline(delivered), variants={})
    for v in VARIANTS:
        new = score(v)
        diff = other_differences(new, delivered)
        res['variants'][v] = dict(headline=headline(new), other_arm_differences=diff)
        print(f'{v}: {len(diff)} summary values outside RASQUAL differ from the delivered summary {diff[:5]}', flush=True)
    C.write_json(OUT / 'compare.json', res)
    for k in ('power', 'auc', 'anchor_null', 'causal_detection_005', 'missing_causal'):
        print(f'{k}: delivered {res["delivered"][k]}; ' + '; '.join(f'{v} {res["variants"][v]["headline"][k]}' for v in VARIANTS))
    print(f'wrote {OUT / "compare.json"}')


if __name__ == '__main__':
    main()
