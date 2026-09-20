"""One-off: give compare_observed_results.py the same opt-in HWE hook,
and route filtered outputs to their own paths so the two variant sets
cannot overwrite each other as they did once already.
"""

P = 'scripts/compare_observed_results.py'
HOOK = """    I = load_inputs()
    if os.environ.get('HWE', '0') == '1':
        from make_hwe_filtered_variants import apply_hwe_filter
        I = apply_hwe_filter(I)
        print(f"[HWE-filtered variant set: dropped "
              f"{I['n_dropped_by_hwe']:,} variants]")
    else:
        print('[unfiltered variant set]')"""


def main():
    s = open(P).read()
    if 'apply_hwe_filter' in s:
        print('already patched')
        return
    s = s.replace('    I = load_inputs()', HOOK, 1)
    s = s.replace(
        "OUT = f'{D}/mixqtl_replication_20260919'",
        "OUT = f'{D}/mixqtl_replication_20260919'\n"
        "TAG = '_hwe' if os.environ.get('HWE', '0') == '1' else ''", 1)
    for f in ['observed_matched_variants.parquet', 'observed_per_gene.tsv',
              'observed_comparison.json']:
        base, ext = f.rsplit('.', 1)
        s = s.replace('{OUT}/' + f, '{OUT}/' + base + '{TAG}.' + ext)
    s = s.replace("tmp = f'{OUT}/_nominal_tmp'",
                  "tmp = f'{OUT}/_nominal_tmp{TAG}'")
    open(P, 'w').write(s)
    print('patched', P)


if __name__ == '__main__':
    main()
