"""One-off: add an opt-in HWE-filter hook to the diagnostic scripts.

Sets HWE=1 in the environment to restrict to the HWE-passing variant set.
Default stays unfiltered so previous results remain reproducible, and each
run prints which variant set it used.
"""

TARGETS = ['scripts/sandwich_under_permutation.py',
           'scripts/denominator_winners_curse.py',
           'scripts/carriers_needed.py']

HOOK = """    I = load_inputs()
    if os.environ.get('HWE', '0') == '1':
        from make_hwe_filtered_variants import apply_hwe_filter
        I = apply_hwe_filter(I)
        print(f"[HWE-filtered variant set: dropped "
              f"{I['n_dropped_by_hwe']:,} variants]\\n")
    else:
        print('[unfiltered variant set]\\n')"""


def main():
    for p in TARGETS:
        s = open(p).read()
        if 'apply_hwe_filter' in s:
            print('already patched:', p)
            continue
        s = s.replace('    I = load_inputs()', HOOK, 1)
        if '\nimport os\n' not in s:
            s = s.replace('import json\nimport sys',
                          'import json\nimport os\nimport sys', 1)
        open(p, 'w').write(s)
        print('patched:', p)


if __name__ == '__main__':
    main()
