import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from half_read_gene_cache import leads


def fixed(genes):
    return pd.DataFrame({'gene': genes, 'is_null': [False] * len(genes)})


def test_leads_orders_equal_p_by_statistic_then_variant():
    fitted = pd.DataFrame({
        'gene': ['g1', 'g1', 'g1'], 'variant_id': ['z', 'b', 'a'],
        'pval_nominal': [.01, .01, .01], 'slope': [2., 3., 3.], 'slope_se': [1., 1., 1.],
    })
    selected, exclusions = leads(fitted, fixed(['g1']))
    assert selected.loc[0, 'lead_variant'] == 'a'
    assert selected.loc[0, 'lead_p'] == .01
    assert exclusions.empty


@pytest.mark.parametrize('value, message', [('bad-p', 'malformed'), (np.inf, 'nonfinite'), (-.01, 'out-of-domain'), (1.01, 'out-of-domain')])
def test_leads_rejects_malformed_or_domain_pvalues(value, message):
    fitted = pd.DataFrame({'gene': ['g1'], 'variant_id': ['v1'], 'pval_nominal': [value],
                           'slope': [1.], 'slope_se': [1.]})
    with pytest.raises(ValueError, match=message):
        leads(fitted, fixed(['g1']))


def test_leads_retains_gene_family_and_records_untestable_pair():
    fitted = pd.DataFrame({'gene': ['g1', 'g2', 'g2'], 'variant_id': ['v1', 'v2', 'v3'],
                           'pval_nominal': [.05, np.nan, 0.], 'slope': [1., 1., 2.], 'slope_se': [1., 1., 1.]})
    selected, exclusions = leads(fitted, fixed(['g1', 'g2']))
    assert selected.gene.tolist() == ['g1', 'g2']
    assert selected.lead_variant.tolist() == ['v1', 'v3']
    assert selected.lead_p.tolist() == [.05, 0.]
    assert exclusions[['gene', 'exclusion']].to_dict('records') == [
        {'gene': 'g2', 'exclusion': 'untestable_nonfinite_nominal_p'}]
    with pytest.raises(AssertionError, match='g2'):
        leads(fitted.iloc[:2], fixed(['g1', 'g2']))
