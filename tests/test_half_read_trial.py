"""Count-unit and transform-domain invariants for the experimental half-read trial."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from half_read_trial import half_read


def test_prior_is_half_read_not_one_cpm():
    libraries = np.array([1e6, 20e6])
    counts = np.array([[0., 0.], [0.5, 0.5]])
    # Going from zero to half a read doubles the numerator at every library size.
    np.testing.assert_allclose(np.diff(half_read(counts, libraries), axis=0), 1., atol=1e-14)
    assert np.isfinite(half_read(counts, libraries)).all()
    assert not np.allclose(half_read(counts, libraries), np.log2(counts/libraries*1e6+1))


@pytest.mark.parametrize('count,library', [([-1.], [1.]), ([np.nan], [1.]), ([0.], [0.]), ([0.], [np.inf])])
def test_invalid_domain(count, library):
    with pytest.raises(ValueError):
        half_read(count, library)
