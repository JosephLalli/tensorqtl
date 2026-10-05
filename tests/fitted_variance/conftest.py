"""Tests for the DEPRECATED fitted-variance machinery.

Everything in this directory exercises `tensorqtl/fitted_variance.py`: variance
functions fitted per gene from that gene's own squared residuals (`tau_g` alone
under `additive`, `(c_g, tau_g)` jointly under `two_component` /
`library_scaled`), optionally shrunk by `estimate_variance_priors`. None of it
is part of either shipped mode -- mixQTL mode, or default mode
(`Var(eps_i) = sigma^2 v_i` on the Gibbs across-draw variance).

They are kept, and kept passing, for one reason: deprecated is not broken, and
the only justification for retaining the module at all is that it still
reproduces results recorded before 2026-09-23. They are segregated so the
default suite exercises only the two live modes.

Every test here must name the deprecated configuration EXPLICITLY
(`tau_mode='estimate'`, and `se_mode='model'` where a golden predates the
2026-09-21 default flip). Relying on defaults is what silently broke them when
those defaults changed.

This conftest puts the parent `tests/` directory on the path, because these
files borrow fixtures such as `_make_dataset` from `test_hapmixqtl`.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # tests/
sys.path.insert(0, str(HERE.parent.parent))   # repo root
