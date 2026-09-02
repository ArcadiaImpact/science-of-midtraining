"""GPU-side entry points for the diverse-response AFT study.

Importing any of these puts the checkout's ``src/`` on ``sys.path``, the way
``dispatch_final_v1/pod/train_aft.py`` and ``pod/chain.py`` do. Pods run from a
plain git clone at ``/workspace/scimt`` with no editable install, so
``import scimt`` resolves only if something adds it -- and finding that out
inside ``train_cell`` costs a paid GPU-hour, not a desk-time import error.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: .../<repo>/experiments/prior_coins/dispatch_final_v1/diverse_response_v1/pod
_REPO_ROOT = Path(__file__).resolve().parents[5]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
