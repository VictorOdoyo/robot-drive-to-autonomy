# Reproducing results

Use Python 3.11 or newer. The checked-in uv lock resolves supported dependency
versions across Python versions; it does not require the same NumPy build on
every platform.

```sh
uv sync --locked --extra dev
uv run robotlab list
uv run robotlab run --case warehouse --output outputs/warehouse
uv run robotlab batch --output outputs/catalog
uv run python examples/build_gallery.py
```

A batch runs the 11 scenarios in catalog order and writes numbered report
folders, plus aggregate JSON. The default warehouse lasts 40 simulated
seconds; commissioning and fault cases last seven. Wall time depends on CPU
and Python/BLAS builds. All random streams use explicit scenario seeds.

Reports contain an HTML page, scientific PNG, labeled SI CSV, metrics JSON,
provenance JSON, a compressed NPZ replay archive and a SHA-256 file manifest.
The archive contains numeric/string arrays and JSON metadata, without Python
object arrays. Loading disables pickle, validates schema, trace shapes,
finite values, covariance structure and duty bounds.

```sh
uv run robotlab replay outputs/warehouse/mission.npz --output outputs/replayed
uv run robotlab verify outputs/replayed
```

Replay regenerates presentation from recorded samples without rerunning the
physics. It preserves the original source/configuration provenance. A
configuration hash covers both the actual scenario and plant parameters.
Source checkouts record a Git revision; installed wheels report the package
version and may have no Git revision. Hashes detect accidental file changes;
they are not signed attestations or a guarantee of model validity.

The gallery retains per-experiment configurations, measured metrics and
source identifiers. Its generator rejects an archive whose scenario differs
from the current catalog. It does not silently relabel old results.

Custom JSON uses the Scenario fields and disturbance objects shown by
`examples/custom_case.json`. Unknown keys and invalid geometry/timing fail
before the simulation starts. CLI duration overrides truncate disturbance
windows to the requested horizon. Reducing duration can intentionally leave
a mission unfinished.

Floating-point trajectories need not be bitwise identical across platforms.
Regression limits use physical tolerances and conservation residuals rather
than comparing every trace bit to a golden archive.
