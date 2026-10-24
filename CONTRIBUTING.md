# Contributing

Install the development environment with `uv sync --locked --extra dev`.
Run `uv run ruff check src tests examples`,
`uv run ruff format --check src tests examples`, and
`uv run python -m pytest -q` before proposing a change.

For model changes, state units, sign conventions, state ordering, assumptions
and affected energy terms. Add an independent physical check: analytical
limits, power balance, a known solution, finite-difference Jacobians or a
cross-method comparison. Prefer targeted tests while developing, followed by
the complete suite when the numerical source is stable.

Changes to timing must retain disturbance boundaries and independent task
clocks. Changes to estimation must preserve covariance structure, wrap
angular innovations and explain new noise assumptions.

Evaluate navigation changes on the full warehouse scenario and relevant
fault experiments. Preserve a regression for any discovered failure,
including negative outcomes. A stable linear pole plot alone does not
establish nonlinear tracking performance or actuator compliance.

Regenerate gallery evidence using the documented scripts after changing a
published scenario. Never edit generated metrics to fit an expected result.
Keep source/configuration identifiers with measurements and describe
incomplete missions accurately.

Use focused commits for completed capabilities, experiments, tests or
documentation. Do not add empty commits, artificial timestamps or split
formatting changes solely to increase history size.

Hardware integrations need their own parameter identification, limits,
timing, sensor characterization and validation. Keep simulation evidence
clearly distinguished from measurements.
