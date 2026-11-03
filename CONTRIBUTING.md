# Contributing

Thanks for your interest! Bug reports, questions and pull requests are welcome.

## Development setup

```
git clone https://github.com/amartinez118/unzipy && cd unzipy
uv sync --extra dev --extra fast   # creates .venv from uv.lock
uv run pytest                      # ~25 s
uv run ruff check . && uv run ruff format --check .
uv run pyright                     # type check
```

## Guidelines

- **Style.** Google-style docstrings (`Args:`, `Returns:`, `Raises:`,
  `Attributes:` for classes) on every module, class and function, lines of at
  most 79 columns; `ruff check .` and `ruff format --check .` enforce both.
- **Units** are pN, nm and kT; state units in docstrings. Prefer `numpy.random.Generator` through an
  `rng` argument (seed or generator) over global state.
- **Vectorise.** Functions should accept arrays; avoid Python loops over data points or trajectories.
- **Test against something independent**: an analytic limit, a numerical integral, a published
  number, a brute-force simulation. A test that re-implements the formula it checks is not enough.
- **Validate inputs** at the public boundary and raise `ValueError` with an actionable message.
- **Molecules are parameters.** Never hard-code a stem or loop length, or a loop free energy.
- Cite the paper for any method taken from the literature, in the docstring and in
  `docs/methods.md`.
- Keep pull requests focused; add a line to `CHANGELOG.md`.

## Reporting a bug

Please include the versions (`pip list | grep -E "numpy|scipy|unzipy"`), a minimal script that
reproduces it (synthetic data is fine), the expected and the observed result.
