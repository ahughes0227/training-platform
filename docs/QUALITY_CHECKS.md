# Quality checks

`scripts/quality_checks.py` composes optional security, dependency, and dead-code checks into one compact JSON result. It never installs packages or contacts a network by default. A missing tool is reported as `unavailable`; the default exit code remains zero so local offline checks are safe.

Run the complete lane after installing the tools in the environment:

```sh
python scripts/quality_checks.py --pretty
```

For CI, install the tools through the repository's dependency workflow and make missing tools or findings fail the job:

```sh
python scripts/quality_checks.py --strict --require-tools --output quality-checks.json
```

The checks are:

- `bandit`: Python security findings in `src/` and `scripts/`.
- `pip-audit`: installed environment vulnerabilities. It is skipped unless `--allow-network` is supplied because its advisory database may require network access:

  ```sh
  python scripts/quality_checks.py --only pip-audit --allow-network --strict
  ```

- `vulture`: likely unused Python definitions.
- `deptry`: dependencies declared in `pyproject.toml` but unused, and imports that are not declared.

The JSON envelope has `schema_version`, `network_allowed`, one record per check, and a summary with counts. Each finding is reduced to its file/line/rule or package/fix fields where the tool provides them. Human-oriented output is retained only as a bounded diagnostics list for tool errors.

The repository does not add these tools to runtime dependencies. Install the optional quality set with `uv pip install --python .venv/bin/python -r requirements-quality.txt` (or through the CI environment that owns developer tooling). Run that install after `uv sync`, because `uv sync` reconciles the environment to the lockfile and removes packages that are only in this standalone quality set. The script also accepts tools installed as console commands or importable modules, so it works with both virtualenv and `uv` environments. The property tests skip cleanly when Hypothesis is not installed.
