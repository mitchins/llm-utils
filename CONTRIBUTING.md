# Contributing

## Installing test dependencies

The test suite uses **pytest** and depends on a few additional packages. At a minimum you will need

- `numpy`
- `httpx`
- `pytest`

Some tests import modules such as `torch`, `transformers`, `datasets` and
`sentence-transformers`. These heavy libraries are patched with dummy modules so
installing them is optional unless you want to run the library code yourself.

Install the minimal set with:

```bash
pip install pytest numpy httpx
```

If you prefer to run the tests with the real libraries installed, also install
`torch`, `transformers`, `datasets` and `sentence-transformers`.

## Running the tests

From the repository root simply run:

```bash
pytest
```

This will execute all tests in the `tests/` directory.
