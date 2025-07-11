# Agent Guidelines

## Testing
To run the full test suite you **must** install the optional training and test
dependencies (this pulls in heavy packages such as `datasets`):

```bash
pip install -e .[training,test]
pip install evaluate
```

Once dependencies are installed, execute the tests with `pytest`.
