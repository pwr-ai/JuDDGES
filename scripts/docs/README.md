# Documentation Scripts

Scripts for documentation generation, validation, and testing.

## test_code_examples.py

Automated testing system for Python code examples in documentation.

### Quick Start

```bash
# Test all documentation
python scripts/docs/test_code_examples.py

# Test specific file
python scripts/docs/test_code_examples.py docs/reference/api/README.md

# Verbose output
python scripts/docs/test_code_examples.py --verbose

# Generate JSON report
python scripts/docs/test_code_examples.py --report report.json
```

### Features

- **Automatic Extraction**: Scans markdown files for Python code blocks
- **Syntax Validation**: Validates Python syntax with AST parsing
- **Rich Console Output**: Progress bars and formatted results
- **Report Generation**: Creates JSON reports for CI/CD
- **Annotation Support**: Skip examples or mark requirements
- **Parallel Execution**: Tests multiple examples concurrently

### Configuration

Edit `.doctest.yaml` in the project root to configure:

- Paths to scan/exclude
- Execution timeouts
- Skip markers and annotations
- Mock configurations
- Reporting options

See [Code Example Testing Reference](/home/laugustyniak/github/legal-ai/JuDDGES/docs/reference/CODE_EXAMPLE_TESTING.md) for complete documentation.

### Integration

**Pre-commit Hook**: Automatically runs on documentation changes

**GitHub Actions**: Runs on PRs and commits to main

**pytest**: Integrated with pytest test suite

```bash
pytest tests/docs/test_documentation_examples.py -v
```

### Dependencies

- `pyyaml`: Configuration parsing
- `rich`: Console output
- `loguru`: Logging

Install with:

```bash
pip install pyyaml rich loguru
```

Or install the project in development mode:

```bash
pip install -e .
```

## Future Scripts

Additional documentation scripts can be added here:

- API documentation generators
- Diagram generation tools
- Documentation coverage analyzers
- Link validation utilities
