# API Documentation Quick Start

Get the JuDDGES API documentation up and running in 5 minutes.

## Installation

### Step 1: Install Dependencies

```bash
# Install documentation tools
uv pip install mkdocs mkdocs-material mkdocstrings[python]

# Or install all dev dependencies at once
uv pip install -e ".[dev]"
```

### Step 2: Install JuDDGES Package

```bash
# Install in development mode (from project root)
uv pip install -e .
```

## Viewing Documentation

### Local Development Server

Start a local server with auto-reload:

```bash
# Method 1: Direct mkdocs command
mkdocs serve

# Method 2: Using convenience script
./scripts/docs/generate_api_docs.sh --serve
```

Then open your browser to: **http://127.0.0.1:8000**

The server will automatically reload when you:
- Modify Python docstrings
- Update markdown files
- Change configuration

### Building Static Site

Generate static HTML for deployment:

```bash
# Method 1: Direct mkdocs command
mkdocs build

# Method 2: Using convenience script
./scripts/docs/generate_api_docs.sh --build
```

Output will be in the `site/` directory.

## Exploring the Documentation

### Key Entry Points

1. **[API Reference Index](index.md)**
   - Overview of all modules
   - Quick navigation
   - Use case guides

2. **[Data Management](data/index.md)**
   - Dataset loading
   - Weaviate operations
   - Ingestion pipelines

3. **[LLM Operations](llm/factory.md)**
   - Model loading
   - Quantization
   - Adapter management

4. **[Information Extraction](extraction/gemini_chain.md)**
   - Gemini-based extraction
   - Schema definitions
   - Batch processing

5. **[Evaluation Metrics](evals/metrics.md)**
   - Field-level metrics
   - Evaluation patterns
   - Aggregation

## Common Workflows

### Workflow 1: Finding an API

```
1. Visit http://127.0.0.1:8000
2. Use search (Ctrl+K or click search icon)
3. Search for "DatasetLoader" or relevant term
4. Click result to view documentation
5. Copy example code
```

### Workflow 2: Adding Documentation

```bash
# 1. Write code with docstrings
cat > juddges/mymodule.py << 'EOF'
def my_function(arg: str) -> dict:
    """Brief description.

    Args:
        arg: Description

    Returns:
        Result dict

    Example:
        >>> result = my_function("test")
    """
    return {"result": arg}
EOF

# 2. Create documentation page
cat > docs/reference/api/mymodule.md << 'EOF'
# My Module

::: juddges.mymodule.my_function
    options:
      show_root_heading: true
      show_source: true
EOF

# 3. View locally
mkdocs serve

# 4. Commit both files
git add juddges/mymodule.py docs/reference/api/mymodule.md
git commit -m "Add my_function with documentation"
```

### Workflow 3: Building for Deployment

```bash
# 1. Clean previous build
rm -rf site/

# 2. Build with strict validation
./scripts/docs/generate_api_docs.sh --build --strict

# 3. Test locally
cd site && python -m http.server 8000

# 4. Deploy to hosting service
# (GitHub Pages, Netlify, Vercel, etc.)
```

## Navigation Tips

### Using Search

**Keyboard Shortcuts**:
- `Ctrl+K` or `/` - Open search
- `Esc` - Close search
- Arrow keys - Navigate results

**Search Tips**:
- Search by module name: "DatasetLoader"
- Search by function: "evaluate_date"
- Search by concept: "quantization"

### Browsing by Category

**By Module Type**:
- Data → `docs/reference/api/data/`
- LLM → `docs/reference/api/llm/`
- Extraction → `docs/reference/api/extraction/`
- Evaluation → `docs/reference/api/evals/`
- Preprocessing → `docs/reference/api/preprocessing/`

**By Use Case**:
- Data ingestion path: Loaders → Ingester → Database
- Model inference path: Factory → Predict → Metrics
- Extraction path: Gemini Chain → Metrics → Judge

## Troubleshooting

### MkDocs Not Found

```bash
# Error: mkdocs: command not found

# Solution: Install mkdocs
uv pip install mkdocs mkdocs-material mkdocstrings[python]
```

### Module Import Errors

```bash
# Error: No module named 'juddges'

# Solution: Install package
uv pip install -e .
```

### Port Already in Use

```bash
# Error: [Errno 98] Address already in use

# Solution: Use different port
mkdocs serve -a localhost:8001
```

### Documentation Not Updating

```bash
# Problem: Changes not reflected

# Solution 1: Hard refresh browser (Ctrl+Shift+R)

# Solution 2: Restart server
# Ctrl+C to stop, then:
mkdocs serve
```

## Next Steps

### For Users

1. **Explore APIs**: Browse [API Index](index.md)
2. **Try Examples**: Copy code examples to test
3. **Read Guides**: Check [How-To Guides](../../how-to/)
4. **Ask Questions**: Use GitHub Discussions

### For Contributors

1. **Read Guide**: [API Documentation Guide](API_DOCUMENTATION_GUIDE.md)
2. **Follow Standards**: [Style Guide](../STYLE_GUIDE.md)
3. **Add Docstrings**: Use Google style
4. **Test Locally**: Verify docs render correctly
5. **Submit PR**: Include docs with code

### For Maintainers

1. **Review Coverage**: Check module coverage
2. **Update Examples**: Keep examples current
3. **Fix Links**: Validate cross-references
4. **Monitor Issues**: Address doc-related issues
5. **Release Docs**: Tag docs with releases

## Configuration Reference

### MkDocs Configuration

Located in `/home/laugustyniak/github/legal-ai/JuDDGES/mkdocs.yml`:

- **Site metadata**: Name, description, repository
- **Theme**: Material with light/dark mode
- **Plugins**: mkdocstrings for Python autodoc
- **Extensions**: Mermaid, tabs, admonitions
- **Navigation**: 50+ page structure

### mkdocstrings Options

Common options in documentation pages:

```markdown
::: module.Class
    options:
      show_root_heading: true        # Show class name
      show_source: true              # Show source code
      heading_level: 2               # Heading level (1-6)
      show_signature_annotations: true  # Show types
      separate_signature: true       # Multi-line signature
      show_if_no_docstring: false   # Hide undocumented
```

## Resources

### Documentation Files

```
/home/laugustyniak/github/legal-ai/JuDDGES/
├── mkdocs.yml                          # Configuration
├── docs/reference/api/
│   ├── index.md                        # API index
│   ├── README.md                       # This file
│   ├── QUICK_START.md                  # Quick start guide
│   ├── API_DOCUMENTATION_GUIDE.md      # Complete guide
│   ├── API_IMPLEMENTATION_SUMMARY.md   # Implementation summary
│   └── [module]/                       # Module docs
└── scripts/docs/
    └── generate_api_docs.sh            # Generation script
```

### External Links

- [MkDocs](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Google Docstring Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### JuDDGES Documentation

- [Main Documentation](../../README.md)
- [Tutorials](../../tutorials/)
- [How-To Guides](../../how-to/)
- [Explanation](../../explanation/)

## Quick Reference

### Essential Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Serve locally
mkdocs serve

# Build static site
mkdocs build

# Validate
./scripts/docs/generate_api_docs.sh

# Build with script
./scripts/docs/generate_api_docs.sh --build

# Serve with script
./scripts/docs/generate_api_docs.sh --serve
```

### File Locations

| Purpose | Location |
|---------|----------|
| Configuration | `mkdocs.yml` |
| API docs | `docs/reference/api/` |
| Scripts | `scripts/docs/` |
| Output | `site/` |

### URLs

| Environment | URL |
|-------------|-----|
| Local dev | http://127.0.0.1:8000 |
| API reference | http://127.0.0.1:8000/reference/api/ |
| Search | http://127.0.0.1:8000/?q=query |

---

**Need help?** Check the [API Documentation Guide](API_DOCUMENTATION_GUIDE.md) or open an issue on GitHub.

**Ready to explore?** → [Browse API Reference](index.md)
