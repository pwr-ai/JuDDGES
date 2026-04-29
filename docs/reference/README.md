# Reference Documentation

## Purpose

Reference documentation is **information-oriented** material that provides technical descriptions of the system. It's designed for users who need to look up specific information about APIs, configurations, schemas, and technical specifications.

## What Belongs Here

- API documentation
- Configuration parameters
- Schema definitions
- Command-line interfaces
- Error codes and messages
- File formats
- Data structures
- Function/method signatures
- Class hierarchies

## Characteristics

- Comprehensive and accurate
- Structured for easy lookup
- Dry but precise language
- Consistent formatting
- Complete parameter descriptions
- Examples for clarity
- Searchable and indexed

## Current Categories

### [API](api/)

- LLM fields quick reference
- Raw text processing reference
- Module and function documentation

### [Schemas](schemas/)

- Document schema mappings
- Dataset to Weaviate mappings
- Data structure definitions

### [Configurations](configurations/)

- Configuration file formats
- Parameter descriptions
- Environment variables

### Style Guide

- [Style Guide](STYLE_GUIDE.md) - Coding and documentation standards

## Reference Template

```markdown
# [Component] Reference

## Overview
Brief technical description.

## API Methods

### methodName()
**Description**: What it does.

**Parameters**:
- `param1` (type): Description
- `param2` (type): Description

**Returns**: Return type and description

**Example**:
​```language
code example
​```

**Errors**:
- `ERROR_CODE`: Description

## Configuration

### parameter_name
**Type**: string | number | boolean
**Default**: value
**Description**: What this parameter controls
**Example**: `parameter_name: "value"`

## See Also
- [Related Reference]
- [How-To Guide]
```

## Writing Guidelines

1. Be exhaustive - document everything
2. Use consistent terminology
3. Provide type information
4. Include default values
5. Show practical examples
6. List all possible errors/exceptions
7. Maintain alphabetical ordering where appropriate
