# Contributing to NeuroRiskLogic

First off, thank you for considering contributing to NeuroRiskLogic! It's people like you that make this tool better for everyone in the healthcare community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

1. **Fork the Repository**
   ```bash
   # Click the 'Fork' button on GitHub
   git clone https://github.com/yourusername/NeuroRiskLogic.git
   cd NeuroRiskLogic
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Relevant logs or error messages

**Template:**
```markdown
## Bug Description
A clear and concise description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.9.5]
- NeuroRiskLogic Version: [e.g., 1.0.0]

## Additional Context
Any other context or screenshots.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- Why this enhancement would be useful
- Possible implementation approach

### Contributing Code

#### Clinical Features

When adding new clinical features:

1. Provide scientific justification
2. Include relevant citations
3. Update feature definitions in `data/feature_definitions.json`
4. Add appropriate tests
5. Update documentation

#### ML Model Improvements

For model enhancements:

1. Benchmark against current model
2. Provide performance metrics
3. Ensure backward compatibility
4. Document changes in model card

#### API Endpoints

When adding new endpoints:

1. Follow RESTful conventions
2. Add comprehensive documentation
3. Include request/response examples
4. Write integration tests
5. Update OpenAPI schema

## Development Process

### 1. Local Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run linting
flake8 app/
black app/ --check
mypy app/

# Run the application
uvicorn app.main:app --reload
```

### 2. Testing Requirements

- All new features must have tests
- Maintain test coverage above 80%
- Include both unit and integration tests
- Test edge cases and error conditions

### 3. Documentation

- Update README.md if adding features
- Add docstrings to all functions/classes
- Update API documentation
- Include examples where appropriate

## Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for function arguments and returns
- Use descriptive variable names
- Add docstrings to all public functions

**Example:**
```python
from typing import Dict, List, Optional

def calculate_risk_score(
    features: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate neurodevelopmental risk score.
    
    Args:
        features: Dictionary of clinical features
        weights: Optional custom feature weights
        
    Returns:
        Risk score between 0.0 and 1.0
        
    Raises:
        ValueError: If required features are missing
    """
    # Implementation
    pass
```

### API Design

- Use consistent naming conventions
- Return appropriate HTTP status codes
- Include error details in responses
- Version the API appropriately

### Database Schema

- Use meaningful table and column names
- Add comments to complex fields
- Include appropriate indexes
- Maintain referential integrity

## Commit Messages

We use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

### Examples

```bash
feat(api): Add risk factor analysis endpoint

- Implement /api/v1/stats/risk-factors endpoint
- Add comprehensive risk factor statistics
- Include prevalence and impact analysis

Closes #123

---

fix(predictor): Handle missing features gracefully

- Add validation for required features
- Provide meaningful error messages
- Add test cases for edge conditions

Fixes #456
```

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Update documentation
   - Add yourself to CONTRIBUTORS.md
   - Squash commits if necessary

2. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No new warnings
   ```

3. **Review Process**
   - At least one maintainer review required
   - All CI checks must pass
   - Address review comments
   - Maintain professional discourse

4. **After Merge**
   - Delete your feature branch
   - Update your local main branch
   - Celebrate your contribution! 🎉

## Questions?

Feel free to open an issue for any questions about contributing. We're here to help!

---

Thank you for contributing to NeuroRiskLogic! Your efforts help improve neurodevelopmental healthcare accessibility worldwide.