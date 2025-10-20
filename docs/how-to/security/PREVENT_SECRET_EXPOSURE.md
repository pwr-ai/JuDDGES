# How to Prevent Secret Exposure in Git

This guide provides best practices and tools to prevent accidentally committing sensitive information like API keys, passwords, and tokens to your git repository.

## Quick Start Checklist

- [ ] Ensure `.env` files are in `.gitignore`
- [ ] Install pre-commit hooks with secret detection
- [ ] Use `.env.example` templates instead of real secrets
- [ ] Enable GitHub secret scanning (automatic for public repos)
- [ ] Review all commits before pushing
- [ ] Use environment variables for all secrets
- [ ] Never hardcode credentials in code or documentation

## Environment File Management

### 1. Use .env Files for Local Development

Create a `.env` file for local secrets (NEVER commit this):

```bash
# .env - DO NOT COMMIT
GOOGLE_API_KEY=AIzaSyD...actual-key-here
GEMINI_API_KEY=AIzaSyD...actual-key-here
LANGFUSE_SECRET_KEY=sk-lf-...actual-key-here
DATABASE_PASSWORD=secure-password-here
```

### 2. Create .env.example Template

Create a `.env.example` file with placeholder values (safe to commit):

```bash
# .env.example - Commit this file as a template
GOOGLE_API_KEY=your-google-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
DATABASE_URL=postgresql://user:pass@host:5432/dbname
DATABASE_PASSWORD=your-database-password
WEAVIATE_API_KEY=your-weaviate-api-key
```

### 3. Update .gitignore

Ensure your `.gitignore` file includes all sensitive files:

```gitignore
# Environment variables
.env
.env.local
.env.*.local
*.env
.envrc

# Secrets and credentials
*secret*
*apikey*
*api-key*
*api_key*
credentials.json
service-account*.json
*.pem
*.key
*.pfx
*.p12

# Secret directories
.secrets/
secrets/
private/
.credentials/

# Cloud provider configs
.aws/
.gcloud/
.azure/

# Python cache (might contain secrets)
__pycache__/
*.pyc
.cache/

# Jupyter notebooks with outputs (might contain secrets)
.ipynb_checkpoints/
*-checkpoint.ipynb

# IDE settings that might contain paths
.vscode/settings.json
.idea/

# Docker environment files
docker-compose.override.yml
.env.docker
```

### 4. Verify Files are Ignored

```bash
# Check if a file is properly ignored
git check-ignore .env
git check-ignore credentials.json

# If not ignored, add to .gitignore
echo ".env" >> .gitignore
echo "credentials.json" >> .gitignore

# Remove from git if already tracked
git rm --cached .env
git rm --cached credentials.json
git commit -m "chore: remove sensitive files from git tracking"
```

## Pre-commit Hooks

Pre-commit hooks scan your staged changes before committing to catch secrets automatically.

### Install pre-commit

```bash
# Using pip
pip install pre-commit

# Using brew (macOS)
brew install pre-commit

# Verify installation
pre-commit --version
```

### Configure pre-commit Hooks

Create `.pre-commit-config.yaml` in your repository root:

```yaml
repos:
  # Built-in pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: detect-private-key
        name: Detect private keys
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace

  # Secret detection with detect-secrets
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        name: Detect secrets
        args:
          - '--baseline'
          - '.secrets.baseline'
          - '--exclude-files'
          - 'package-lock.json'
          - '--exclude-files'
          - 'poetry.lock'
          - '--exclude-files'
          - '\.ipynb$'
        exclude: |
          (?x)^(
            package-lock.json|
            poetry.lock|
            .*\.ipynb
          )$

  # GitLeaks for comprehensive secret scanning
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  # Python-specific checks
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
        additional_dependencies: ['bandit[toml]']
```

### Set Up pre-commit Hooks

```bash
# Install the git hook scripts
pre-commit install

# Generate baseline for detect-secrets (initial setup)
detect-secrets scan > .secrets.baseline

# Test pre-commit on all files
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

### Using pre-commit

```bash
# Hooks run automatically on git commit
git add .
git commit -m "feat: add new feature"
# Pre-commit hooks will run and block commit if secrets detected

# Run manually on specific files
pre-commit run --files path/to/file.py

# Run all hooks on all files
pre-commit run --all-files

# Skip hooks (ONLY if absolutely necessary)
git commit --no-verify -m "emergency fix"
# WARNING: Use --no-verify sparingly and with caution
```

## Code Practices

### 1. Load Secrets from Environment

**Python example**:

```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access secrets
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Use the API key
client = SomeAPIClient(api_key=api_key)
```

### 2. Never Hardcode Secrets

**BAD - Never do this**:

```python
# ❌ NEVER hardcode secrets
api_key = "AIzaSyDomtCP_KKA1G2z9BL8cl70uBTbZ9t-0K4"
password = "my-secret-password"
connection_string = "postgresql://user:password123@host:5432/db"
```

**GOOD - Use environment variables**:

```python
# ✅ Always use environment variables
api_key = os.getenv("GOOGLE_API_KEY")
password = os.getenv("DATABASE_PASSWORD")
connection_string = os.getenv("DATABASE_URL")
```

### 3. Configuration Files

**Use Hydra/OmegaConf for configuration**:

```yaml
# configs/database.yaml
database:
  host: ${oc.env:DB_HOST}
  port: ${oc.env:DB_PORT,5432}
  user: ${oc.env:DB_USER}
  password: ${oc.env:DB_PASSWORD}
  name: ${oc.env:DB_NAME}
```

**Python code**:

```python
from omegaconf import OmegaConf

# Load config (values resolved from environment)
cfg = OmegaConf.load("configs/database.yaml")
connection = connect(
    host=cfg.database.host,
    password=cfg.database.password
)
```

### 4. Documentation Best Practices

**In markdown documentation, NEVER include actual secrets**:

```markdown
❌ BAD:
```bash
export GEMINI_API_KEY=AIzaSyDomtCP_KKA1G2z9BL8cl70uBTbZ9t-0K4
```

✅ GOOD:
```bash
export GEMINI_API_KEY=your-api-key-here
```

Or even better - reference the .env file:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your actual API keys
nano .env
```
```

### 5. Docker and Docker Compose

**Use environment files with Docker**:

```yaml
# docker-compose.yml
services:
  app:
    image: myapp:latest
    env_file:
      - .env  # Load environment variables from .env
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - DATABASE_PASSWORD=${DATABASE_PASSWORD}
```

**Never commit .env but provide .env.example**:

```bash
# .env.example
GOOGLE_API_KEY=your-google-api-key-here
DATABASE_PASSWORD=your-database-password
WEAVIATE_API_KEY=your-weaviate-api-key
```

## Secret Scanning Tools

### 1. GitHub Secret Scanning

GitHub automatically scans public repositories for known secret patterns.

**Enable for private repos**:
1. Go to repository Settings
2. Navigate to "Security & analysis"
3. Enable "Secret scanning"
4. Enable "Push protection"

### 2. TruffleHog

Scan git history for secrets:

```bash
# Install
brew install trufflehogsecurity/trufflehog/trufflehog
# or
pip install truffleHog

# Scan entire git history
trufflehog git file://. --json

# Scan specific commits
trufflehog git file://. --since-commit HEAD~10

# Scan GitHub repository
trufflehog github --repo https://github.com/username/repo
```

### 3. GitLeaks

Fast secret scanner:

```bash
# Install
brew install gitleaks
# or download from https://github.com/gitleaks/gitleaks/releases

# Scan repository
gitleaks detect --source . --verbose

# Scan with custom config
gitleaks detect --config .gitleaks.toml

# Scan commits
gitleaks detect --log-opts="--since=2024-01-01"
```

Create `.gitleaks.toml` config:

```toml
title = "gitleaks config"

[[rules]]
id = "google-api-key"
description = "Google API Key"
regex = '''AIza[0-9A-Za-z\-_]{35}'''

[[rules]]
id = "generic-api-key"
description = "Generic API Key"
regex = '''(?i)(api[_-]?key|apikey)['"]?\s*[:=]\s*['"]?([0-9a-zA-Z\-_]{20,})'''
entropy = 3.5

[allowlist]
paths = [
  '''\\.env\\.example$''',
  '''\\.md$'''  # Allow in documentation
]
```

### 4. detect-secrets

Yelp's secret detection tool:

```bash
# Install
pip install detect-secrets

# Scan repository
detect-secrets scan

# Create baseline
detect-secrets scan > .secrets.baseline

# Audit secrets
detect-secrets audit .secrets.baseline

# Update baseline
detect-secrets scan --baseline .secrets.baseline
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/secret-scan.yml`:

```yaml
name: Secret Scanning

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master, develop ]

jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}

  trufflehog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: TruffleHog OSS
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
```

## Secret Management Solutions

For production environments, use dedicated secret management:

### 1. HashiCorp Vault

```bash
# Store secret
vault kv put secret/myapp/db password="secure-password"

# Retrieve secret
vault kv get secret/myapp/db
```

### 2. AWS Secrets Manager

```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager')

    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except ClientError as e:
        raise e

# Usage
api_key = get_secret("prod/gemini/api_key")
```

### 3. Google Secret Manager

```python
from google.cloud import secretmanager

def access_secret(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Usage
api_key = access_secret("my-project", "gemini-api-key")
```

### 4. Kubernetes Secrets

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
stringData:
  gemini-api-key: your-api-key-here
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      containers:
      - name: myapp
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: gemini-api-key
```

## Git-crypt for Encrypted Secrets

For teams that need to share secrets in git:

```bash
# Install git-crypt
brew install git-crypt  # macOS
sudo apt install git-crypt  # Ubuntu

# Initialize in repository
git-crypt init

# Add collaborators
git-crypt add-gpg-user USER_GPG_KEY

# Configure .gitattributes
echo "secrets.env filter=git-crypt diff=git-crypt" >> .gitattributes
echo ".env.production filter=git-crypt diff=git-crypt" >> .gitattributes

# Lock/unlock
git-crypt lock
git-crypt unlock
```

## Emergency Response

If you accidentally commit secrets:

1. **Immediately revoke the exposed secret**
2. **Generate new credentials**
3. **Follow the removal guide**: See `docs/how-to/security/REMOVE_EXPOSED_API_KEY.md`
4. **Notify your team**
5. **Monitor for unauthorized usage**

## Security Checklist

### Before Every Commit

- [ ] Review `git diff` for any secrets
- [ ] Ensure pre-commit hooks are installed and running
- [ ] Verify `.env` files are not staged
- [ ] Check documentation for hardcoded credentials
- [ ] Run `git status` to see what's being committed

### Regular Maintenance

- [ ] Weekly: Run `trufflehog` or `gitleaks` on full repository
- [ ] Monthly: Review and update `.gitignore`
- [ ] Monthly: Audit pre-commit hook configuration
- [ ] Quarterly: Rotate API keys and secrets
- [ ] Quarterly: Review access logs for API keys

### Project Setup

- [ ] Add comprehensive `.gitignore`
- [ ] Create `.env.example` template
- [ ] Set up pre-commit hooks
- [ ] Configure secret scanning in CI/CD
- [ ] Document secret management process
- [ ] Train team on security practices

## References

- [OWASP Secret Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [12 Factor App: Config](https://12factor.net/config)
- [pre-commit hooks](https://pre-commit.com/)
- [detect-secrets](https://github.com/Yelp/detect-secrets)
- [gitleaks](https://github.com/gitleaks/gitleaks)
- [trufflehog](https://github.com/trufflesecurity/trufflehog)

## Training Resources

- [GitHub: Securing your repository](https://docs.github.com/en/code-security/getting-started/securing-your-repository)
- [Git Guardian Academy](https://www.gitguardian.com/academy)
- [SANS: Secure Coding Practices](https://www.sans.org/security-resources/posters/secure-coding-practices-checklist/)
