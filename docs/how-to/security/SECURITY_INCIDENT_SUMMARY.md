# Security Incident Summary - Exposed GEMINI_API_KEY

## Incident Overview

**Date Detected**: 2025-10-13
**Incident Type**: Exposed API Key in Public Repository
**Severity**: HIGH
**Status**: REQUIRES IMMEDIATE ACTION

## What Was Exposed

- **Secret Type**: Google Gemini API Key
- **Key Value**: `[REDACTED-API-KEY]`
- **Location**: `docs/EXTRACTION_STATUS.md` lines 64-65
- **Branch**: `feat/umap-calc`
- **First Commit**: `c123bd8` (feat(extraction): add Gemini LLM extraction chain for legal documents)
- **Total Affected Commits**: 9 commits

## Affected Commits

```
f54d631 - fix: update Weaviate gRPC port to 8085 across multiple scripts and docker-compose
a06285d - Update Weaviate gRPC port and add large-scale extraction script
930acec - fix: update gRPC port configuration and enhance Gemini extraction chain
0a1b8f4 - docs: add comprehensive extraction and schema documentation
5a0d3a0 - Implement code changes to enhance functionality and improve performance
2b97322 - (earlier commits)
9d2fbb5 - (earlier commits)
85ab20b - docs: reorganize documentation following Diátaxis framework
c123bd8 - feat(extraction): add Gemini LLM extraction chain for legal documents [FIRST EXPOSURE]
```

## Current Status

### Working Tree (Current Branch)
✅ **Key removed** - The exposed key has been removed from `docs/EXTRACTION_STATUS.md` in the working tree

### Git History
⚠️ **Still exposed** - The key still exists in git history across 9 commits

### Remote Repository
⚠️ **Status unknown** - If these commits were pushed, the key is exposed publicly

## IMMEDIATE ACTIONS REQUIRED

### 1. REVOKE THE API KEY (PRIORITY 1)

**Do this RIGHT NOW before anything else:**

Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and:
1. Find the key ending in `...t-0K4`
2. Click "Delete" or "Revoke"
3. Confirm deletion

**Why this is critical**: Even after cleaning git history, the key may have already been discovered by automated scanners that index public GitHub repositories.

### 2. Generate New API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create new API key
3. Store in local `.env` file (NOT in git)
4. Update your local environment

### 3. Clean Git History

You have three options (in order of recommendation):

#### Option A: git-filter-repo (RECOMMENDED)

```bash
# Backup first
tar -czf ../JuDDGES-backup-$(date +%Y%m%d-%H%M%S).tar.gz .

# Create replacement expressions
cat > /tmp/replace-expressions.txt << 'EOF'
[REDACTED-API-KEY]==>[REDACTED-API-KEY]
EOF

# Run git-filter-repo
pip install git-filter-repo
git filter-repo --replace-text /tmp/replace-expressions.txt --force

# Verify
git log -S "[REDACTED-API-KEY]" --all
# Should return nothing
```

#### Option B: BFG Repo-Cleaner

```bash
# Backup first
tar -czf ../JuDDGES-backup-$(date +%Y%m%d-%H%M%S).tar.gz .

# Install BFG
brew install bfg  # macOS
# or download from https://rtyley.github.io/bfg-repo-cleaner/

# Create passwords file
echo "[REDACTED-API-KEY]" > /tmp/passwords.txt

# Run BFG
bfg --replace-text /tmp/passwords.txt .

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

#### Option C: Interactive Rebase (Only if not pushed)

```bash
# Find first bad commit
git log -S "[REDACTED-API-KEY]" --all --oneline

# Rebase from parent of c123bd8
git rebase -i c123bd8~1

# Mark commits for 'edit', remove the key from each
# Then continue the rebase
```

### 4. Force Push (After Cleaning)

⚠️ **WARNING**: This rewrites history. Coordinate with your team first.

```bash
# Check if commits were pushed
git log origin/feat/umap-calc..HEAD

# If you need to force push:
git push origin feat/umap-calc --force-with-lease

# Notify all team members to re-clone or reset their branches
```

### 5. Verify Removal

```bash
# Check local history
git log -S "[REDACTED-API-KEY]" --all

# Check all files in history
git grep "[REDACTED-API-KEY]" $(git rev-list --all)

# Check working tree
grep -r "[REDACTED-API-KEY]" . --exclude-dir=.git

# All commands should return no results (except in security documentation)
```

## Prevention Measures

### Immediate Setup

1. **Install pre-commit hooks**:
```bash
pip install pre-commit detect-secrets
pre-commit install
```

2. **Create `.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

3. **Verify `.env` is in `.gitignore`**:
```bash
git check-ignore .env
echo ".env" >> .gitignore
```

4. **Create `.env.example` template**:
```bash
cp .env .env.example
# Edit .env.example to replace actual values with placeholders
git add .env.example
git commit -m "docs: add .env.example template"
```

### Long-term Measures

- Set up GitHub secret scanning
- Install gitleaks or trufflehog
- Regular security audits
- Team training on secret management

## Impact Assessment

### Potential Risks

1. **Unauthorized API Usage**: Attackers could use the key for API calls
2. **Quota Exhaustion**: Key could be used to exhaust your API quota
3. **Cost Impact**: If billing is enabled, unauthorized usage could incur costs
4. **Data Access**: Depending on key permissions, access to Google Cloud resources

### Mitigation

- [x] Key identified and documented
- [ ] **Key revoked** (DO THIS NOW)
- [ ] New key generated
- [ ] Git history cleaned
- [ ] Force pushed to remote (if applicable)
- [ ] Team notified
- [ ] Prevention measures implemented
- [ ] Monitoring set up for unusual activity

## Monitoring

After revoking the key, monitor:

1. **Google Cloud Console**: Check for unusual API activity
2. **Billing**: Watch for unexpected charges
3. **Logs**: Review access logs for the revoked key
4. **Quotas**: Check if quotas were exhausted

## Documentation Created

- ✅ `docs/how-to/security/REMOVE_EXPOSED_API_KEY.md` - Detailed removal instructions
- ✅ `docs/how-to/security/PREVENT_SECRET_EXPOSURE.md` - Prevention best practices
- ✅ `docs/how-to/security/SECURITY_INCIDENT_SUMMARY.md` - This file

## Next Steps

1. **IMMEDIATELY**: Revoke the exposed API key at https://aistudio.google.com/app/apikey
2. Generate new API key and store in `.env` (not in git)
3. Clean git history using git-filter-repo or BFG
4. Force push cleaned history (coordinate with team)
5. Verify key is completely removed
6. Set up pre-commit hooks for future prevention
7. Monitor for unauthorized usage

## Team Communication Template

```
Subject: URGENT - Security Incident: Exposed API Key

Team,

We discovered an exposed Google Gemini API key in our git repository.

IMMEDIATE ACTIONS REQUIRED:
1. The exposed key has been revoked
2. Git history is being cleaned
3. A force push will be done to feat/umap-calc branch
4. You will need to reset your local branch after the force push

After the force push is complete:
```bash
git fetch origin
git reset --hard origin/feat/umap-calc
```

Or simply re-clone the repository.

PREVENTION:
- Install pre-commit hooks: pip install pre-commit && pre-commit install
- Never commit .env files
- Use .env.example for templates
- Review git diff before every commit

Questions? Please reach out.
```

## References

- Detailed removal guide: `docs/how-to/security/REMOVE_EXPOSED_API_KEY.md`
- Prevention guide: `docs/how-to/security/PREVENT_SECRET_EXPOSURE.md`
- Google AI Studio: https://aistudio.google.com/app/apikey
- GitHub secret scanning: https://docs.github.com/en/code-security/secret-scanning

## Contact

For questions or assistance with this incident, contact your security team or repository administrator.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-13
**Status**: ACTIVE INCIDENT - REQUIRES IMMEDIATE ACTION
