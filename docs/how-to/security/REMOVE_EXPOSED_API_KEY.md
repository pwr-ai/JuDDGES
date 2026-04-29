# How to Remove Exposed GEMINI_API_KEY from Git History

## CRITICAL SECURITY INCIDENT

**Status**: ACTIVE SECURITY BREACH
**Exposed Key**: `[REDACTED-API-KEY]`
**Location**: `docs/EXTRACTION_STATUS.md` (lines 64-65)
**First Commit**: `c123bd8` (feat(extraction): add Gemini LLM extraction chain for legal documents)
**Affected Commits**: 9 commits from c123bd8 to f54d631

## IMMEDIATE ACTIONS (DO THESE FIRST)

### Step 1: Revoke the Exposed API Key

**CRITICAL**: Do this BEFORE cleaning git history. Even after removing from git, the key may already be compromised.

1. Go to [Google AI Studio API Keys](https://aistudio.google.com/app/apikey)
2. Find the API key ending in `...t-0K4`
3. Click "Delete" or "Revoke"
4. Confirm deletion

Alternative via Google Cloud Console:
```bash
# List all API keys
gcloud alpha services api-keys list

# Delete the specific key (replace KEY_ID with actual ID)
gcloud alpha services api-keys delete KEY_ID
```

### Step 2: Generate New API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Select or create a Google Cloud project
4. Copy the new key
5. Store it securely in your local `.env` file (NOT in git)

```bash
# Add to .env file (ensure .env is in .gitignore)
GOOGLE_API_KEY=your-new-key-here
GEMINI_API_KEY=your-new-key-here
```

### Step 3: Verify .env is in .gitignore

```bash
# Check if .env is properly ignored
git check-ignore .env

# If not, add it
echo ".env" >> .gitignore
git add .gitignore
git commit -m "chore: ensure .env is in .gitignore"
```

## CLEAN GIT HISTORY

### Option A: Using git-filter-repo (RECOMMENDED)

This is the most efficient and safe method for rewriting git history.

#### Install git-filter-repo

```bash
# On Ubuntu/Debian
sudo apt install git-filter-repo

# On macOS
brew install git-filter-repo

# Using pip
pip install git-filter-repo
```

#### Remove the exposed key

```bash
# 1. Make a backup of your repository first
cd <path-to-JuDDGES>
tar -czf ../JuDDGES-backup-$(date +%Y%m%d-%H%M%S).tar.gz .

# 2. Create expressions file to replace the key
cat > /tmp/replace-expressions.txt << 'EOF'
GEMINI_API_KEY=[REDACTED-API-KEY]==>GEMINI_API_KEY=[REDACTED]
GOOGLE_API_KEY=[REDACTED-API-KEY]==>GOOGLE_API_KEY=[REDACTED]
[REDACTED-API-KEY]==>[REDACTED-API-KEY]
EOF

# 3. Run git-filter-repo to replace the key across all history
git filter-repo --replace-text /tmp/replace-expressions.txt --force

# 4. Verify the key is removed
git log -S "[REDACTED-API-KEY]" --all
# Should return no results

# 5. Clean up
rm /tmp/replace-expressions.txt
```

### Option B: Using BFG Repo-Cleaner (ALTERNATIVE)

BFG is faster than git-filter-branch but requires Java.

#### Install BFG

```bash
# Download BFG
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -O bfg.jar

# Or on macOS
brew install bfg
```

#### Remove the exposed key

```bash
# 1. Make a backup
cd <path-to-JuDDGES>
tar -czf ../JuDDGES-backup-$(date +%Y%m%d-%H%M%S).tar.gz .

# 2. Create a file with text to replace
echo "[REDACTED-API-KEY]" > /tmp/passwords.txt

# 3. Run BFG
java -jar bfg.jar --replace-text /tmp/passwords.txt .

# 4. Clean up with git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Verify
git log -S "[REDACTED-API-KEY]" --all

# 6. Clean up
rm /tmp/passwords.txt
```

### Option C: Interactive Rebase (ONLY FOR RECENT COMMITS)

**WARNING**: Only use if the key was added in the last few commits and hasn't been pushed to shared branches.

```bash
# 1. Find the first commit with the key
git log -S "[REDACTED-API-KEY]" --all --oneline
# Result: c123bd8 feat(extraction): add Gemini LLM extraction chain

# 2. Start interactive rebase from the parent of that commit
git rebase -i c123bd8~1

# 3. Mark the commit for 'edit' instead of 'pick'
# 4. When the rebase pauses, edit the file
sed -i 's/[REDACTED-API-KEY]/[REDACTED-API-KEY]/g' docs/EXTRACTION_STATUS.md

# 5. Amend the commit
git add docs/EXTRACTION_STATUS.md
git commit --amend --no-edit

# 6. Continue the rebase
git rebase --continue

# 7. Handle any conflicts that arise
```

## FIX CURRENT BRANCH

Remove the key from the current working branch:

```bash
# Edit the file to remove the exposed key
cd <path-to-JuDDGES>

# Remove the key from EXTRACTION_STATUS.md
sed -i 's/GEMINI_API_KEY=[REDACTED-API-KEY]/GEMINI_API_KEY=[REDACTED]/g' docs/EXTRACTION_STATUS.md
sed -i 's/GOOGLE_API_KEY=[REDACTED-API-KEY]/GOOGLE_API_KEY=[REDACTED]/g' docs/EXTRACTION_STATUS.md

# Verify the change
git diff docs/EXTRACTION_STATUS.md

# Commit the fix
git add docs/EXTRACTION_STATUS.md
git commit -m "security: remove exposed GEMINI_API_KEY from documentation"
```

## FORCE PUSH TO REMOTE

**WARNING**: This will rewrite history on the remote repository. Coordinate with your team first.

### If working on a feature branch

```bash
# Force push the cleaned branch
git push origin feat/umap-calc --force-with-lease

# Or force push all branches
git push origin --all --force-with-lease
```

### If the key reached the main/master branch

```bash
# CRITICAL: Notify all team members BEFORE doing this
# They will need to re-clone or reset their local copies

# Force push master
git checkout master
git push origin master --force-with-lease

# Force push all branches and tags
git push origin --all --force-with-lease
git push origin --tags --force-with-lease
```

### Important Notes About Force Push

1. **Communicate with team**: Everyone needs to know history is being rewritten
2. **Timing**: Do this during low-activity periods
3. **Backup**: Ensure all team members have local backups before force pushing
4. **Re-clone**: Team members should re-clone the repository after the force push

## VERIFY KEY REMOVAL

### Verify locally

```bash
# Search for the key in all history
git log -S "[REDACTED-API-KEY]" --all --oneline
# Should return: (empty)

# Search in all files across all commits
git grep "[REDACTED-API-KEY]" $(git rev-list --all)
# Should return: (empty)

# Check current working tree
grep -r "[REDACTED-API-KEY]" .
# Should return: (empty)
```

### Verify on remote (after force push)

```bash
# Clone a fresh copy
cd /tmp
git clone https://github.com/yourusername/JuDDGES.git JuDDGES-verify
cd JuDDGES-verify

# Search the fresh clone
git log -S "[REDACTED-API-KEY]" --all
git grep "[REDACTED-API-KEY]" $(git rev-list --all)

# Clean up
cd /tmp && rm -rf JuDDGES-verify
```

## NOTIFY TEAM MEMBERS

After force pushing, all team members must update their local repositories:

```bash
# Option 1: Reset local branch (DESTRUCTIVE - loses local changes)
git fetch origin
git reset --hard origin/feat/umap-calc

# Option 2: Re-clone the repository
cd ~/github/legal-ai
mv JuDDGES JuDDGES-old-backup
git clone <repository-url> JuDDGES
cd JuDDGES

# Option 3: Rebase local changes (if you have unpushed work)
git fetch origin
git rebase origin/feat/umap-calc
```

## PREVENT FUTURE EXPOSURE

### Add pre-commit hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: detect-private-key
      - id: check-added-large-files

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: package.lock.json
```

Install and set up:

```bash
pip install pre-commit detect-secrets
pre-commit install

# Generate baseline
detect-secrets scan > .secrets.baseline

# Test
pre-commit run --all-files
```

### Update .gitignore

Add these patterns to `.gitignore`:

```gitignore
# Environment files
.env
.env.local
.env.*.local
*.env

# API keys and secrets
*secret*
*apikey*
*api-key*
*api_key*
credentials.json
service-account*.json

# Common secret files
.secrets
secrets/
private/
```

### Use environment variables correctly

Create a `.env.example` template:

```bash
# .env.example - Commit this file
GOOGLE_API_KEY=your-google-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
```

Then in documentation, reference the example:

```markdown
## Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your actual API keys
3. Never commit the `.env` file
```

### Use secret management tools

Consider using:

1. **GitHub Secrets** (for CI/CD)
2. **HashiCorp Vault** (for production)
3. **AWS Secrets Manager** / **Google Secret Manager**
4. **git-crypt** (for encrypted secrets in git)

## MONITORING

### Set up secret scanning

1. **GitHub Secret Scanning**: Enabled automatically for public repos
2. **GitGuardian**: https://www.gitguardian.com/
3. **TruffleHog**: https://github.com/trufflesecurity/trufflehog

### Monitor API key usage

```bash
# Check if the old key was used maliciously
# Monitor Google Cloud Console for unexpected API calls
# Check billing for unusual charges
```

## INCIDENT RESPONSE CHECKLIST

- [ ] Revoke exposed API key immediately
- [ ] Generate new API key
- [ ] Update .env file locally (do NOT commit)
- [ ] Verify .env is in .gitignore
- [ ] Backup repository
- [ ] Clean git history (git-filter-repo/BFG)
- [ ] Verify key removed from all commits
- [ ] Remove key from current branch
- [ ] Force push to remote
- [ ] Notify all team members
- [ ] Verify key removed from remote
- [ ] Set up pre-commit hooks
- [ ] Update .gitignore
- [ ] Create .env.example
- [ ] Enable secret scanning
- [ ] Monitor API usage for abuse
- [ ] Document incident and prevention measures

## SUMMARY OF EXPOSED KEY

```
Key: [REDACTED-API-KEY]
File: docs/EXTRACTION_STATUS.md
Lines: 64-65
First commit: c123bd8 (2025-01-XX)
Branch: feat/umap-calc
Affected commits: 9 commits
```

## REFERENCES

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [git-filter-repo](https://github.com/newren/git-filter-repo)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- [Google Cloud: Managing API Keys](https://cloud.google.com/docs/authentication/api-keys)
- [OWASP: Secret Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)

## SUPPORT

For questions or issues with this process:

1. Review GitHub documentation on removing sensitive data
2. Consult your team's security officer
3. Contact Google Cloud Support for API key concerns
4. Review commit history: `git log --oneline --graph --all`
