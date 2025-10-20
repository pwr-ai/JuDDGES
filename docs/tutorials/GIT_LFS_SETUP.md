# Git LFS Setup for Large Model Files

## Files to Track

- `models/umap/umap_model_LegalDocuments.pkl` (477MB)
- `umap_coords/LegalDocuments_coords.parquet` (13MB)

## Setup (Run on HOST machine, not in Docker)

### 1. Install Git LFS

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs
```

### 2. Initialize and Add Files

```bash
# Initialize Git LFS
git lfs install

# Verify .gitattributes has LFS rules
cat .gitattributes
# Should show:
# *.pkl filter=lfs diff=lfs merge=lfs -text
# *.parquet filter=lfs diff=lfs merge=lfs -text

# Add files
git add models/umap/umap_model_LegalDocuments.pkl
git add umap_coords/LegalDocuments_coords.parquet

# Verify LFS tracking
git lfs ls-files

# Commit
git commit -m "Add UMAP model and coordinates to LFS"
```

## Troubleshooting

### Files not tracking with LFS

```bash
# Re-track
git rm --cached models/umap/umap_model_LegalDocuments.pkl
git add models/umap/umap_model_LegalDocuments.pkl
git lfs ls-files  # Verify
```

### Storage Limits

- **GitHub**: 1GB free storage + 1GB/month bandwidth
- **GitLab**: 10GB per repository
- Consider DVC for larger models
