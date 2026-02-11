# Git Fork Workflow - Quick Reference

## Current Setup ✅

- **origin**: git@github.com:KennethWang1221/minimind.git (your fork)
- **upstream**: https://github.com/jingyaogong/minimind.git (original repo)
- **master branch**: tracking origin/master

## Common Commands

### Sync Latest Changes from Upstream

```bash
# Fetch latest from original repo
git fetch upstream

# Merge into your master
git checkout master
git merge upstream/master

# Push to your fork
git push origin master
```

### Working on Your Customizations

**Option 1: Work on master directly (simple)**
```bash
# Make your changes
git add .
git commit -m "Your commit message"
git push origin master
```

**Option 2: Use feature branches (recommended)**
```bash
# Create a new branch for your work
git checkout -b my-feature

# Make changes and commit
git add .
git commit -m "Add my feature"
git push origin my-feature

# When ready, merge to master
git checkout master
git merge my-feature
git push origin master
```

### Check Status

```bash
# See what remotes are configured
git remote -v

# Check current branch and status
git status

# See commits in upstream that you don't have
git fetch upstream
git log master..upstream/master --oneline

# See commits you have that upstream doesn't
git log upstream/master..master --oneline
```

### Handling Merge Conflicts

If conflicts occur when merging upstream:
```bash
# Git will mark conflicts in files
# Edit the files to resolve
# Then:
git add <resolved-files>
git commit
git push origin master
```

## Your Untracked Files (Safe)

These files are not tracked and won't be affected by pulls:
- `.envrc`
- `MiniMind2-PyTorch/`
- `MiniMind2/`
- `dataset/` files (*.jsonl)
- `minimind_dataset/`
- `trainer/__pycache__/`

Consider adding them to `.gitignore` if you don't want to track them.

## Workflow Summary

1. **Sync from upstream**: `git fetch upstream && git merge upstream/master`
2. **Make your changes**: Work on feature branches or master
3. **Push to your fork**: `git push origin <branch>`
4. **Repeat**: Keep your fork updated regularly

## Pro Tips

- Fetch from upstream regularly to stay updated
- Use branches for experimental features
- Keep master stable and in sync with upstream
- Your fork is backed up on GitHub - safe to experiment!
