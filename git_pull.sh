#!/usr/bin/env bash
set -euo pipefail

echo "Fetching remotes..."
git fetch origin
git fetch upstream

echo "Syncing master with upstream/master..."
git switch master
git reset --hard upstream/master
git push --force-with-lease origin master

echo "Rebasing feat onto master..."
git switch feat

if git rebase master; then
  git push --force-with-lease origin feat
  echo "Done: master is synced and feat is updated."
else
  echo "Rebase stopped due to conflicts on feat."
  echo "Resolve conflicts, then run:"
  echo "  git add <files>"
  echo "  git rebase --continue"
  echo "  git push --force-with-lease origin feat"
  exit 1
fi
