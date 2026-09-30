#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .delivery
verify() {
  [[ "$(git branch --show-current)" == Abhilash ]]
  [[ "$(git remote get-url origin)" == https://github.com/abhi-s99/Team_Aftermath.git ]]
  [[ "$(git config user.name)" == abhi-s99 ]]
  [[ "$(git config user.email)" == abhilashsingh2005@gmail.com ]]
}
can_push() {
  [[ "$(gh api user --jq .login 2>/dev/null || true)" == abhi-s99 ]]
}
flush() {
  can_push || { echo 'Push deferred: GitHub must authenticate as abhi-s99.'; return 0; }
  [[ -f .delivery/pending ]] || return 0
  # Use the same authenticated account for Git transport as for the API check.
  git config --local --replace-all credential.https://github.com.helper ''
  git config --local --add credential.https://github.com.helper '!gh auth git-credential'
  # A queue may survive a manual push; never try to move the remote backwards.
  git fetch origin Abhilash
  while [[ -s .delivery/pending ]]; do
    read -r sha < .delivery/pending
    if git merge-base --is-ancestor "$sha" origin/Abhilash; then
      echo "Already on origin/Abhilash: $sha"
    else
      git push origin "$sha:refs/heads/Abhilash"
    fi
    echo "$sha" >> .delivery/pushed
    tail -n +2 .delivery/pending > .delivery/pending.next
    mv .delivery/pending.next .delivery/pending
  done
}
verify
if [[ "${1:-}" == --flush ]]; then
  flush
else
  message="$1"; shift
  git add -- "$@"
  git diff --cached --check
  git commit -m "$message"
  git rev-parse HEAD >> .delivery/pending
  flush
fi
