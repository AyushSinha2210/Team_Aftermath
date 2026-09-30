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
  while read -r sha; do
    [[ -n "$sha" ]] || continue
    git push origin "$sha:refs/heads/Abhilash"
    echo "$sha" >> .delivery/pushed
  done < .delivery/pending
  : > .delivery/pending
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
