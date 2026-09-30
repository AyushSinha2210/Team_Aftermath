# Ariadne frontend

React, Vite, Tailwind and Framer Motion frontend implemented by Abhilash.
Self-hosted Fraunces, Manrope and JetBrains Mono fonts work without Google Fonts.

## Run with Git Bash

From the repository root, use Python 3.11 for the existing PyTorch pins:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# Linux/macOS: source .venv/bin/activate
python -m pip install -r ariadne/frontend_api/requirements.txt
python -m ariadne.frontend_api.server
```

In another Bash terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The JSON API listens on 127.0.0.1:8765 and
Vite proxies /api for both dev and preview. `--mode hybrid` enables real
BM25/RRF attribution; dense is the recommended default. The demo searches
15 functions in the checked-in JavaScript voice-assistant sample.

`/design-system` displays all tokens and interaction states. `/?intro=off`
skips the laptop immediately. `VITE_DISABLE_INTRO=true npm run dev` disables
it for a presentation; rebuild production assets after changing build-time
flags. OS reduced motion or a low-power device also skips the introduction.

## Validate

```bash
cd frontend
npm test
npm run build
node scripts/audit-ui.mjs
npm audit
npx playwright install chromium
npm run test:e2e
# Use installed Edge on Windows if desired:
PLAYWRIGHT_CHANNEL=msedge RUN_LIVE_API=true npm run test:e2e
```

The default browser suite runs controlled failure scenarios and skips the
real-model query. RUN_LIVE_API=true adds the live query and requires the API
above. Browser screenshots stay under ignored reports/qa. API tests run from
the repository root: `python -m pytest ariadne/frontend_api -q`.

## Honest evidence and fallback

The large numerals show the checked-in official MTEB **test** result:
NDCG@10 0.15916 and MRR@10 0.1371307363. The separate ablation table shows
the 50-query / 500-candidate **validation** experiment, not the same test.
Latency cells say “Not recorded” because that result file has no timing data.
No benchmark was rerun during frontend development. Refresh extracted
evidence with `node scripts/sync-evidence.mjs`.

The pipeline artwork shows available stages. The live mini pipeline and tags
reflect actual dense or hybrid execution; reranking is not enabled. Scores
are raw cosine similarity, BM25 scores or RRF values, not probabilities.

Network and proxy outages may show real pre-recorded responses for the exact
three example questions, visibly labeled as recorded. Arbitrary offline
queries, timeouts, malformed responses and structured API failures show an
inline error and retry. They never borrow another example's results. Refresh
the recordings from a running API with `python scripts/record-examples.py`.

## Incremental delivery

Repository-local author: abhi-s99 / abhilashsingh2005@gmail.com.
Remote: https://github.com/abhi-s99/Team_Aftermath.git. Branch: Abhilash.
`bash scripts/deliver.sh "message" path...` commits a completed unit and
pushes it after verifying the author, branch, remote and authenticated account.
If authentication is unavailable, it saves an ordered local queue.

After signing in as abhi-s99 with GitHub CLI:

```bash
gh auth login --hostname github.com --git-protocol https --web
bash scripts/deliver.sh --flush
```

The flush pushes queued commits in order and records progress after each
successful push. It does not force-push or modify the submission release tag.

The three-page work report is in the repository root as `work done.docx`.
