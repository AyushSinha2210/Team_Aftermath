# Frontend validation record

Date: 30 September 2026. Checkout: Abhilash Prism. Branch: Abhilash.

## Results

| Check | Observed result |
| --- | --- |
| Python API contract and adapter tests | 24 passed; 6 upstream Tree-sitter deprecation warnings |
| Frontend protocol and state tests | 21 passed |
| Edge browser tests with real CPU API enabled | 14 passed |
| Automated accessibility scan | Zero WCAG A/AA violations in the reduced-motion page scan |
| Production build | Passed; JS 381.11 kB raw, 121.80 kB gzip; CSS 50.17 kB raw, 23.69 kB gzip |
| npm dependency audit | Zero reported vulnerabilities |
| UI source audit | Passed for color tokens, no emoji/shadows, transform/opacity animation properties |
| Layout checks | No page overflow at 1280, 1366 and 390 pixels; diagrams/tables use contained scrolling on narrow screens |
| Live query | Real fine-tuned CPU response; authTool.js returned with exact lines; no fallback notice |

## Validation scope

Browser tests cover input validation, Enter submission, double-click prevention,
loading, empty results, malformed payloads, structured server errors, network
failure, reverse-proxy failure, recorded-example labeling, arbitrary offline
queries, eight-second timeout, retry availability, reduced motion, keyboard
skip links, intro skip/handoff, font loading, accessibility and responsive
screenshots. Dense and hybrid adapter tests use deterministic vectors; the
browser's live query uses the actual fine-tuned model on CPU.

Existing Python 3.11 inference dependencies were reused for this machine's
checks without changing the earlier checkout. The missing Tree-sitter wheels
were installed into this checkout's ignored .cache/api-deps. Reproducible fresh
environment instructions and API requirements are included in frontend/README.md.

The official MTEB test and small validation results were extracted from existing
JSON files; no new benchmark run or performance improvement is claimed. The
new CI workflow is configured locally and has not run on GitHub while push
authentication is unresolved. A steady 60 fps on the presentation hardware
has not been measured. Projector and final demonstration hardware still need
an on-device check; the introduction can be disabled immediately.

Rendered browser images were visually inspected under reports/qa. That
directory, dependency installations, model caches and delivery logs are ignored.
The work report is a final tracked artifact.
