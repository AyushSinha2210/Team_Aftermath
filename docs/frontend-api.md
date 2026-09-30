# Ariadne browser API contract

The repository has a Streamlit UI, not a browser JSON API. Its `search` function
returns status, results and dense cosine scores. The new loopback HTTP bridge
uses Person D's `IncrementalIndex` and shared encoder directly, avoiding a
Streamlit runtime dependency. Dense mode follows the existing recommended
configuration; optional hybrid mode calls `HybridPipeline.retrieve`.

`GET /api/health` returns `{ "status": "ready", "mode": "dense",
"documents": 20, "api_version": 1 }` (document count is determined at runtime).

`POST /api/search`, JSON `{ "query": "verify user session", "top_k": 5 }`:

```json
{
  "api_version": 1,
  "status": "success",
  "mode": "dense",
  "query": "verify user session",
  "elapsed_ms": 10.2,
  "results": [{
    "id": "tools/authTool.js:5:verifySession",
    "file": "tools/authTool.js",
    "start_line": 5,
    "end_line": 8,
    "title": "verifySession",
    "code": "async function verifySession(sessionId) { ... }",
    "score": 0.5,
    "dense_score": 0.5,
    "sparse_score": null,
    "sources": ["dense"]
  }]
}
```

The numbers and shortened code above illustrate the schema, not measurements.
Empty or low-confidence results use `status: "empty"` and `results: []`.
Scores are cosine similarity or RRF ranking values, not probabilities. File
and line coordinates come from Tree-sitter function boundaries in the actual
JavaScript voice-assistant sample. This demo searches that sample, not the
CoIR benchmark corpus. Reranking is never implied by the pipeline artwork.

Invalid JSON, invalid query length/type, boolean or out-of-range top_k, large
request bodies and unknown routes return structured errors. Model failures
and busy inference return generic errors; details stay in server logs. Bind
to 127.0.0.1 and use Vite's same-origin /api proxy. There is no external CORS
allowlist or unrestricted repository-read endpoint.
