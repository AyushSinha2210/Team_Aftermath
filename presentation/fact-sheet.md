# Slide 4: inspected template and architecture fact sheet

Source deck: `C:/Users/admin/Desktop/PRISM/Ariadne_Team_Aftermath.pptx`. Repository inspected: `C:/Users/admin/Desktop/PRISM/Abhilash Prism`, branch Abhilash. Evidence was read before drawing. VERIFIED means the stated, scoped claim is supported by the repository; it does not certify every deployment uses that component. No new benchmarks were run for this slide edit.

## Template inspection

- Canvas: 13.3333 × 7.5 inches, 12 slides. Slide 4 retains layout `OBJECT`, layout part `slideLayout2.xml`, and its original master.
- Title placeholder: `Google Shape;113;p4`, ID 113; x=0.9951, y=0.2915, width=11.5, height=1.4497 inches. The entire original title XML is retained, including inherited formatting. The rendered title is large black text, left aligned.
- Safe content bounds from the layout: x=0.917, y=1.997, width=11.5, height=4.759 inches. Reserved footer/date/slide-number band starts at y=6.951. Background is white (`lt1`).
- Major/minor theme fonts: Arial/Arial. Actual body cards on slides 2, 3, 5 and 6 use Aptos. Body sizes: slide 2: 14–16 pt and 23 pt section text; slide 3: 12.5/15 pt; slide 5: 12.5/13/15 pt; slide 6: 11.5/14/15 pt. Existing slide-4 labels: Aptos 12.5 pt bold. The unchanged title inherits its original template styling.
- Theme colors: dk1 000000, lt1 FFFFFF, dk2 44546A, lt2 E7E6E6; accent1 4472C4, accent2 ED7D31, accent3 A5A5A5, accent4 FFC000, accent5 5B9BD5, accent6 70AD47.
- Existing deck RGB palette: 12122B, 14142B, 1E1F30, 345BD2, 42A6A4, 5C5E70, 63637E, 672DBE, 6D28D9, D9D3F0, E1E2EB, F6F7FB, FFFFFF. Diagram accents: purple 672DBE, blue 345BD2, teal 42A6A4; all already occur in the deck. Theme references are used for white and neutral dark text. Accent fills use light brightness tints of these existing hues.
- Recurring motifs: white rounded cards, thin pale borders, narrow colored section bars, bold dark headings; typical outer margin about 0.9 inches. The new flat diagram retains the section bars and typography; its nodes use square corners and 1 pt borders as requested, without the old shadows.
- Original slide 4: 16 shapes, comprising the title, seven rounded flow nodes (including the final output), seven connectors, and one AST/versioning bullet text box; no images. Flow terminology: User Query → Query Intent & Syntax Density → Dense Vector Retrieval / Sparse Lexical Retrieval → Calibrated Hybrid Fusion → Confidence Cascade Router → Final calibrated ranked snippets. AST/BFS/PageRank/SHA-256 appear as a disconnected bullet list.
- Nearby terminology: slides 2/6 emphasize AST, function boundaries, MiniLM, RRF k=20 and SHA-256; slide 3 distinguishes vector retrieval, reranking and rebuild cost; slide 5 supplies the authTool-before-bluetoothTool example at settingsAgent.js:18–29.

## Claim table

All source paths below are relative to the inspected repository, except explicit slide references. Scope qualifications are part of the verified claim.

| Claim | Source | Status / treatment |
|---|---|---|
| all-MiniLM-L6-v2 model backbone | `ariadne/config.yaml`, model section; README Core Technical Pillars | VERIFIED — shown under dense retrieval |
| 384-dimensional embeddings | `ariadne/config.yaml`, embedding_dim; README model backbone | VERIFIED — shown as 384-d |
| CodeBM25Retriever exists | `ariadne/retrieval/code_bm25.py`; README Hybrid Retrieval | VERIFIED — available code-aware sparse component, described in fact sheet |
| CodeBM25Retriever is the default HybridPipeline sparse implementation | `ariadne/retrieval/pipeline.py` imports/constructs SparseRetriever | UNVERIFIED — contradicted by current wiring; diagram says Sparse BM25 |
| camelCase/snake_case tokenization exists in code-aware sparse component | `ariadne/retrieval/code_tokenizer.py`; `code_bm25.py` | VERIFIED — component capability; not attributed to the default sparse path |
| Hybrid RRF uses k=20 | `ariadne/config.yaml`, retrieval; `retrieval/pipeline.py`, retrieve() | VERIFIED — shown in fusion node |
| Margin Δ = score1 − score2 for ranked candidates, clamped at zero | `ariadne/reranking/cascade_router.py`, compute_margin() | VERIFIED — shown under router; dense_score has priority |
| Configured cascade threshold is 0.08 | `ariadne/config.yaml`, reranking.cascade; `reranking/checkpoint_eval.py`, Config E | VERIFIED — shown in diamond; constructor default is separately 0.05 |
| 58% fast-path in the 50-query Config E dense-cascade validation | `ariadne/eval/results/checkpoint_eval_20260930_120930.json`, config_e; `reranking/checkpoint_eval.py`, Config E dense candidate loop | VERIFIED — prominent badge explicitly scoped to dense validation |
| 58% fast-path after hybrid RRF, or in production | Same evaluation evaluates dense candidates for Config E | UNVERIFIED — omitted; cascade branches from dense, hybrid has a separate output path |
| Cross-encoder handles low-margin cascade escalation | `ariadne/reranking/cascade_router.py`, route(); `cross_encoder.py` | VERIFIED — NO branch |
| SHA-256 function-level content hashing | `ariadne/versioning/content_hash.py`; README Incremental Indexer | VERIFIED — shown in offline lane |
| Incremental indexing reuses unchanged embeddings and updates changed/added chunks | `ariadne/versioning/incremental_index.py`, update(); `incremental_sparse.py` | VERIFIED — shown as incremental index update |
| Tree-Sitter parses functions / AST | `ariadne/reranking/structural/ast_parser.py`, parse_js_file()/parse_repo(); `ariadne/retrieval/agentic_search.py` | VERIFIED — offline parsing and structural path |
| Call graph builder exists | `ariadne/reranking/structural/call_graph.py`, build_call_graph(); `retrieval/agentic_search.py`, index_repo() | VERIFIED — offline graph output and AST module label |
| Bidirectional BFS exists | `ariadne/reranking/structural/call_graph.py`, resolve_call_path_bidirectional() | VERIFIED — module capability caption |
| PageRank exists | `ariadne/reranking/structural/call_graph.py`, compute_call_graph_pagerank() | VERIFIED — module capability caption |
| The ordered-tool example is answered with AST call-order analysis | `ariadne/retrieval/agentic_search.py`, _find_structural_query() | VERIFIED — shown as a separate structural query path |
| Calibration exists and is evaluated in Config C/D | `ariadne/reranking/calibration.py`; `reranking/checkpoint_eval.py`, Config C/D | VERIFIED — separate caption, not a mandatory cascade stage |
| Cascade automatically executes cross-encoder → AST → calibration | `cascade_router.py`, route() executes only rerank_fn; AST query engine is separate | UNVERIFIED — serial chain omitted |
| Query embedding cache exists | `ariadne/retrieval/dense_retriever.py`, query_cache | VERIFIED — not needed on slide |
| Query cache uses LRU eviction | README says LRU; dense_retriever.py does not refresh hits and evicts first inserted key | UNVERIFIED — omitted; observed policy is insertion-order eviction |
| “which files call tool authTool before bluetoothTool?” returns agents/settingsAgent.js:18–29 | Slide 5; `ariadne/retrieval/agentic_search.py`; `ariadne/data/voice_assistant_js/agents/settingsAgent.js`, openBluetoothSettings; checked-in theme1 benchmark report | VERIFIED — shown as one matching result, not the only match |
| Lines 18–29 call verifySession before navigateToDeeplink | `ariadne/data/voice_assistant_js/agents/settingsAgent.js`, lines 20/27 | VERIFIED — mini trace uses these methods |
| verifySession → scanDevices belongs to lines 18–29 | Same file, scanDevices is in pairBluetoothDevice at lines 34–42 | UNVERIFIED — corrected to navigateToDeeplink |
| authTool.verifySession directly calls bluetoothTool.scanDevices | Same file: the methods are sibling calls made by an agent | UNVERIFIED — arrows represent ordered calls from a common caller, not a direct tool-to-tool call edge |
| An LLM generates the final answer in this flow | No evidence in these retrieval implementations | UNVERIFIED — output is snippets / file-line locations |

## Planned deviations for factual accuracy and template fidelity

1. Show dense-cascade, hybrid retrieval, and structural analysis as distinct implemented paths. Do not invent the requested automatic hybrid → router → AST → calibration chain. The 58% badge is limited to the measured 50-query dense validation.
2. Use Sparse BM25 in the main pipeline; keep CodeBM25 and its code-aware tokenizer as verified component facts in this table, without implying default integration.
3. Correct the mini example to verifySession then navigateToDeeplink, with a common caller and ordered-call arrows. Omit the unsupported scanDevices edge and LRU label.
4. Use Aptos for new labels because it matches the actual neighboring slides; Arial remains the unchanged theme. Consolas is introduced only for filename captions, explicitly allowed by the brief.
5. Fit the computed 12-column grid to the actual content bounds. Allocate more space to the readable online branches and use a narrow bottom example strip rather than rigid 30/55/15 proportions. Five offline boxes summarize three outputs inside the final box.
6. Accent colors are existing deck RGB values because its purple/teal are not theme accent entries. White and dark neutral use theme references. Light tints add no new base hues.

## Final validation

- Three render/review iterations completed using LibreOffice → PDF → PNG. The final slide and contact sheet were visually inspected.
- Exactly two package parts changed: `ppt/slides/slide4.xml` and `ppt/notesSlides/notesSlide4.xml`. All other package parts retain identical uncompressed bytes, including the other eleven slides, relationships, themes, masters, layouts and media. The ZIP container is necessarily rewritten.
- The original title XML is identical; slide 4 still uses `OBJECT`. The original input was never overwritten.
- Sixteen diagram nodes plus a small example strip. Native lane groups and editable native shapes; 58 connector segments are glued at both ends. Invisible native bend anchors ensure explicit orthogonal routes survive LibreOffice rendering.
- Layout checks found no node overlap, no shape outside the template content area, and no connector crossing a text label. Font-metric estimates using installed Aptos/Consolas found no overflow; the final render showed no clipping. All labels are at least 11 pt; six filename captions use Consolas at 9–9.5 pt. Three speaker-note lines were added.
- New labels match the neighboring Aptos cards. Consolas, explicitly requested for filenames, is the only added font. No new base color hues were introduced. Dark text on the near-white tints has contrast above 4.5:1.
- Slides 3 and 5 render with identical pixels before and after, at the same resolution. The contact sheet shows their fonts, margins, palette and title treatment alongside the edited slide.
- The diagram depicts verified components and separately evidenced paths, rather than claiming one fully integrated execution chain. In particular, code-aware BM25 tokenization and calibration are not silently attributed to the default hybrid/cascade implementation.

Files in this folder: the edited PPTX, before/after PNGs, the template contact sheet, this fact sheet, and a machine-readable `verification.json`. Build/inspection scripts and intermediate renders remain in the ignored `.delivery/slide4` folder in the same checkout.
