# Kitfind

> Last updated: 2026-06-05

A searchable index of AI coding agent skills, deployed as a GitHub Pages static site, consumable by Kitout.

Kitfind is a companion to Kitout. Kitout loads skills into your session. Kitfind helps you find which skills you need in the first place.

## Repository layout

```
/
├── kitfind.toml              # Curated repo list (TOML, 6 repos)
├── schema.json               # index.json schema (JSON Schema draft-07)
├── pyproject.toml            # Python dependencies + mypy config
├── .gitignore
├── AGENTS.md                 # This file — architecture & implementation docs
├── PLAN.md                   # Status, phases, followup thoughts
├── README.md                 # Usage, badges, dev commands
├── assets/
│   └── kitfind-logo.png      # Site logo
├── tools/
│   ├── build.py              # Aggregator + site generator (Python, mypy-annotated)
│   ├── validate.py           # Structural validator (index.json, embeddings, model, template JS)
│   ├── search.js             # Pure JS search functions (CommonJS, shared, tested)
│   ├── templates/
│   │   └── index.html        # Self-contained HTML (search.js inlined at build)
│   └── tests/
│       └── search.test.mjs   # 26 Node.js tests
├── site/                     # Build output — deployed to gh-pages
│   ├── index.html            # Rendered site
│   ├── index.json            # Search blob (~3.4 MB)
│   ├── index.tfidf           # TF-IDF sparse vectors (~1.3 MB)
│   ├── index.embeddings      # Uint8-quantized 384-dim vectors (~808 KB)
│   └── model/
│       ├── onnx/model_quantized.onnx  # All-MiniLM-L6-v2 (~22 MB)
│       ├── vocab.json        # WordPiece vocab extracted from tokenizer
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
├── .github/
│   └── workflows/
│       ├── ci.yml            # push/PR: build + validate + mypy + JS tests + preview deploy
│       └── deploy.yml        # push to main: build + deploy to gh-pages
└── kitfind.lock              # Resolved commit SHAs per build (audit trail)
```

## Data pipeline

```
kitfind.toml ──> tools/build.py ──> site/ ──> gh-pages branch
                  │
                  ├─ clone each repo (shallow, depth=1, pinned ref)
                  ├─ discover SKILL.md files
                  ├─ parse YAML frontmatter (name, description, triggers, tags, domain)
                  ├─ fallback: regex extraction for malformed YAML
                  ├─ enrich with companion meta.json
                  ├─ deduplicate by (source.url, name)
                  ├─ classify domain via keyword matching (72 domains)
                  ├─ compute TF-IDF sparse vectors
                  ├─ compute BERT embeddings (ONNX Runtime, no torch)
                  ├─ inline tools/search.js into template
                  └─ write site/index.html, index.json, index.tfidf, index.embeddings, model/
```

### Build steps (build.py)

| Step | Function | Description |
|------|----------|-------------|
| Config | `load_config()` | Reads `kitfind.toml` |
| Clone | `git_clone(url, ref, target)` | Shallow clone at pinned ref, returns commit SHA |
| Parse | `parse_yaml_frontmatter(text, filepath)` | Strict YAML first, regex fallback |
| Fallback | `_extract_fields(raw)` | Regex extraction for malformed YAML frontmatter |
| Discover | `discover_skills(repo_dir)` | Finds all SKILL.md, parses, enriches with meta.json |
| Classify | `classify_domain(...)` | Keyword-based domain classifier (72 domains) |
| Keywords | `extract_keywords(text)` | Derives triggers from description, filters stop words |
| Model | `_ensure_model(model_dir)` | Downloads `Xenova/all-MiniLM-L6-v2` ONNX + configs via huggingface_hub |
| Vocab | `_extract_vocab(model_dir)` | Extracts flat vocab.json from tokenizer.json |
| Embed | `_pool_and_normalize(hidden, mask)` | Mean pool + L2 normalize transformer output |
| Embed | `compute_embeddings(skills, model_path)` | ONNX Runtime inference on all skills |
| Embed | `save_embeddings(embeddings, path)` | Writes uint8-quantized binary sidecar |
| TF-IDF | `_tokenize_tfidf(text)` | Tokenizer matching client-side JS stop word list |
| TF-IDF | `compute_tfidf(skills)` | Builds vocab, IDF weights, sparse L2-normalized vectors |
| TF-IDF | `save_tfidf(data, path)` | Compact JSON sidecar |
| Site | `build_site(skills, stats, sources)` | Generates HTML, inlines search.js at `<!-- SEARCH_JS -->` marker |

## Blob schema

### index.json — Search blob

`site/index.json` follows `schema.json`. Top-level structure:

```jsonc
{
  "$schema": "https://raw.githubusercontent.com/gliptak/kitfind/main/schema.json",
  "version": 1,
  "generated_at": "...",
  "catalog": {
    "sources": [{ "url": "...", "ref": "main", "commit": "...", "skill_count": N }],
    "stats": { "total_skills": 2150, "total_sources": 6, "by_domain": { ... } }
  },
  "skills": [
    {
      "id": "repo/path/name",
      "name": "skill-name",
      "source": { "url": "...", "ref": "main", "path": "skills/..." },
      "description": "...",
      "triggers": ["tdd", "test", ...],
      "harnesses": ["claude-code"],
      "domain": "testing",
      "tags": [],
      "install_hint": "kitout install owner/repo"
    }
  ]
}
```

### index.tfidf — TF-IDF ranking vectors

Sparse TF-IDF vectors for client-side cosine similarity ranking. Each skill's vector is unit-normalized. Typical skill has 20-100 non-zero terms.

```jsonc
{
  "version": 1,
  "vocab": ["debug", "python", "async", "testing", ...],  // ~7,365 terms
  "skills": [
    { "indices": [0, 5, 12], "values": [0.31, 0.12, 0.08] },  // unit-normalized
    ...
  ]
}
```

### index.embeddings — Semantic embedding sidecar

Binary format: `int32[3]` header (magic `0x4B4650`, dim=384, count=N) + `float32 mins[384]` + `float32 scales[384]` + `uint8 data[N x 384]`. Original float32 vectors recovered via `float_val = uint8_val * scale + min`. All vectors L2-normalized. Built with ONNX Runtime (no torch).

### model/ — ONNX Runtime Web model

The `Xenova/all-MiniLM-L6-v2` quantized ONNX model deployed as a static asset. Model artifact structure preserved exactly as HuggingFace repo. Client-side JS tokenizer uses extracted `vocab.json`. No HuggingFace API calls — `ort.min.js` from CDN handles inference.

## Search modes

Three search paths, selected automatically. All results rendered (no 200-card cap).

| Path | Trigger | Scoring | Term boosting |
|------|---------|---------|---------------|
| **Semantic (BERT)** | query >= 5 chars, model ready | Cosine sim against 384-dim ONNX embeddings (all skills scored, no AND filter) | `score *= (1.0 + 0.5 × matchingTerms)` |
| **Keyword (TF-IDF)** | query < 5 chars | AND word-boundary filter + sparse TF-IDF cosine sim | N/A (AND filter already filters to matching terms) |
| **Fallback** | model not ready / no query | AND word-boundary filter + alphabetical sort | N/A |

### Term boosting

Configurable `TERM_BOOST = 0.5` per exact query term appearing in the skill's name, description, triggers, or tags. Standard Elasticsearch-style `function_score` pattern. For a query with N terms:

```
finalScore = semanticScore × (1.0 + 0.5 × matchingTermCount)
```

A skill matching all query terms gets a strong relevance multiplier without hard cutoff filtering.

### AND filter

For TF-IDF and fallback paths: word-boundary regex `\bterm\b` to avoid substring false positives (e.g., "cto" matching "connectors"). Terms shorter than 4 characters use `includes()` instead of word boundary.

Semantic path skips the AND filter — BERT naturally handles plurals and synonyms.

## Ref pinning strategy

`kitfind.toml` stores branch names (`main`, `v2.0.0`) — always builds the latest commit. The resolved commit SHA is recorded in `index.json` (`sources[].commit`) and `kitfind.lock` for audit trail. To pin to a specific SHA, set `ref` to the full SHA in `kitfind.toml`.

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Python over Go** | Easier access to sentence-transformers and NLP libraries for embedding/classification |
| **Static site with async fetch** | Previously embedded 2.8 MB JSON caused "Page Unresponsive". Async fetch + loading spinner solves it. |
| **ONNX Runtime Web, not transformers.js** | `transformers.js` calls HuggingFace API at runtime regardless of local model paths, causing 429 rate limits. `ort.min.js` from CDN + custom JS tokenizer avoids any external API calls. |
| **No torch at build time** | ONNX Runtime + `huggingface_hub` + `tokenizers` handle everything. No 2 GB torch install in CI. |
| **Hybrid search (TF-IDF + BERT)** | Short queries (<5 chars) use TF-IDF to avoid BERT `[UNK]` noise. Longer queries get full semantic ranking. |
| **No AND filter for semantic path** | BERT naturally maps plurals and synonyms. AND filter only applied for TF-IDF and fallback paths. |
| **Term boosting** | Soft boost (`0.5×` per matching term) instead of hard cutoff. Preserves semantic discovery while promoting exact matches. |
| **Build-time embedding, client-side query encoding** | Pre-computes all skill embeddings at build time (no GPU needed at inference). Client only needs the model for query embedding. |
| **mypy type checking** | All Python function signatures annotated. CI runs `mypy` with explicit file list. |
| **JS as testable modules** | Pure search functions extracted to `tools/search.js`, inlined into template at build. Tested with `node --test` (26 tests). |
| **Template JS syntax validation** | `validate.py --html` inlines search.js into the template and runs `node --check` on the result. Runs in CI. |
| **uv over pip** | Faster dependency management, auto-lockfile. |
| **No Jinja2** | Template is self-contained HTML with JS. Reduces dependency surface. |
| **Ref pinning** | Branches in `kitfind.toml`, resolved SHAs in `kitfind.lock`. |
| **Deduplication** | By `(source.url, name)` using nested `source` object. |

## Development commands

```bash
# Build the site
uv run tools/build.py

# Validate everything (structures, embeddings, model, template JS)
uv run python tools/validate.py --all

# Validate template HTML JS (fast, no build needed)
uv run python tools/validate.py --html

# Type check (all function signatures annotated)
uv run mypy tools/build.py tools/validate.py

# Run JS tests (pure Node.js)
node --test tools/tests/search.test.mjs
```

## CI/CD pipeline

### CI (`ci.yml`) — push to main + PR

1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v5`
3. Build + validate: `uv sync --all-extras && uv run tools/build.py && uv run python tools/validate.py --all`
4. Type check: `uv run mypy tools/build.py tools/validate.py`
5. JS tests: `node --test tools/tests/search.test.mjs`
6. Template JS syntax: `uv run python tools/validate.py --html`
7. Deploy preview: `rossjrw/pr-preview-action@v1` → `gh-pages/pr-preview/pr-N/`

### Deploy (`deploy.yml`) — push to main

1. Build: `uv sync && uv run tools/build.py`
2. Deploy to `gh-pages` (root, keep_files: true)

### PR preview

PR previews auto-deploy to `https://gliptak.github.io/kitfind/pr-preview/pr-N/` and auto-cleanup on merge/close.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Skills repos change without notice | Pin refs in kitfind.toml, rebuild on push |
| SKILL.md format varies across repos | Flexible parser: YAML frontmatter + meta.json + regex fallback |
| Blob size grows with more repos | TF-IDF sidecar adds ~80 KB per 1k skills; JSON + TF-IDF under 5 MB combined for 10k skills |
| Duplicate skills across repos | Deduplicated by (source_url, name) |
| ONNX model (~22 MB) largest asset | Could switch to TinyBERT/DistilBERT variants; model only downloaded once per browser |

## Current stats

- **6 repos** indexed
- **2,150 unique skills**
- **72 domains** (keyword-classified)
- `index.json`: ~3.4 MB
- `index.embeddings`: ~808 KB (uint8-quantized 384-dim vectors)
- `index.tfidf`: ~1.3 MB (7,365 vocabulary terms)
- `model/onnx/model_quantized.onnx`: ~22 MB

## Integration with Kitout

The Kitout codebase already has registry loading and fuzzy search. Integration points:

- Kitout fetches `https://gliptak.github.io/kitfind/index.json` as a remote registry source
- The blob lets Kitout recommend skills without requiring users to manually configure repos
- Potential future integration: `kitout search <query>`, `kitout add <skill-name>` from kitfind results
