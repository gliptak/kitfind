# Kitfind

[![CI](https://github.com/gliptak/kitfind/actions/workflows/ci.yml/badge.svg)](https://github.com/gliptak/kitfind/actions/workflows/ci.yml)
[![Deploy](https://github.com/gliptak/kitfind/actions/workflows/deploy.yml/badge.svg)](https://github.com/gliptak/kitfind/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python >=3.11](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](pyproject.toml)
[![Last commit](https://img.shields.io/github/last-commit/gliptak/kitfind/main)](https://github.com/gliptak/kitfind)

A searchable index of AI coding agent skills, deployed as a GitHub Pages static site.

Kitfind crawls popular skills repos at build time, parses their SKILL.md files, generates a unified search blob (`index.json`), and creates a searchable static site.

See [AGENTS.md](AGENTS.md) for full architecture, search modes, blob schema, and development guide.
## Usage

### Web

Browse and search at: **https://gliptak.github.io/kitfind**

Search by keyword, filter by domain. Results are scored by relevance.

### The blob

The raw index is available as `index.json`:

```
https://gliptak.github.io/kitfind/index.json
```

Kitout and other tools can consume this blob for local search and recommendation.

## Currently indexed

See [`kitfind.toml`](kitfind.toml) — the curated list of repos is the single source of truth.

## Adding a repo

Edit `kitfind.toml` and add an entry:

```toml
[[repo]]
url = "https://github.com/owner/repo"
ref = "main"
description = "Brief description"
```

Then run `uv run tools/build.py` to regenerate.

## Development

```bash
# Build the site
uv run tools/build.py

# Validate structural checks
uv run python tools/validate.py --all

# Type check (all function signatures annotated)
uv run mypy tools/build.py tools/validate.py

# Run JS tests (pure Node.js, no browser)
node --test tools/tests/search.test.mjs

# Validate template HTML JS syntax (fast, no build needed)
uv run python tools/validate.py --html
```

JS tests run in pure Node.js with the built-in test runner, covering `matches()`, `tfidfTokenize()`, `tokenizeBERT()`, and `wordpieceEncode()`. Runs in CI on every PR.
