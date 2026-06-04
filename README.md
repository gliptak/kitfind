# Kitfind

A searchable index of AI coding agent skills, deployed as a GitHub Pages static site.

Kitfind crawls popular skills repos at build time, parses their SKILL.md files, generates a unified search blob (`index.json`), and creates a searchable static site.

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

## Architecture

```
kitfind.toml  ──>  tools/build.py  ──>  site/index.json  ──>  GitHub Pages
(curated            (Python aggregator)   (search blob)        (static site)
 repo list)              │
                         ├─ clones repos at pinned refs
                         ├─ parses SKILL.md frontmatter
                         ├─ classifies domains (keyword matcher)
                         ├─ deduplicates skills
                         └─ generates HTML + index.json
```

## Adding a repo

Edit `kitfind.toml` and add an entry:

```toml
[[repos]]
url = "https://github.com/owner/repo"
ref = "main"
description = "Brief description"
```

Then run `uv run tools/build.py` to regenerate.

## Development

```bash
uv run tools/build.py
# Output in site/
```

### Validate output

```bash
uv run python tools/validate.py
# Validated: 1806 skills, 5 sources, 70 domains
```

Checks that `site/index.json` has the right structure — required fields, non-empty arrays, valid version. Runs in CI on every PR.

