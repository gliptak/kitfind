# Build-time aggregator for Kitfind.
#
# Reads kitfind.toml, clones repos, parses SKILL.md files,
# generates index.json (the search blob) and a static HTML site.

import json
import os
import re
import shutil
import subprocess
import sys
import numpy as np
import tempfile
import tomli
from pathlib import Path
from datetime import date, datetime, timezone


# ── Paths ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
TOML_PATH = REPO_ROOT / "kitfind.toml"
SCHEMA_PATH = REPO_ROOT / "schema.json"
OUTPUT_DIR = REPO_ROOT / "site"

# ── Helpers ──────────────────────────────────────────────────────────


def parse_yaml_frontmatter(text: str, filepath: str = "") -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown file.
    Tries strict YAML first, falls back to regex extraction on parse errors.
    """
    frontmatter: dict = {}
    body = text

    # Match --- ... --- at the start
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if m:
        import yaml

        raw = m.group(1)
        try:
            frontmatter = yaml.safe_load(raw) or {}
        except yaml.YAMLError:
            # Fallback: regex extraction for problematic YAML
            frontmatter = _extract_fields(raw)
            if not frontmatter.get("name"):
                print(f"  [warn] YAML fallback extracted no name for {filepath}", file=sys.stderr)
        body = text[m.end() :]

    if not isinstance(frontmatter, dict):
        frontmatter = {}
    return frontmatter, body.strip()


def _extract_fields(raw: str) -> dict:
    """Regex-based fallback for malformed YAML frontmatter.
    Handles unquoted colons in values, encoded chars, and multi-line values.
    """
    fields: dict[str, list[str]] = {}
    current_key = None

    for line in raw.split("\n"):
        # Top-level key: value
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)\s*$", line)
        if m:
            current_key = m.group(1)
            val = m.group(2).strip()
            if val:
                # Remove trailing colon artifacts from description bleed
                val = re.sub(r"\b[a-z_]+:\s*[^\s]*\s*$", "", val).strip()
                fields.setdefault(current_key, []).append(val)
        elif current_key and line.startswith(" ") and line.strip():
            # Continuation of previous multi-line value
            val = line.strip()
            val = re.sub(r"\b[a-z_]+:\s*[^\s]*\s*$", "", val).strip()
            if val:
                fields[current_key].append(val)

    result: dict[str, str | list[str]] = {}
    for k, vlist in fields.items():
        combined = " ".join(vlist)
        # Try parsing simple list values e.g. [item1, item2]
        if combined.startswith("[") and combined.endswith("]"):
            items = [i.strip().strip("'\"") for i in combined[1:-1].split(",")]
            result[k] = items
        else:
            result[k] = combined
    return result


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def extract_keywords(text: str, max_len: int = 40) -> list[str]:
    """Extract meaningful keywords from a line of text, filtering stop words."""
    _SW = frozenset({'the','a','an','is','are','was','were','be','been','being',
        'have','has','had','do','does','did','will','would','could','should',
        'may','might','can','shall','to','of','in','for','on','with','at','by',
        'from','as','into','through','during','then','once','here','there',
        'when','where','why','how','all','each','every','both','few','more',
        'most','other','some','such','no','nor','not','only','own','same','so',
        'than','too','very','just','about','which','who','whom','this','that',
        'these','those','and','but','or','if','while','although','since',
        'unless','until','like','it','its','you','your','we','our','they',
        'them','their','common','use','using'})
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\\-]{2,}", text)
    return [w.lower() for w in words if len(w) <= max_len and w.lower() not in _SW][:20]


def git_clone(url: str, ref: str, target: Path) -> str | None:
    """Shallow-clone a repo at a ref. Returns commit SHA or None."""
    if target.exists():
        shutil.rmtree(target)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, url, str(target)],
            capture_output=True,
            timeout=120,
            check=True,
        )
        result = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"  [warn] git clone failed for {url}@{ref}: {e.stderr.decode() if e.stderr else e}", file=sys.stderr)
        # Try without branch
        try:
            shutil.rmtree(target)
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target)],
                capture_output=True, timeout=120, check=True,
            )
            result = subprocess.run(
                ["git", "-C", str(target), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30, check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e2:
            print(f"  [warn] git clone (no branch) also failed: {e2}", file=sys.stderr)
            return None


# ── Repo configuration ──────────────────────────────────────────────


def load_config() -> list[dict]:
    """Load kitfind.toml and return list of repo entries."""
    if not TOML_PATH.exists():
        print("error: kitfind.toml not found", file=sys.stderr)
        sys.exit(1)
    with open(TOML_PATH, "rb") as f:
        data = tomli.load(f)
    return data.get("repos", [])


# ── Domain classification (keyword-based) ─────────────────────────


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "testing": [
        "tdd", "test-driven", "unit test", "integration test", "e2e",
        "coverage", "assert", "spec", "qa", "quality assurance",
        "red-green", "refactor", "test suite", "regression",
    ],
    "debugging": [
        "debug", "bug", "root cause", "trace", "diagnose", "fix",
        "troubleshoot", "reproduce", "stack trace", "log analysis",
        "defect", "fault", "error investigation",
    ],
    "devops": [
        "deploy", "ci/cd", "ci cd", "pipeline", "docker", "kubernetes",
        "k8s", "terraform", "infrastructure", "ansible", "helm",
        "monitoring", "observability", "sre", "reliability",
        "incident response", "rollback", "canary", "blue-green",
    ],
    "security": [
        "security", "audit", "vulnerability", "owasp", "secrets",
        "compliance", "penetration test", "threat", "auth", "rbac",
        "encryption", "cve", "supply chain", "dependency check",
    ],
    "frontend": [
        "frontend", "ui", "css", "react", "component", "layout",
        "responsive", "web design", "accessibility", "a11y",
        "browser", "dom", "javascript", "typescript", "spa",
        "user interface", "ux", "tailwind", "styled",
    ],
    "backend": [
        "backend", "api", "server", "database", "rest", "graphql",
        "microservice", "middleware", "endpoint", "authentication",
        "authorization", "sql", "nosql", "orm", "cache", "redis",
        "message queue", "grpc", "webhook",
    ],
    "data": [
        "data", "analytics", "etl", "data pipeline", "sql query",
        "data warehouse", "data lake", "big data", "spark", "airflow",
        "data engineering", "data science", "statistics", "visualization",
    ],
    "ai-ml": [
        "ai", "machine learning", "ml", "model", "prompt", "llm",
        "embedding", "vector", "rag", "retrieval augmented",
        "fine[- ]tune", "neural", "deep learning", "nlp",
        "language model", "chatbot", "gpt", "claude", "openai",
        "inference", "training", "dataset",
    ],
    "writing": [
        "writing", "documentation", "content", "copywriting",
        "blog post", "article", "technical writing", "readme",
        "prose", "editorial", "story", "narrative",
    ],
    "design": [
        "design", "architecture", "planning", "spec", "brainstorm",
        "prototype", "wireframe", "mockup", "figma", "design doc",
        "system design", "technical design", "adr",
    ],
    "productivity": [
        "productivity", "workflow", "automate", "shortcut",
        "efficiency", "scaffold", "boilerplate", "template",
        "convention", "lint", "formatter", "pre-commit", "hook",
    ],
    "project-management": [
        "project management", "roadmap", "agile", "milestone",
        "triage", "sprint", "backlog", "estimation", "jira",
        "linear", "task", "stakeholder",
    ],
    "mobile": [
        "mobile", "ios", "android", "swift", "kotlin", "react native",
        "flutter", "dart", "app store", "play store", "mobile dev",
    ],
    "game-development": [
        "game", "gamedev", "unity", "unreal", "godot", "sprite",
        "physics engine", "rendering", "shader",
    ],
    "code-review": [
        "code review", "pr", "pull request", "peer review",
        "merge request", "review checklist", "diff",
    ],
    "cli": [
        "cli", "command line", "terminal", "shell", "bash",
        "zsh", "scripting", "stdin", "stdout",
    ],
}


def classify_domain(
    name: str,
    description: str,
    triggers: list[str],
    body_preview: str,
    current_domain: str = "",
) -> str:
    """
    Classify a skill into a domain using keyword matching.

    Returns the original current_domain if it's already set to something
    meaningful (not uncategorized), or the best keyword-based match.
    """
    # If the repo already provides a meaningful domain, trust it
    if current_domain and current_domain != "uncategorized":
        return current_domain

    # Build a combined text corpus for matching
    corpus = " ".join([
        name.lower(),
        description.lower(),
        " ".join(t.lower() for t in triggers),
        body_preview.lower()[:3000],  # first 3k chars of body
    ])

    # Score each domain
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # Count occurrences; bonus for phrase matches
            count = corpus.count(kw.lower())
            if count:
                # Weighted: phrase matches (with space) score higher
                if " " in kw:
                    score += count * 3
                else:
                    score += count
        if score > 0:
            scores[domain] = score

    if not scores:
        return "uncategorized"

    # Return the highest-scoring domain
    best = max(scores, key=lambda d: (scores[d], d))
    return best


# ── Skill discovery ─────────────────────────────────────────────────


def discover_skills(repo_dir: Path) -> list[dict]:
    """Discover all skills in a cloned repo by scanning for SKILL.md files."""
    skills = []

    # Walk the repo directory looking for SKILL.md
    for sk_path in repo_dir.rglob("SKILL.md"):
        # Skip if in .git or node_modules or similar
        rel = sk_path.relative_to(repo_dir)
        parts = rel.parts
        if any(p.startswith(".") or p == "node_modules" for p in parts):
            continue
        # Skip if SKILL.md is too shallow (e.g. root-level)
        if len(parts) < 2:
            continue

        skill_dir = sk_path.parent
        content = sk_path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_yaml_frontmatter(content, filepath=str(sk_path))

        skill = {
            "name": meta.get("name") or skill_dir.name,
            "description": meta.get("description") or "",
            "body_preview": body[:500] if body else "",
            "path": str(rel.parent),
        }

        # Try meta.json companion file
        meta_json_path = skill_dir / "meta.json"
        if meta_json_path.exists():
            try:
                with open(meta_json_path) as f:
                    meta_json = json.load(f)
                    meta.update(meta_json)
            except (json.JSONDecodeError, OSError):
                pass

        # Extract tags/triggers
        raw_tags = meta.get("tags")
        if raw_tags and isinstance(raw_tags, str):
            raw_tags = [t.strip() for t in raw_tags.replace("[", "").replace("]", "").split(",") if t.strip()]
        skill["tags"] = raw_tags if isinstance(raw_tags, list) else []
        if "trigger" in meta:
            raw = meta["trigger"]
            skill["triggers"] = [raw] if isinstance(raw, str) else raw
        elif "triggers" in meta:
            raw = meta["triggers"]
            if isinstance(raw, str):
                skill["triggers"] = [t.strip() for t in raw.replace("[", "").replace("]", "").split(",") if t.strip()]
            else:
                skill["triggers"] = raw if isinstance(raw, list) else []
        else:
            # Derive triggers from description keywords
            skill["triggers"] = extract_keywords(meta.get("description", ""))
        # Domain / category — keyword classifier (falls back to "uncategorized")
        skill["domain"] = classify_domain(
            name=skill["name"],
            description=skill["description"],
            triggers=(lambda t: t if isinstance(t, list) else [])(skill.get("triggers")),
            body_preview=skill.get("body_preview", ""),
            current_domain=meta.get("domain") or meta.get("category") or "",
        )

        # Harness compatibility
        skill["harnesses"] = (
            meta.get("harnesses")
            or meta.get("model_compatibility")
            or ["claude-code"]
        )

        # Risk / source (antigravity-style)
        if "risk" in meta:
            skill["risk"] = meta["risk"]
        if "source" in meta:
            skill["source_name"] = meta["source"]
        if "date_added" in meta:
            d = meta["date_added"]
            skill["date_added"] = d.isoformat() if isinstance(d, date) else str(d)

        skills.append(skill)

    return skills
# ── Semantic embeddings ──────────────────────────────────────────────


_MODEL_DOWNLOADED = False


def _extract_vocab(model_dir: Path):
    """Extract word -> id vocab from tokenizer.json as a flat JSON."""
    import json
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        tok_data = json.loads(tok_path.read_text())
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab:
            vocab_path = model_dir / "vocab.json"
            vocab_path.write_text(json.dumps(vocab))
            print(f"  Extracted vocab ({len(vocab)} tokens) to model/vocab.json")
        else:
            print("  WARNING: Could not find vocab in tokenizer.json")
    else:
        print("  WARNING: tokenizer.json not found")


def _github_repo_slug() -> str | None:
    """Infer owner/repo for the current checkout when available."""
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if re.match(r"^[^/]+/[^/]+$", repo):
        return repo

    try:
        remote = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None

    match = re.match(
        r"^(?:https://github\.com/|git@github\.com:)([^/]+/[^/.]+?)(?:\.git)?$",
        remote,
    )
    return match.group(1) if match else None


def _model_base_urls() -> list[str]:
    """Return candidate base URLs for model assets, preferring existing gh-pages files."""
    bases: list[str] = []
    override = os.environ.get("KITFIND_MODEL_BASE_URL", "").strip().rstrip("/")
    if override:
        bases.append(override)

    repo = _github_repo_slug()
    if repo:
        bases.append(f"https://raw.githubusercontent.com/{repo}/gh-pages/model")

    bases.append("https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main")
    return list(dict.fromkeys(bases))


def _download_file(url: str, dest: Path):
    """Download a file atomically to avoid leaving partial artifacts behind."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "kitfind-build/1.0"})
    tmp_path = None
    try:
        with urllib.request.urlopen(req, timeout=60) as src:
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                shutil.copyfileobj(src, tmp)
                tmp_path = Path(tmp.name)
        tmp_path.replace(dest)
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        if dest.exists():
            dest.unlink()
        raise


def _ensure_model(model_dir: Path) -> Path:
    """Download Xenova all-MiniLM-L6-v2 ONNX model files if not present.

    Prefer the repo's existing gh-pages copy, then fall back to Hugging Face.
    Returns path to the .onnx file.
    """
    global _MODEL_DOWNLOADED
    if _MODEL_DOWNLOADED and model_dir.exists():
        target = model_dir / "onnx" / "model_quantized.onnx"
        if target.exists():
            return target

    import time
    import urllib.error
    model_dir.mkdir(parents=True, exist_ok=True)
    _MODEL_DOWNLOADED = True

    files = [
        "onnx/model_quantized.onnx",
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    bases = _model_base_urls()
    for rel in files:
        dest = model_dir / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        for base in bases:
            url = f"{base}/{rel}"
            for attempt in range(6):
                try:
                    print(f"    Downloading {url.split('/')[-1]} from {base}...")
                    _download_file(url, dest)
                    errors = []
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        errors.append(f"{url} -> HTTP 404")
                        break
                    if attempt == 5:
                        errors.append(f"{url} -> HTTP {e.code}")
                        break
                    wait = 2 ** attempt
                    print(f"      Retry {attempt + 1}/5 after {wait}s (HTTP {e.code})")
                    time.sleep(wait)
                except Exception as e:
                    if attempt == 5:
                        errors.append(f"{url} -> {e}")
                        break
                    wait = 2 ** attempt
                    print(f"      Retry {attempt + 1}/5 after {wait}s ({e})")
                    time.sleep(wait)
            if dest.exists():
                break
        if not dest.exists():
            raise RuntimeError(
                f"Failed to download {rel} from any source:\n  " + "\n  ".join(errors)
            )

    # Extract vocab from tokenizer.json for client-side tokenizer
    _extract_vocab(model_dir)
    return model_dir / "onnx" / "model_quantized.onnx"
def _pool_and_normalize(hidden: "np.ndarray", mask: "np.ndarray") -> "np.ndarray":
    """Mean pool and L2-normalize transformer output."""
    import numpy as np
    mask_exp = mask.astype(np.float32).reshape(-1, 1)
    summed = (hidden * mask_exp).sum(axis=0)
    pooled = summed / max(mask.sum(), 1)
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 0 else pooled


def compute_embeddings(skills: list[dict], model_path: Path) -> "np.ndarray":
    """Embed all skills using ONNX Runtime, returning float32 array (N x 384)."""
    import numpy as np
    import onnxruntime
    from tokenizers import Tokenizer
    import sys

    tok_dir = model_path.parent.parent  # model/onnx/model_quantized.onnx -> model/
    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))

    session = onnxruntime.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    mask_name = session.get_inputs()[1].name

    embeddings = []
    for s in skills:
        text = " ".join([
            s.get("name", "") or "",
            s.get("description", "") or "",
            " ".join(s.get("triggers") or []),
            s.get("domain", "") or "",
        ])
        encoded = tokenizer.encode(text)
        attn = encoded.attention_mask
        real_len = sum(attn) if attn and any(attn) else min(len(encoded.ids), 256)
        tokens = encoded.ids[:256] if len(encoded.ids) > 0 else [0]
        if len(tokens) < 256:
            tokens = tokens + [0] * (256 - len(tokens))
        mask = [1] * min(real_len, 256) + [0] * max(0, 256 - min(real_len, 256))

        result = session.run(None, {
            input_name: np.array([tokens], dtype=np.int64),
            mask_name: np.array([mask], dtype=np.int64),
            'token_type_ids': np.zeros((1, 256), dtype=np.int64),
        })[0][0]

        embedding = _pool_and_normalize(result, np.array(mask))
        embeddings.append(embedding)

    return np.array(embeddings, dtype=np.float32)


def save_embeddings(embeddings: "np.ndarray", path: Path):
    """Save uint8-quantized embeddings to a compact binary file.
    Format: int32 header (version=1, dim=384, count=N) +
            float32 mins (384) + float32 scales (384) +
            uint8 data (N x 384)
    """
    import numpy as np

    mins = embeddings.min(axis=0)
    maxs = embeddings.max(axis=0)
    scales = (maxs - mins) / 255.0
    scales = np.where(scales == 0, 1.0, scales)
    quantized = ((embeddings - mins) / scales).clip(0, 255).astype(np.uint8)

    header = np.array([1, 384, len(embeddings)], dtype=np.int32)
    with open(path, "wb") as f:
        header.tofile(f)
        mins.astype(np.float32).tofile(f)
        scales.astype(np.float32).tofile(f)
        quantized.tofile(f)
    print(f"  Wrote site/index.embeddings ({len(embeddings)} skills, {embeddings.nbytes // 1024} KB raw)")
# ── TF-IDF fallback for short queries ───────────────────────────────────


def _tokenize_tfidf(text: str) -> list[str]:
    """Tokenize text for TF-IDF: lowercase, word chars only, filter stop words."""
    import re
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", text.lower())
    _SW = frozenset({
        'the','a','an','is','are','was','were','be','been','being',
        'have','has','had','do','does','did','will','would','could','should',
        'may','might','can','shall','to','of','in','for','on','with','at','by',
        'from','as','into','through','during','then','once','here','there',
        'when','where','why','how','all','each','every','both','few','more',
        'most','other','some','such','no','nor','not','only','own','same','so',
        'than','too','very','just','about','which','who','whom','this','that',
        'these','those','and','but','or','if','while','although','since',
        'unless','until','like','it','its','you','your','we','our','they',
        'them','their','common','use','using',
    })
    return [w for w in words if w not in _SW and len(w) <= 30]


def compute_tfidf(skills: list[dict]) -> dict:
    """Compute TF-IDF vectors for all skills.
    Returns {vocab, idf (array len=V), vectors (sparse rows), n_docs}.
    """
    import math
    from collections import Counter

    n = len(skills)
    term_doc_count: Counter = Counter()
    docs: list[set] = []

    for s in skills:
        text = " ".join([
            s.get("name", "") or "",
            s.get("description", "") or "",
            " ".join(s.get("triggers") or []),
            " ".join(s.get("tags") or []),
            s.get("domain", "") or "",
        ])
        tokens = _tokenize_tfidf(text)
        uniq = set(tokens)
        docs.append(uniq)
        for t in uniq:
            term_doc_count[t] += 1

    # Build vocab: keep terms that appear in 1..80% of skills
    vocab: dict[str, int] = {}
    idf: list[float] = []
    for term, df in sorted(term_doc_count.items()):
        if 1 <= df <= n * 0.8:
            idx = len(vocab)
            vocab[term] = idx
            idf.append(math.log(n / (1 + df)))

    # Build sparse TF vectors (pre-normalized)
    vectors: list[list] = []
    for s in skills:
        text = " ".join([
            s.get("name", "") or "",
            s.get("description", "") or "",
            " ".join(s.get("triggers") or []),
            " ".join(s.get("tags") or []),
            s.get("domain", "") or "",
        ])
        tokens = _tokenize_tfidf(text)
        tf: Counter = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []
        for term, count in tf.items():
            vidx = vocab.get(term)
            if vidx is not None:
                indices.append(vidx)
                tf_weight = 1.0 + (math.log(count) if count > 0 else 0)
                values.append(tf_weight * idf[vidx])
        vectors.append([indices, values])

    # L2-normalize each vector
    for row in vectors:
        norm = math.sqrt(sum(v * v for v in row[1]))
        if norm > 0:
            inv = 1.0 / norm
            row[1] = [v * inv for v in row[1]]

    print(f"  TF-IDF vocab: {len(vocab)} terms, {len(vectors)} vectors")
    return {"vocab": vocab, "idf": idf, "vectors": vectors, "n_docs": n}


def save_tfidf(data: dict, path: Path):
    """Write TF-IDF sidecar as compact JSON."""
    path.write_text(json.dumps(
        {
            "vocab": data["vocab"],
            "idf": data["idf"],
            "vectors": data["vectors"],
            "n_docs": data["n_docs"],
        },
        separators=(",", ":"),
    ))
    print(f"  Wrote site/index.tfidf ({path.stat().st_size // 1024} KB)")
# ── Index generation ────────────────────────────────────────────────





# ── Static site generation ──────────────────────────────────────────


def render_html(template_name: str, **context) -> str:
    """Read a static HTML template (no Jinja2 — self-contained with fetch)."""
    path = REPO_ROOT / "tools" / "templates" / template_name
    return path.read_text(encoding="utf-8")


def build_site(skills: list[dict], stats: dict, sources: list[dict]):
    """Generate the static HTML site."""
    # Group skills by domain
    by_domain: dict[str, list[dict]] = {}
    for s in skills:
        domain = s.get("domain", "uncategorized")
        by_domain.setdefault(domain, []).append(s)

    # Write lock file with resolved commit SHAs
    lock = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "url": s["url"],
                "ref": s["ref"],
                "commit": s["commit"],
            }
            for s in sources
        ],
    }
    (OUTPUT_DIR / "kitfind.lock").write_text(json.dumps(lock, indent=2))
    print(f"  Wrote site/kitfind.lock")

    # Sort domains by count descending
    sorted_domains = sorted(by_domain.items(), key=lambda x: -len(x[1]))

    # Home page
    html = render_html(
        "index.html",
        stats=stats,
        sources=sources,
        domains=sorted_domains,
        skills_json=json.dumps(skills, indent=2),
        total=len(skills),
    )
    # Inline search.js into the template (shared between site and tests)
    search_js_path = REPO_ROOT / "tools" / "search.js"
    if search_js_path.exists():
        search_js = search_js_path.read_text(encoding="utf-8")
        # Strip CommonJS exports block
        search_js = re.sub(
            r'\n// ── Exports ────────────────────────────────────────────.*',
            '', search_js, flags=re.DOTALL
        )
        html = html.replace('<!-- SEARCH_JS -->', search_js)
    (OUTPUT_DIR / "index.html").write_text(html)
    print(f"  Wrote site/index.html")

    # Copy assets
    assets_src = REPO_ROOT / "assets"
    if assets_src.exists():
        for f in assets_src.iterdir():
            if f.suffix in (".svg", ".png", ".jpg"):
                shutil.copy2(f, OUTPUT_DIR / f.name)
                print(f"  Copied asset {f.name}")

    # Copy index.json
    (OUTPUT_DIR / "index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "catalog": {
                    "sources": sources,
                    "stats": stats,
                },
                "skills": skills,
            },
            indent=2,
        )
    )
    print(f"  Wrote site/index.json ({len(skills)} skills)")


# ── Main ────────────────────────────────────────────────────────────


def main():
    repos = load_config()
    print(f"Loaded {len(repos)} repos from kitfind.toml")

    all_skills: list[dict] = []
    source_info: list[dict] = []
    total_skill_count = 0

    # Clean output dir
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Process each repo
    for i, repo in enumerate(repos, 1):
        url = repo["url"]
        ref = repo.get("ref", "main")
        desc = repo.get("description", "")
        print(f"\n[{i}/{len(repos)}] {desc}")
        print(f"  Cloning {url} @ {ref}")

        # Clone to temp directory
        repo_name = url.rstrip("/").split("/")[-1]
        clone_dir = Path(tempfile.mkdtemp()) / repo_name

        commit = git_clone(url, ref, clone_dir)
        if not commit:
            print(f"  [skip] could not clone {url}")
            shutil.rmtree(clone_dir.parent, ignore_errors=True)
            continue

        print(f"  Commit: {commit[:12]}")

        # Discover skills
        skills = discover_skills(clone_dir)
        print(f"  Found {len(skills)} SKILL.md files")

        # Enrich with source info
        for s in skills:
            source_path = s.pop("path")
            s["source"] = {"url": url, "ref": ref, "path": source_path}
            s["id"] = f"{repo_name}/{source_path}"
            s["install_hint"] = (
                f"kitout install {url.split('/')[-2]}/{repo_name}"
            )

        all_skills.extend(skills)
        total_skill_count += len(skills)

        source_info.append({
            "url": url,
            "ref": ref,
            "description": desc,
            "commit": commit,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "skill_count": len(skills),
        })

        # Cleanup
        shutil.rmtree(clone_dir.parent, ignore_errors=True)

    # Deduplicate by (source url, name)
    seen = set()
    unique_skills = []
    for s in all_skills:
        key = (s.get("source", {}).get("url", ""), s.get("name", ""))
        if key not in seen:
            seen.add(key)
            unique_skills.append(s)
    dups = len(all_skills) - len(unique_skills)
    if dups:
        print(f"\nDeduplicated {dups} duplicates")

    # Compute stats
    domain_counts: dict[str, int] = {}
    for s in unique_skills:
        d = s.get("domain", "uncategorized") or "uncategorized"
        domain_counts[d] = domain_counts.get(d, 0) + 1

    stats = {
        "total_skills": len(unique_skills),
        "total_sources": len(source_info),
        "by_domain": dict(sorted(domain_counts.items(), key=lambda x: -x[1])),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


    # Generate static site
    # Compute semantic embeddings
    print(f"  Computing semantic embeddings...")
    model_path = _ensure_model(OUTPUT_DIR / "model")
    embeddings = compute_embeddings(unique_skills, model_path)
    save_embeddings(embeddings, OUTPUT_DIR / "index.embeddings")
    # TF-IDF sidecar for short queries
    tfidf = compute_tfidf(unique_skills)
    save_tfidf(tfidf, OUTPUT_DIR / "index.tfidf")

    print(f"\nGenerating site with {len(unique_skills)} skills...")
    build_site(unique_skills, stats, source_info)

    print(f"\nDone! Output in {OUTPUT_DIR}")
    print(f"  Skills: {len(unique_skills)}")
    print(f"  Sources: {len(source_info)}")
    print(f"  Domains: {len(domain_counts)}")


if __name__ == "__main__":
    main()
