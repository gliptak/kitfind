"""Validate the generated kitfind output: index.json, index.embeddings, and model files.

Usage:
    python tools/validate.py                    # validate index.json
    python tools/validate.py --all              # validate everything
    python tools/validate.py path/to/index.json # custom path
"""

from typing import Optional
import json
import struct
import sys
from pathlib import Path


def validate_json(path: Path) -> list[str]:
    errors = []
    with open(path) as f:
        data = json.load(f)

    # Version
    if data.get("version") != 1:
        errors.append(f"version: expected 1, got {data.get('version')}")

    # Skills
    skills = data.get("skills", [])
    if len(skills) == 0:
        errors.append("skills: empty array")

    for s in skills:
        if not s.get("source"):
            errors.append(f"  {s.get('id', '?')}: missing source")
        if not s.get("name"):
            errors.append(f"  {s.get('id', '?')}: missing name")

    # Catalog
    catalog = data.get("catalog", {})
    stats = catalog.get("stats", {})
    total_sources = stats.get("total_sources", 0)
    if total_sources == 0:
        errors.append("catalog.stats.total_sources: expected > 0")

    by_domain = stats.get("by_domain", {})
    total_skills = stats.get("total_skills", 0)

    if not errors:
        print(
            f"  index.json: {len(skills)} skills, "
            f"{total_sources} sources, "
            f"{len(by_domain)} domains"
        )

    return errors


def validate_embeddings(path: Path, expected_n: int) -> list[str]:
    errors = []
    if not path.exists():
        errors.append(f"index.embeddings: file not found at {path}")
        return errors

    data = path.read_bytes()
    if len(data) < 12:
        errors.append(f"index.embeddings: too small ({len(data)} bytes, need >= 12)")
        return errors

    header = struct.unpack("3i", data[:12])
    version, dim, n = header

    if version != 1:
        errors.append(f"index.embeddings: version={version}, expected 1")
    if dim != 384:
        errors.append(f"index.embeddings: dim={dim}, expected 384")
    if n != expected_n:
        errors.append(f"index.embeddings: n={n}, expected {expected_n}")

    # Check file size matches header
    expected_bytes = 12 + dim * 4 + dim * 4 + n * dim * 1
    if len(data) != expected_bytes:
        errors.append(
            f"index.embeddings: size mismatch {len(data)} vs {expected_bytes}"
        )

    # Check normalization (spot-check first 10)
    import numpy as np

    mins = np.frombuffer(data[12 : 12 + dim * 4], dtype=np.float32)
    scales = np.frombuffer(data[12 + dim * 4 : 12 + dim * 8], dtype=np.float32)
    quantized = np.frombuffer(data[12 + dim * 8 :], dtype=np.uint8)

    bad_norms = 0
    for i in range(min(10, n)):
        emb = quantized[i * dim : (i + 1) * dim].astype(np.float32) * scales + mins
        norm = float(np.linalg.norm(emb))
        if abs(norm - 1.0) > 0.05:
            bad_norms += 1
            if bad_norms <= 3:
                errors.append(f"  embedding[{i}]: norm={norm:.4f} (expected ~1.0)")

    if bad_norms > 3:
        errors.append(f"  ... {bad_norms - 3} more embeddings have bad norms")

    if not errors:
        print(f"  index.embeddings: {n} skills x {dim} dims, {len(data) / 1024:.0f} KB")

    return errors


def validate_model(model_dir: Path) -> list[str]:
    errors = []
    required = [
        "onnx/model_quantized.onnx",
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
    ]
    for fname in required:
        p = model_dir / fname
        if not p.exists():
            errors.append(f"model/{fname}: missing")
        elif p.stat().st_size == 0:
            errors.append(f"model/{fname}: empty")

    onnx_file = model_dir / "onnx/model_quantized.onnx"
    if onnx_file.exists():
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"  model/model_quantized.onnx: {size_mb:.0f} MB")

    if not errors:
        print(f"  model/: all required files present")

    return errors

def validate_html_js(html_path: Path) -> list[str]:
    import re
    import subprocess
    import tempfile
    import os
    errors = []
    try:
        text = html_path.read_text(encoding="utf-8")
        match = re.search(r"<script>\s*\n(.*?)\n\s*</script>", text, re.DOTALL)
        if not match:
            errors.append("index.html: no <script> block found")
            return errors
        js_code = match.group(1)
        if not js_code.strip():
            errors.append("index.html: empty <script> block")
            return errors
        tmp = tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False, encoding="utf-8")
        try:
            tmp.write(js_code)
            tmp.close()
            result = subprocess.run(["node", "--check", tmp.name], capture_output=True, text=True)
            if result.returncode != 0:
                errors.append(f"index.html JS syntax: {result.stderr.strip()}")
            else:
                print(f"  HTML JS: syntax OK")
        finally:
            os.unlink(tmp.name)
    except Exception as e:
        errors.append(f"HTML JS check failed: {e}")
    return errors

def validate_template_html(repo_root: Optional[Path] = None) -> list[str]:
    """Validate JS syntax in the template HTML (without building).
    Inlines search.js at the placeholder, then checks script content.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    import re
    import subprocess
    import tempfile
    import os
    errors = []
    try:
        template_path = repo_root / "tools" / "templates" / "index.html"
        search_js_path = repo_root / "tools" / "search.js"
        if not template_path.exists():
            errors.append(f"Template not found: {template_path}")
            return errors
        if not search_js_path.exists():
            errors.append(f"search.js not found: {search_js_path}")
            return errors
        text = template_path.read_text(encoding="utf-8")
        search_js = search_js_path.read_text(encoding="utf-8")
        search_js = re.sub(
            r"\n// ── Exports ────────────────────────────────────────────.*",
            "", search_js, flags=re.DOTALL
        )
        if "<!-- SEARCH_JS -->" not in text:
            errors.append("Template missing <!-- SEARCH_JS --> placeholder")
            return errors
        text = text.replace("<!-- SEARCH_JS -->", search_js)
        match = re.search(r"<script>\s*\n(.*?)\n\s*</script>", text, re.DOTALL)
        if not match:
            errors.append("Template: no <script> block found")
            return errors
        js_code = match.group(1)
        if not js_code.strip():
            errors.append("Template: empty <script> block")
            return errors
        tmp = tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False, encoding="utf-8")
        try:
            tmp.write(js_code)
            tmp.close()
            result = subprocess.run(["node", "--check", tmp.name], capture_output=True, text=True)
            if result.returncode != 0:
                errors.append(f"Template JS syntax: {result.stderr.strip()}")
            else:
                print(f"  Template JS: syntax OK")
        finally:
            os.unlink(tmp.name)
    except Exception as e:
        errors.append(f"Template JS check failed: {e}")
    return errors

def spotcheck_embeddings(model_dir: Path, embed_path: Path, skills_path: Path) -> list[str]:
    """Run a few queries and verify that cosine similarities make sense."""
    errors = []
    try:
        import numpy as np
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError:
        errors.append("spotcheck: missing numpy/onnxruntime/tokenizers")
        return errors

    try:
        session = onnxruntime.InferenceSession(
            str(model_dir / "onnx/model_quantized.onnx")
        )
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    except Exception as e:
        errors.append(f"spotcheck: failed to load model/tokenizer: {e}")
        return errors

    # Load embeddings
    with open(embed_path, "rb") as f:
        data = f.read()
    header = struct.unpack("3i", data[:12])
    dim, n = header[1], header[2]
    mins = np.frombuffer(data[12 : 12 + dim * 4], dtype=np.float32)
    scales = np.frombuffer(data[12 + dim * 4 : 12 + dim * 8], dtype=np.float32)
    quantized = np.frombuffer(data[12 + dim * 8 :], dtype=np.uint8)

    # Load skills list
    with open(skills_path) as sf:
        skills_data = json.load(sf)
    skills = skills_data.get("skills", [])

    def embed(text: str) -> np.ndarray:
        encoded = tokenizer.encode(text)
        attn = encoded.attention_mask
        real_len = sum(attn) if attn and any(attn) else min(len(encoded.ids), 256)
        tokens = list(encoded.ids[:256]) if len(encoded.ids) > 0 else [0]
        tokens = tokens + [0] * max(0, 256 - len(tokens))
        mask = [1] * min(real_len, 256) + [0] * max(0, 256 - min(real_len, 256))
        mask_arr = np.array(mask[:256], dtype=np.int64)
        result = session.run(None, {
            "input_ids": np.array([tokens], dtype=np.int64),
            "attention_mask": mask_arr[np.newaxis, :],
            "token_type_ids": np.zeros((1, 256), dtype=np.int64),
        })[0][0]
        mp = (result * mask_arr.astype(np.float32).reshape(-1, 1)).sum(axis=0)
        pooled = mp / max(mask_arr.sum(), 1)
        norm = np.linalg.norm(pooled)
        return pooled / norm if norm > 0 else pooled

    queries = [
        "python testing",
        "machine learning",
        "docker deployment",
        "frontend react",
    ]

    for query in queries:
        qv = embed(query)
        scores = [
            (float(np.dot(qv, quantized[i * dim:(i + 1) * dim].astype(np.float32) * scales + mins)), skills[i]["name"])
            for i in range(min(n, len(skills)))
        ]
        scores.sort(reverse=True)
        top3 = [name for _, name in scores[:3]]
        # Sanity: top score should be > 0.3 (meaningful match)
        if scores[0][0] < 0.3:
            errors.append(
                f"  query '{query}': top score {scores[0][0]:.3f} too low "
                f"(top: {top3})"
            )
        else:
            print(f"  query '{query}': top={scores[0][0]:.3f} {top3}")

    return errors


def validate_all() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    site_dir = repo_root / "site"
    model_dir = site_dir / "model"
    json_path = site_dir / "index.json"
    embed_path = site_dir / "index.embeddings"

    all_errors = []

    # 1. index.json
    print("Validating index.json...")
    all_errors.extend(validate_json(json_path))

    # How many skills?
    with open(json_path) as f:
        data = json.load(f)
    skill_count = len(data.get("skills", []))

    if skill_count > 0:
        # 2. index.embeddings
        print("Validating index.embeddings...")
        all_errors.extend(validate_embeddings(embed_path, skill_count))

        # 3. model files
        print("Validating model/...")
        all_errors.extend(validate_model(model_dir))

        # 3b. HTML JS syntax
        print("Validating index.html JS...")
        all_errors.extend(validate_html_js(site_dir / "index.html"))

        # 4. Spot-check
        print("Spot-checking queries...")
        all_errors.extend(spotcheck_embeddings(model_dir, embed_path, json_path))

    if all_errors:
        for e in all_errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        return 1

    print("\nAll validations passed.")
    return 0


def validate(path: Path) -> int:
    """Legacy: validate only index.json."""
    errors = validate_json(path)
    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    if "--all" in sys.argv:
        sys.exit(validate_all())
    elif "--html" in sys.argv:
        errors = validate_template_html()
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1 if errors else 0)
    else:
        repo_root = Path(__file__).resolve().parent.parent
        path = (
            Path(sys.argv[1])
            if len(sys.argv) > 1 and sys.argv[1] != "--all"
            else repo_root / "site" / "index.json"
        )
        sys.exit(validate(path))
