"""Validate the generated kitfind index.json.

Usage: python tools/validate.py [path/to/index.json]
Default: site/index.json in the repo root (../site/index.json relative to this script).
"""

import json
import sys
from pathlib import Path


def validate(path: Path) -> int:
    with open(path) as f:
        data = json.load(f)

    errors = []

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

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(
        f"Validated: {len(skills)} skills, "
        f"{total_sources} sources, "
        f"{len(by_domain)} domains"
    )
    return 0


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "site" / "index.json"
    sys.exit(validate(path))
