"""Verify the conservative public-artifact candidate without project dependencies."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re


FORBIDDEN_SUFFIXES = {".mat", ".npy", ".npz", ".pth", ".pkl", ".pickle", ".pt"}
FORBIDDEN_PARTS = {"data", "data_e2", "results", "checkpoints_bspc_v2", "checkpoints_exploratory"}
LOCAL_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]|/home/|/Users/|/mnt/|/workspace/)")


def digest(path: pathlib.Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    self_path = pathlib.Path(__file__).resolve()
    failures: list[str] = []
    files = [path for path in root.rglob("*") if path.is_file()]
    restricted = [
        path
        for path in files
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or set(path.relative_to(root).parts) & FORBIDDEN_PARTS
    ]
    failures.extend(f"restricted file: {path.relative_to(root).as_posix()}" for path in restricted)

    text_files = [
        path
        for path in files
        if path != self_path
        and path.suffix.lower() in {".csv", ".json", ".log", ".md", ".ps1", ".py", ".txt", ".yml", ".yaml"}
    ]
    path_hits = [path for path in text_files if LOCAL_PATH_RE.search(path.read_text(encoding="utf-8-sig"))]
    failures.extend(f"local path: {path.relative_to(root).as_posix()}" for path in path_hits)

    json_files = [path for path in files if path.suffix.lower() == ".json"]
    json_failures = []
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            json_failures.append(f"{path.relative_to(root).as_posix()}: {error}")
    failures.extend(f"invalid JSON: {item}" for item in json_failures)

    python_files = [path for path in files if path.suffix.lower() == ".py"]
    syntax_failures = []
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError) as error:
            syntax_failures.append(f"{path.relative_to(root).as_posix()}: {error}")
    failures.extend(f"invalid Python: {item}" for item in syntax_failures)

    hash_path = root / "metadata/PUBLIC_FILE_SHA256SUMS.txt"
    hash_failures = []
    if not hash_path.is_file():
        hash_failures.append("missing metadata/PUBLIC_FILE_SHA256SUMS.txt")
    else:
        for line in hash_path.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = root / pathlib.PurePosixPath(relative)
            if not path.is_file():
                hash_failures.append(f"missing {relative}")
            elif digest(path) != expected:
                hash_failures.append(f"mismatch {relative}")
    failures.extend(f"hash: {item}" for item in hash_failures)

    result = {
        "status": "passed" if not failures else "failed",
        "files": len(files),
        "restricted_files": len(restricted),
        "local_path_files": len(path_hits),
        "json_files": len(json_files),
        "python_files": len(python_files),
        "hash_entries": len(hash_path.read_text(encoding="utf-8").splitlines()) if hash_path.is_file() else 0,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
