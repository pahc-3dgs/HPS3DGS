#!/usr/bin/env python3
"""Refresh the source manifest without staging, committing, or network access.

The root index selects files (including staged additions); working-tree bytes are
hashed. Dependencies must be clean and checked out at their parent's gitlink.
The manifest and current root HEAD are intentionally excluded from self-hashing.
Python 3.8+; standard library only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
LFS_POINTER_LIMIT = 8192


class ManifestError(RuntimeError):
    pass


def git(repo, *args, **kwargs):
    result = subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        input=kwargs.get("input_bytes"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestError(
            "git {} failed in {}: {}".format(" ".join(args), repo, error[:3000])
        )
    return result.stdout


def path_text(value):
    return value.decode("utf-8", errors="surrogateescape")


def relative_path(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        raise ManifestError("Path is outside release root: {}".format(path))


def tracked_entries(repo):
    """Read the index, not HEAD, so staged additions/deletions are respected."""
    entries = []
    for record in git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, oid, stage = metadata.split(b" ")
        path = path_text(name)
        if stage != b"0":
            raise ManifestError("Unmerged index entry in {}: {}".format(repo, path))
        entries.append((path, mode.decode("ascii"), oid.decode("ascii")))
    return sorted(entries)


def small_blobs(repo, entries):
    """Inspect small INDEX blobs to detect LFS pointers before reading weights."""
    object_ids = sorted({oid for _, mode, oid in entries if mode != "160000"})
    if not object_ids:
        return {}
    request = ("\n".join(object_ids) + "\n").encode("ascii")
    metadata = git(repo, "cat-file", "--batch-check", input_bytes=request)
    small_ids = []
    for line in metadata.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[1] != b"blob":
            raise ManifestError("Unexpected tracked object in {}: {!r}".format(repo, line))
        if int(parts[2]) <= LFS_POINTER_LIMIT:
            small_ids.append(parts[0].decode("ascii"))
    if not small_ids:
        return {}
    payload = git(
        repo,
        "cat-file",
        "--batch",
        input_bytes=("\n".join(small_ids) + "\n").encode("ascii"),
    )
    blobs = {}
    offset = 0
    for expected_oid in small_ids:
        end = payload.find(b"\n", offset)
        if end < 0:
            raise ManifestError("Truncated git cat-file output in {}".format(repo))
        oid, kind, length = payload[offset:end].split()
        size = int(length)
        start = end + 1
        stop = start + size
        if oid.decode("ascii") != expected_oid or kind != b"blob" or payload[stop:stop + 1] != b"\n":
            raise ManifestError("Invalid git cat-file output in {}".format(repo))
        blobs[expected_oid] = payload[start:stop]
        offset = stop + 1
    return blobs


def lfs_pointer(blob):
    if not blob or blob.splitlines()[0] != LFS_HEADER:
        return None
    oid = re.search(br"(?m)^oid sha256:([0-9a-f]{64})\r?$", blob)
    size = re.search(br"(?m)^size ([0-9]+)\r?$", blob)
    if not oid or not size:
        raise ManifestError("Malformed Git LFS pointer in tracked index")
    return {
        "oid": "sha256:" + oid.group(1).decode("ascii"),
        "size": int(size.group(1)),
        "pointer_sha256": hashlib.sha256(blob).hexdigest(),
    }


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def untracked_files(repo):
    return [
        path_text(value)
        for value in git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if value
    ]


def assert_dependency(repo, root, parent, index_head):
    if not repo.is_dir():
        raise ManifestError("Submodule is missing: {}".format(relative_path(repo, root)))
    actual_root = Path(path_text(git(repo, "rev-parse", "--show-toplevel").strip())).resolve()
    if actual_root != repo.resolve():
        raise ManifestError("Submodule is not initialized: {}".format(relative_path(repo, root)))
    head = git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    if head != index_head:
        raise ManifestError(
            "Submodule HEAD differs from parent gitlink: {} (HEAD {}, index {})".format(
                relative_path(repo, root), head, index_head
            )
        )
    dirty = git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    if dirty:
        raise ManifestError(
            "Submodule must be clean: {}\n{}".format(
                relative_path(repo, root), dirty.decode("utf-8", errors="replace")[:3000]
            )
        )
    return {"head": head, "parent": relative_path(parent, root), "index_head": index_head}


def root_base(root):
    provenance = root / "provenance" / "sources_before.json"
    if not provenance.exists():
        return None
    try:
        value = json.loads(provenance.read_text(encoding="utf-8"))
        head = value["hps-3dgs"]["head"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ManifestError("Cannot read hps-3dgs.head from {}: {}".format(provenance, error))
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
        raise ManifestError("Invalid hps-3dgs.head in {}".format(provenance))
    return head.lower()


def build_manifest(root, output):
    expected_root = Path(path_text(git(root, "rev-parse", "--show-toplevel").strip())).resolve()
    if expected_root != root:
        raise ManifestError("--root must be the repository root, not a subdirectory")
    output_relative = relative_path(output, root)
    excluded_manifests = {"release_manifest.json", output_relative}
    untracked = sorted(set(untracked_files(root)) - excluded_manifests)
    if untracked:
        raise ManifestError(
            "Untracked files would be omitted. Stage intended source explicitly or move unrelated output outside the repository:\n"
            + "\n".join(untracked[:100])
            + ("\n... ({} files total)".format(len(untracked)) if len(untracked) > 100 else "")
        )

    files = {}
    repositories = {}
    lfs_pointers = {}
    symlinks = {}
    visited = set()

    def collect(repo):
        resolved_repo = repo.resolve()
        relative_path(resolved_repo, root)
        if resolved_repo in visited:
            raise ManifestError("Duplicate or cyclic submodule path: {}".format(repo))
        visited.add(resolved_repo)
        entries = tracked_entries(repo)
        blobs = small_blobs(repo, entries)
        for name, mode, oid in entries:
            source = repo / name
            key = relative_path(source, root)
            if repo == root and key in excluded_manifests:
                continue
            if mode == "160000":
                repositories[key] = assert_dependency(source, root, repo, oid)
                collect(source)
                continue
            if mode not in ("100644", "100755", "120000"):
                raise ManifestError("Unsupported index mode {}: {}".format(mode, key))
            pointer = lfs_pointer(blobs.get(oid))
            if pointer is not None:
                lfs_pointers[key] = pointer
                continue
            resolved_source = source.resolve()
            relative_path(resolved_source, root)
            if not source.is_file():
                raise ManifestError("Tracked source is missing or not a file: {}".format(key))
            if mode == "120000":
                # Match consumers which hash resolved bytes; retain link identity
                # separately. Symlinks escaping the release root are rejected.
                symlinks[key] = {
                    "index_blob": oid,
                    "target": os.readlink(source) if source.is_symlink() else None,
                    "checkout_is_symlink": source.is_symlink(),
                }
            files[key] = hash_file(source)

    collect(root)
    return {
        "schema_version": 1,
        "root_base": root_base(root),
        "files": dict(sorted(files.items())),
        "repositories": dict(sorted(repositories.items())),
        "lfs_pointers": dict(sorted(lfs_pointers.items())),
        "symlinks": dict(sorted(symlinks.items())),
        "excluded": {
            "manifest_paths": sorted(excluded_manifests),
            "manifest_reason": "The manifest does not hash itself; no current root HEAD or timestamps are recorded.",
            "gitlinks": "Gitlinks are recorded in repositories and their tracked files are collected recursively.",
            "lfs_payloads": "LFS payload bytes are not source hashes; index pointer identity is recorded in lfs_pointers.",
            "ignored_untracked": "Git-ignored untracked files are excluded. Runtime-required source must be explicitly tracked.",
        },
    }


def refresh(args):
    root = Path(args.root).resolve()
    requested_output = Path(args.output)
    output = requested_output if requested_output.is_absolute() else root / requested_output
    if output.is_symlink():
        raise ManifestError("Refusing a symlink output: {}".format(output))
    output = output.resolve()
    relative_path(output, root)
    if output.suffix != ".json":
        raise ManifestError("Manifest output must be a .json file inside the release root")
    manifest = build_manifest(root, output)
    encoded = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".release-manifest-", suffix=".tmp", dir=str(output.parent), delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        os.replace(str(temporary), str(output))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(
        "Wrote {}: {} source files, {} submodules, {} LFS pointers".format(
            output, len(manifest["files"]), len(manifest["repositories"]), len(manifest["lfs_pointers"])
        )
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("refresh", help="Hash indexed source selection and clean pinned dependencies")
    command.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Release repository root (default: parent of scripts/)")
    command.add_argument("--output", default="release_manifest.json", help="Output JSON inside release root (default: release_manifest.json)")
    args = parser.parse_args(argv)
    try:
        return refresh(args)
    except (ManifestError, OSError) as error:
        print("release_manifest: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
