#!/usr/bin/env python3
"""Read-only gate for a validated candidate before a manual fast-forward merge.

Run in the candidate worktree, not in the main worktree:
  python scripts/check_merge.py --reports SOURCE.json CPU.json GPU.json
For a documented documentation-only change:
  python scripts/check_merge.py --reports SOURCE.json --require source

Reports and release_manifest.json are read without modification. The gate never
checks out a branch, stages files, merges, pushes, or runs reported commands.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROFILES = ("source", "cpu", "gpu")


class GateError(RuntimeError):
    pass


def git(repo, *args):
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(["git", "-C", str(repo)] + list(args), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GateError("git %s in %s failed: %s" % (" ".join(args), repo, detail[:1500]))
    return result.stdout


def dirty_entries(repo):
    records = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all",
                  "--ignore-submodules=none").split(b"\0")
    entries = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        value = record.decode("utf-8", errors="replace")
        entry = {"status": value[:2], "path": value[3:]}
        if any(flag in value[:2] for flag in "RC") and index < len(records):
            entry["original_path"] = records[index].decode("utf-8", errors="replace")
            index += 1
        entries.append(entry)
    return entries


def inspect_repositories(root, errors):
    """Walk HEAD gitlinks recursively, including uninitialized/mismatched ones."""
    repositories = {}
    seen = set()

    def inspect(path, expected_head=None):
        resolved = path.resolve()
        try:
            key = resolved.relative_to(root).as_posix()
        except ValueError:
            errors.append("Repository path escapes release root: %s" % path)
            return
        if resolved in seen:
            errors.append("Duplicate/cyclic repository path: %s" % key)
            return
        seen.add(resolved)
        try:
            top = Path(git(resolved, "rev-parse", "--show-toplevel").decode().strip()).resolve()
            if top != resolved:
                raise GateError("Expected an initialized repository at %s, found parent repository %s" % (resolved, top))
            head = git(resolved, "rev-parse", "HEAD").decode().strip()
            dirty = dirty_entries(resolved)
            repositories[key] = {"head": head, "dirty_count": len(dirty), "dirty": dirty[:50]}
            if dirty:
                errors.append("Repository is dirty: %s (%d changes)" % (key, len(dirty)))
            if expected_head is not None and head != expected_head:
                errors.append("Submodule %s HEAD %s differs from parent gitlink %s" % (key, head, expected_head))
            for record in git(resolved, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
                if not record:
                    continue
                metadata, name = record.split(b"\t", 1)
                mode, kind, commit = metadata.split()
                if mode == b"160000":
                    inspect(resolved / os.fsdecode(name), commit.decode("ascii"))
        except (GateError, OSError, ValueError) as exc:
            errors.append("Repository %s: %s" % (key, exc))

    inspect(root)
    return repositories


def repository_map(value, root):
    if not isinstance(value, dict):
        raise GateError("repositories must be an object mapping paths to heads")
    normalized = {}
    for name, record in value.items():
        if not isinstance(name, str) or not name:
            raise GateError("Repository keys must be nonempty paths")
        path = Path(name)
        path = (path if path.is_absolute() else root / path).resolve()
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            raise GateError("Reported repository lies outside release root: %s" % name)
        head = record.get("head") if isinstance(record, dict) else record
        if not isinstance(head, str) or not head:
            raise GateError("Repository %s has no head" % name)
        if key in normalized:
            raise GateError("Duplicate normalized repository path: %s" % key)
        normalized[key] = head
    return normalized


def executed_check(check):
    command = check.get("command")
    valid_command = ((isinstance(command, list) and bool(command)
                      and all(isinstance(part, str) and part for part in command))
                     or (isinstance(command, str) and bool(command.strip())))
    return valid_command and type(check.get("returncode")) is int and check["returncode"] == 0


def inspect_report(path, root, current, manifest_hash):
    failures = []
    result = {"path": str(path), "profile": None, "status": "failed", "errors": failures}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise GateError("Report must be a JSON object")
        profile = report.get("profile")
        result["profile"] = profile
        if report.get("schema_version") != 1:
            failures.append("Unsupported or missing schema_version (expected 1)")
        if report.get("status") != "passed":
            failures.append("Report status is not passed")
        if profile not in PROFILES:
            failures.append("Unsupported or missing report profile")
        if report.get("root_head") != current.get(".", {}).get("head"):
            failures.append("Report root_head does not match candidate HEAD")
        if not manifest_hash or report.get("code_manifest_sha256") != manifest_hash:
            failures.append("Report code_manifest_sha256 does not match current release_manifest.json")
        expected = {key: value["head"] for key, value in current.items() if key != "."}
        try:
            reported = repository_map(report.get("repositories"), root)
            if "." in reported:
                if reported.pop(".") != current.get(".", {}).get("head"):
                    failures.append("Reported root repository head is stale")
            missing = sorted(set(expected) - set(reported))
            extra = sorted(set(reported) - set(expected))
            if missing:
                failures.append("Report omits recursive repositories: %s" % ", ".join(missing))
            if extra:
                failures.append("Report contains unknown repositories: %s" % ", ".join(extra))
            for key in sorted(set(expected) & set(reported)):
                if reported[key] != expected[key]:
                    failures.append("Reported repository head is stale: %s" % key)
        except GateError as exc:
            failures.append(str(exc))
        checks = report.get("checks")
        if not isinstance(checks, list):
            failures.append("Report checks must be a list")
        else:
            if not checks:
                failures.append("Every validation profile requires at least one nonempty check")
            for index, check in enumerate(checks):
                if not isinstance(check, dict):
                    failures.append("Check %d is not an object" % index)
                    continue
                label = check.get("name", "check_%d" % index)
                if check.get("status") != "passed":
                    failures.append("Check %s is failed, missing, or not passed" % label)
                if "returncode" in check and (type(check["returncode"]) is not int or check["returncode"] != 0):
                    failures.append("Check %s has a nonzero or invalid returncode" % label)
            if profile in ("cpu", "gpu") and not any(executed_check(check) for check in checks if isinstance(check, dict)):
                failures.append("CPU/GPU report has no successful executed check with command and returncode 0")
    except (OSError, ValueError, GateError) as exc:
        failures.append(str(exc))
    result["status"] = "failed" if failures else "passed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reports", type=Path, nargs="+", required=True,
                        help="validation reports, resolved relative to the invoking working directory")
    parser.add_argument("--require", choices=PROFILES, nargs="+", default=list(PROFILES))
    args = parser.parse_args(argv)
    root = args.release_root.expanduser().resolve()
    errors = []
    repositories = inspect_repositories(root, errors)
    manifest_hash = None
    try:
        manifest_hash = hashlib.sha256((root / "release_manifest.json").read_bytes()).hexdigest()
    except OSError as exc:
        errors.append("Cannot read release_manifest.json: %s" % exc)
    reports = [inspect_report(path.expanduser().resolve(), root, repositories, manifest_hash) for path in args.reports]
    covered = sorted({report["profile"] for report in reports if report["status"] == "passed"})
    required = list(dict.fromkeys(args.require))
    missing = sorted(set(required) - set(covered))
    if missing:
        errors.append("Missing passed validation profiles: %s" % ", ".join(missing))
    if any(report["status"] != "passed" for report in reports):
        errors.append("One or more supplied reports failed candidate validation")
    candidate = repositories.get(".", {}).get("head")
    passed = not errors and candidate is not None
    result = {
        "schema_version": 1, "status": "passed" if passed else "failed",
        "release_root": str(root), "candidate_head": candidate,
        "code_manifest_sha256": manifest_hash, "required_profiles": required,
        "covered_profiles": covered, "repositories": repositories,
        "reports": reports, "errors": errors,
        "merge_candidate_head": candidate if passed else None,
        "manual_next_step": ("In a separately verified clean main checkout, run git merge --ff-only %s. This gate did not merge or change branches." % candidate)
                            if passed else "Do not merge this candidate; resolve the failed checks and regenerate its reports.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
